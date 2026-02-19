"""
Posterior optimization for latent variable models.

This module contains the PosteriorOptimizer dispatcher and inference strategy classes
for optimizing latent variables using variational inference.

Strategies:
    1. AdamVI        — Standard AdamW (baseline)
    2. MetaLearned   — Meta-learned deep momentum with outer-loop learnable MLP
    3. DeltaMomentum — Delta-rule momentum with gradient-dependent forgetting
    4. PreconditionedManifold — Diagonal Shampoo or learned low-rank preconditioner
    5. UnderdampedLangevin   — Stochastic Langevin dynamics with optional learned schedule
"""

import torch
import math
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Base Strategy
# ---------------------------------------------------------------------------

class BaseInferenceStrategy:
    """Base class for all posterior inference strategies."""

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def step(self, data, ctx, scaler, steps, seed, lr):
        raise NotImplementedError

    def get_learnable_parameters(self):
        """Return list of (name, Parameter) for the outer optimizer."""
        return []

    def get_learnable_modules(self):
        """Return list of (name, nn.Module) for DDP registration."""
        return []

    # -- shared helpers -----------------------------------------------------

    def get_fast_lr(self, it: int) -> float:
        fast_lr = self.kwargs.get("lr", 1e-1)
        min_fast_lr = fast_lr / 10
        num_steps = self.kwargs.get("num_steps", 10)
        if it < num_steps:
            decay_ratio = it / num_steps
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_fast_lr + coeff * (fast_lr - min_fast_lr)
        return min_fast_lr

    def _init_latents(self, X, Z, max_z_len, z_dim, persistent_init, const_var):
        _bsz = X.shape[0]
        with torch.no_grad():
            if Z is None:
                mu = torch.zeros(_bsz, max_z_len, z_dim, device=X.device)
                log_var = (torch.randn_like(mu) * 0.1 - 5.0) if not const_var else (torch.zeros_like(mu) - 5.0)
            else:
                mu = Z.clone() if persistent_init else torch.zeros_like(Z)
                log_var = (torch.randn_like(mu) * 0.1 - 5.0) if not const_var else (torch.zeros_like(mu) - 5.0)
            mu = mu.view(_bsz, max_z_len, z_dim)
            log_var = log_var.view(_bsz, max_z_len, z_dim)
        return mu, log_var

    def _finalize(self, mu, log_var, e, const_var, model, X, Y, ctx):
        """Final sampling + metrics.  Detaches z (for non-learnable strategies)."""
        with torch.no_grad():
            std = torch.exp(0.5 * log_var)
            if const_var:
                z = mu
                log_var = torch.zeros_like(log_var) - 5.0
            else:
                z = mu + e * std
        with ctx:
            loss, ppl, h, kl_loss, nlkhd = model.elbo(X, mu, log_var, e, Y, None, eval_mode=True)
        return z.detach(), ppl.detach(), kl_loss.detach(), nlkhd.detach()

    def _finalize_learnable(self, mu, log_var, e, const_var, model, X, Y, ctx):
        """Final sampling + metrics.  Keeps z in the computation graph."""
        std = torch.exp(0.5 * log_var)
        if const_var:
            z = mu
        else:
            z = mu + e * std
        with torch.no_grad():
            with ctx:
                _, ppl, _, kl_loss, nlkhd = model.elbo(
                    X, mu.detach(), log_var.detach(), e, Y, None, eval_mode=True
                )
        return z, ppl.detach(), kl_loss.detach(), nlkhd.detach()

    def _unpack_common(self, steps, lr):
        lr = lr if lr is not None else self.kwargs.get("lr", 1e-1)
        num_steps = self.kwargs.get("num_steps", 10) if steps is None else steps
        max_z_len = self.kwargs.get("max_z_len", 1)
        z_dim = self.kwargs.get("z_dim", 288)
        const_var = self.kwargs.get("const_var", False)
        eval_mode = self.kwargs.get("eval_mode", True)
        persistent_init = self.kwargs.get("persistent_init", True)
        return lr, num_steps, max_z_len, z_dim, const_var, eval_mode, persistent_init


# ---------------------------------------------------------------------------
# Strategy 1: AdamVI  (existing baseline, unchanged behaviour)
# ---------------------------------------------------------------------------

class AdamVIStrategy(BaseInferenceStrategy):

    def step(self, data, ctx, scaler, steps, seed, lr):
        lr, num_steps, max_z_len, z_dim, const_var, eval_mode, persistent_init = \
            self._unpack_common(steps, lr)
        betas = self.kwargs.get("betas", (0.9, 0.999))
        eps = self.kwargs.get("eps", 1e-8)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self.model.eval()
        X, Y, Z = data
        mu, log_var = self._init_latents(X, Z, max_z_len, z_dim, persistent_init, const_var)

        mu.requires_grad_()
        if not const_var:
            log_var.requires_grad_()

        optimizer = torch.optim.AdamW([mu, log_var], lr=lr, betas=betas, eps=eps)
        h = None
        e = torch.randn_like(log_var)

        for s in range(num_steps):
            current_fast_lr = self.get_fast_lr(s)
            for pg in optimizer.param_groups:
                pg['lr'] = current_fast_lr

            optimizer.zero_grad(set_to_none=True)
            with ctx:
                loss, _, h, _, _ = self.model.elbo(X, mu, log_var, e, Y, h, eval_mode=eval_mode)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            h = None

        return self._finalize(mu, log_var, e, const_var, self.model, X, Y, ctx)


# ---------------------------------------------------------------------------
# Strategy 2: Meta-Learned Associative Pen  (Deep Momentum)
# ---------------------------------------------------------------------------

class MetaLearnedStrategy(BaseInferenceStrategy):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        z_dim = kwargs.get("z_dim", 768)
        hidden_dim = kwargs.get("meta_hidden_dim", 128)
        self.bptt_depth = kwargs.get("meta_bptt_depth", 4)

        from inference_modules import MetaOptimizerModule
        self.phi = MetaOptimizerModule(z_dim=z_dim, hidden_dim=hidden_dim)
        self._device_set = False

    def get_learnable_parameters(self):
        return list(self.phi.named_parameters())

    def get_learnable_modules(self):
        return [("meta_optimizer", self.phi)]

    def step(self, data, ctx, scaler, steps, seed, lr):
        if not self._device_set:
            device = next(self.model.parameters()).device
            self.phi = self.phi.to(device)
            self._device_set = True

        lr, num_steps, max_z_len, z_dim, const_var, eval_mode, persistent_init = \
            self._unpack_common(steps, lr)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        if eval_mode:
            self.model.eval()

        X, Y, Z = data
        _bsz = X.shape[0]
        mu, log_var = self._init_latents(X, Z, max_z_len, z_dim, persistent_init, const_var)

        mu = mu.detach().requires_grad_(True)
        if not const_var:
            log_var.requires_grad_(True)
        e = torch.randn_like(log_var)
        h = None

        s = torch.zeros(_bsz, max_z_len, self.phi.hidden_dim, device=X.device)

        for step_idx in range(num_steps):
            current_lr = self.get_fast_lr(step_idx)
            retain_graph = (not eval_mode) and (step_idx >= num_steps - self.bptt_depth)

            with ctx:
                loss, _, h, _, _ = self.model.elbo(X, mu, log_var, e, Y, h, eval_mode=eval_mode)

            grad_mu = torch.autograd.grad(loss, mu, create_graph=retain_graph)[0]

            if retain_graph:
                delta_z, s = self.phi(grad_mu, s)
                mu = mu - current_lr * delta_z
            else:
                with torch.no_grad():
                    delta_z, s = self.phi(grad_mu.detach(), s.detach())
                    mu = (mu - current_lr * delta_z).detach().requires_grad_(True)
                    s = s.detach()

            h = None

        if eval_mode:
            return self._finalize(mu.detach(), log_var.detach(), e, const_var, self.model, X, Y, ctx)
        else:
            return self._finalize_learnable(mu, log_var.detach(), e, const_var, self.model, X, Y, ctx)


# ---------------------------------------------------------------------------
# Strategy 3: Delta Momentum  (Gradient-Dependent Forgetting)
# ---------------------------------------------------------------------------

class DeltaMomentumStrategy(BaseInferenceStrategy):

    def step(self, data, ctx, scaler, steps, seed, lr):
        lr, num_steps, max_z_len, z_dim, const_var, eval_mode, persistent_init = \
            self._unpack_common(steps, lr)
        alpha = self.kwargs.get("delta_alpha", 0.9)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self.model.eval()
        X, Y, Z = data
        mu, log_var = self._init_latents(X, Z, max_z_len, z_dim, persistent_init, const_var)

        mu.requires_grad_(True)
        if not const_var:
            log_var.requires_grad_(True)
        e = torch.randn_like(log_var)
        h = None
        m = torch.zeros_like(mu)

        for step_idx in range(num_steps):
            current_lr = self.get_fast_lr(step_idx)

            if mu.grad is not None:
                mu.grad.zero_()
            if not const_var and log_var.grad is not None:
                log_var.grad.zero_()

            with ctx:
                loss, _, h, _, _ = self.model.elbo(X, mu, log_var, e, Y, h, eval_mode=eval_mode)

            if scaler is not None:
                scaler.scale(loss).backward()
                inv_scale = 1.0 / scaler.get_scale()
                g = mu.grad.data * inv_scale
            else:
                loss.backward()
                g = mu.grad.data.clone()

            # Delta rule: erase momentum component aligned with current gradient
            g_norm_sq = (g * g).sum(dim=-1, keepdim=True).clamp(min=1e-8)
            proj = ((m * g).sum(dim=-1, keepdim=True) / g_norm_sq) * g
            m = m + alpha * (g - proj)

            with torch.no_grad():
                mu.data -= current_lr * m

            if not const_var and log_var.grad is not None:
                with torch.no_grad():
                    lv_grad = log_var.grad.data
                    if scaler is not None:
                        lv_grad = lv_grad * inv_scale
                    log_var.data -= current_lr * lv_grad

            if scaler is not None:
                scaler.update()

            h = None

        return self._finalize(mu, log_var, e, const_var, self.model, X, Y, ctx)


# ---------------------------------------------------------------------------
# Strategy 4: Preconditioned Manifold  (Shampoo / Learned)
# ---------------------------------------------------------------------------

class PreconditionedManifoldStrategy(BaseInferenceStrategy):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.precond_type = kwargs.get("precond_type", "learned")

        if self.precond_type == "learned":
            z_dim = kwargs.get("z_dim", 768)
            rank = kwargs.get("precond_rank", 32)
            self.bptt_depth = kwargs.get("meta_bptt_depth", 4)
            from inference_modules import LearnedPreconditioner
            self.preconditioner = LearnedPreconditioner(z_dim, rank)
            self._device_set = False

    def get_learnable_parameters(self):
        if self.precond_type == "learned":
            return list(self.preconditioner.named_parameters())
        return []

    def get_learnable_modules(self):
        if self.precond_type == "learned":
            return [("preconditioner", self.preconditioner)]
        return []

    def step(self, data, ctx, scaler, steps, seed, lr):
        if self.precond_type == "learned":
            return self._learned_step(data, ctx, scaler, steps, seed, lr)
        else:
            return self._shampoo_step(data, ctx, scaler, steps, seed, lr)

    # -- Option A: diagonal Shampoo ----------------------------------------

    def _shampoo_step(self, data, ctx, scaler, steps, seed, lr):
        lr, num_steps, max_z_len, z_dim, const_var, eval_mode, persistent_init = \
            self._unpack_common(steps, lr)
        beta = self.kwargs.get("precond_beta", 0.9)
        eps_precond = 1e-4

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self.model.eval()
        X, Y, Z = data
        mu, log_var = self._init_latents(X, Z, max_z_len, z_dim, persistent_init, const_var)

        mu.requires_grad_(True)
        if not const_var:
            log_var.requires_grad_(True)
        e = torch.randn_like(log_var)
        h = None

        diag_G = torch.zeros_like(mu)

        for step_idx in range(num_steps):
            current_lr = self.get_fast_lr(step_idx)

            if mu.grad is not None:
                mu.grad.zero_()
            if not const_var and log_var.grad is not None:
                log_var.grad.zero_()

            with ctx:
                loss, _, h, _, _ = self.model.elbo(X, mu, log_var, e, Y, h, eval_mode=eval_mode)

            if scaler is not None:
                scaler.scale(loss).backward()
                inv_scale = 1.0 / scaler.get_scale()
                g = mu.grad.data * inv_scale
            else:
                loss.backward()
                g = mu.grad.data.clone()

            # Accumulate diagonal second-moment estimate
            diag_G = beta * diag_G + (1 - beta) * g * g
            precond_g = g / (torch.sqrt(diag_G + eps_precond))

            with torch.no_grad():
                mu.data -= current_lr * precond_g

            if not const_var and log_var.grad is not None:
                with torch.no_grad():
                    lv_grad = log_var.grad.data
                    if scaler is not None:
                        lv_grad = lv_grad * inv_scale
                    log_var.data -= current_lr * lv_grad

            if scaler is not None:
                scaler.update()

            h = None

        return self._finalize(mu, log_var, e, const_var, self.model, X, Y, ctx)

    # -- Option B: learned low-rank preconditioner --------------------------

    def _learned_step(self, data, ctx, scaler, steps, seed, lr):
        if not self._device_set:
            device = next(self.model.parameters()).device
            self.preconditioner = self.preconditioner.to(device)
            self._device_set = True

        lr, num_steps, max_z_len, z_dim, const_var, eval_mode, persistent_init = \
            self._unpack_common(steps, lr)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        if eval_mode:
            self.model.eval()

        X, Y, Z = data
        mu, log_var = self._init_latents(X, Z, max_z_len, z_dim, persistent_init, const_var)

        mu = mu.detach().requires_grad_(True)
        if not const_var:
            log_var.requires_grad_(True)
        e = torch.randn_like(log_var)
        h = None

        for step_idx in range(num_steps):
            current_lr = self.get_fast_lr(step_idx)
            retain_graph = (not eval_mode) and (step_idx >= num_steps - self.bptt_depth)

            with ctx:
                loss, _, h, _, _ = self.model.elbo(X, mu, log_var, e, Y, h, eval_mode=eval_mode)

            grad_mu = torch.autograd.grad(loss, mu, create_graph=retain_graph)[0]

            if retain_graph:
                precond_grad = self.preconditioner(grad_mu)
                mu = mu - current_lr * precond_grad
            else:
                with torch.no_grad():
                    precond_grad = self.preconditioner(grad_mu.detach())
                    mu = (mu - current_lr * precond_grad).detach().requires_grad_(True)

            h = None

        if eval_mode:
            return self._finalize(mu.detach(), log_var.detach(), e, const_var, self.model, X, Y, ctx)
        else:
            return self._finalize_learnable(mu, log_var.detach(), e, const_var, self.model, X, Y, ctx)


# ---------------------------------------------------------------------------
# Strategy 5: Underdamped Langevin Dynamics
# ---------------------------------------------------------------------------

class UnderdampedLangevinStrategy(BaseInferenceStrategy):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.learn_schedule = kwargs.get("langevin_learn_schedule", False)
        num_steps = kwargs.get("num_steps", 16)

        if self.learn_schedule:
            self.bptt_depth = kwargs.get("meta_bptt_depth", 4)
            from inference_modules import LangevinSchedule
            self.schedule = LangevinSchedule(num_steps)
            self._device_set = False
        else:
            self.schedule = None
            self.gamma = kwargs.get("langevin_gamma", 0.1)
            self.sigma = kwargs.get("langevin_sigma", 0.01)

    def get_learnable_parameters(self):
        if self.schedule is not None:
            return list(self.schedule.named_parameters())
        return []

    def get_learnable_modules(self):
        if self.schedule is not None:
            return [("langevin_schedule", self.schedule)]
        return []

    def step(self, data, ctx, scaler, steps, seed, lr):
        if self.learn_schedule:
            return self._learned_step(data, ctx, scaler, steps, seed, lr)
        else:
            return self._fixed_step(data, ctx, scaler, steps, seed, lr)

    # -- fixed gamma/sigma --------------------------------------------------

    def _fixed_step(self, data, ctx, scaler, steps, seed, lr):
        lr, num_steps, max_z_len, z_dim, const_var, eval_mode, persistent_init = \
            self._unpack_common(steps, lr)
        gamma = self.gamma
        sigma = self.sigma

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self.model.eval()
        X, Y, Z = data
        mu, log_var = self._init_latents(X, Z, max_z_len, z_dim, persistent_init, const_var)

        mu.requires_grad_(True)
        if not const_var:
            log_var.requires_grad_(True)
        e = torch.randn_like(log_var)
        h = None
        v = torch.zeros_like(mu)

        for step_idx in range(num_steps):
            current_lr = self.get_fast_lr(step_idx)

            if mu.grad is not None:
                mu.grad.zero_()
            if not const_var and log_var.grad is not None:
                log_var.grad.zero_()

            with ctx:
                loss, _, h, _, _ = self.model.elbo(X, mu, log_var, e, Y, h, eval_mode=eval_mode)

            if scaler is not None:
                scaler.scale(loss).backward()
                inv_scale = 1.0 / scaler.get_scale()
                g = mu.grad.data * inv_scale
            else:
                loss.backward()
                g = mu.grad.data.clone()

            noise = torch.zeros_like(mu) if eval_mode else torch.randn_like(mu)
            v = (1 - gamma) * v - current_lr * g + sigma * math.sqrt(2 * gamma * current_lr) * noise

            with torch.no_grad():
                mu.data += v

            if not const_var and log_var.grad is not None:
                with torch.no_grad():
                    lv_grad = log_var.grad.data
                    if scaler is not None:
                        lv_grad = lv_grad * inv_scale
                    log_var.data -= current_lr * lv_grad

            if scaler is not None:
                scaler.update()

            h = None

        return self._finalize(mu, log_var, e, const_var, self.model, X, Y, ctx)

    # -- learned schedule ---------------------------------------------------

    def _learned_step(self, data, ctx, scaler, steps, seed, lr):
        if not self._device_set:
            device = next(self.model.parameters()).device
            self.schedule = self.schedule.to(device)
            self._device_set = True

        lr, num_steps, max_z_len, z_dim, const_var, eval_mode, persistent_init = \
            self._unpack_common(steps, lr)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        if eval_mode:
            self.model.eval()

        X, Y, Z = data
        mu, log_var = self._init_latents(X, Z, max_z_len, z_dim, persistent_init, const_var)

        mu = mu.detach().requires_grad_(True)
        if not const_var:
            log_var.requires_grad_(True)
        e = torch.randn_like(log_var)
        h = None
        v = torch.zeros_like(mu)

        for step_idx in range(num_steps):
            current_lr = self.get_fast_lr(step_idx)
            retain_graph = (not eval_mode) and (step_idx >= num_steps - self.bptt_depth)

            gamma, sigma = self.schedule(step_idx)

            with ctx:
                loss, _, h, _, _ = self.model.elbo(X, mu, log_var, e, Y, h, eval_mode=eval_mode)

            grad_mu = torch.autograd.grad(loss, mu, create_graph=retain_graph)[0]

            noise = torch.zeros_like(mu) if eval_mode else torch.randn_like(mu)

            if retain_graph:
                v = (1 - gamma) * v - current_lr * grad_mu + sigma * math.sqrt(2 * current_lr) * torch.sqrt(gamma) * noise
                mu = mu + v
            else:
                with torch.no_grad():
                    gamma_val = gamma.item() if isinstance(gamma, torch.Tensor) else gamma
                    sigma_val = sigma.item() if isinstance(sigma, torch.Tensor) else sigma
                    v = (1 - gamma_val) * v - current_lr * grad_mu.detach() + sigma_val * math.sqrt(2 * gamma_val * current_lr) * noise
                    mu = (mu + v).detach().requires_grad_(True)
                    v = v.detach()

            h = None

        if eval_mode:
            return self._finalize(mu.detach(), log_var.detach(), e, const_var, self.model, X, Y, ctx)
        else:
            return self._finalize_learnable(mu, log_var.detach(), e, const_var, self.model, X, Y, ctx)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_STRATEGIES = {
    "adamVI": AdamVIStrategy,
    "meta_learned": MetaLearnedStrategy,
    "delta_momentum": DeltaMomentumStrategy,
    "preconditioned_manifold": PreconditionedManifoldStrategy,
    "underdamped_langevin": UnderdampedLangevinStrategy,
}


class PosteriorOptimizer:
    def __init__(self, model, inference_method="adamVI", **kwargs):
        self.model = model
        self.inference_method = inference_method
        self.kwargs = kwargs
        print("Optimizer kwargs", self.kwargs)

        if inference_method not in _STRATEGIES:
            raise ValueError(
                f"Unknown inference method: {inference_method}. "
                f"Options: {list(_STRATEGIES.keys())}"
            )
        self._strategy = _STRATEGIES[inference_method](model, **kwargs)

    def step(self, data: List, ctx, scaler=None,
             steps=None, seed=None, lr=None) -> Tuple:
        return self._strategy.step(data, ctx, scaler, steps, seed=seed, lr=lr)

    def get_learnable_parameters(self):
        return self._strategy.get_learnable_parameters()

    def get_learnable_modules(self):
        return self._strategy.get_learnable_modules()

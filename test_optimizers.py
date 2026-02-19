"""
Test sweep for all posterior optimizer strategies.

Creates a tiny model + synthetic data and runs each optimizer configuration,
checking for crashes, correct output shapes, NaN/Inf, and gradient flow.

Usage:
    python test_optimizers.py          # auto-detect CPU/CUDA
    python test_optimizers.py --cpu    # force CPU
"""

import sys
import types
import argparse

# ---------------------------------------------------------------------------
# Stub out liger_kernel so the model loads on CPU without CUDA deps.
# Must be done BEFORE importing model.py / liger_module.py.
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn

class _StubRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class _StubSwiGLUMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, multiple_of=32, dropout=0.0, **kwargs):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class _StubCELoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=-1, **kwargs):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
    def forward(self, inp, target):
        return nn.functional.cross_entropy(inp, target, ignore_index=self.ignore_index, reduction=self.reduction)

# Build a fake liger_kernel package hierarchy so imports succeed
_liger_kernel = types.ModuleType("liger_kernel")
_liger_ops = types.ModuleType("liger_kernel.ops")
for sub in ("rms_norm", "swiglu", "rope", "cross_entropy", "layer_norm"):
    m = types.ModuleType(f"liger_kernel.ops.{sub}")
    setattr(_liger_ops, sub, m)
    sys.modules[f"liger_kernel.ops.{sub}"] = m
sys.modules["liger_kernel"] = _liger_kernel
sys.modules["liger_kernel.ops"] = _liger_ops

# Now build a fake liger_module that the model will import
_liger_module = types.ModuleType("liger_module")
_liger_module.LigerRMSNorm = _StubRMSNorm
_liger_module.LigerSwiGLUMLP = _StubSwiGLUMLP
_liger_module.LigerCrossEntropyLoss = _StubCELoss
_liger_module.LigerLayerNorm = _StubRMSNorm
def _stub_rope(*a, **kw):
    raise NotImplementedError("liger rope stub")
_liger_module.liger_rotary_pos_emb = _stub_rope
sys.modules["liger_module"] = _liger_module

# ---------------------------------------------------------------------------
# Now safe to import project modules
# ---------------------------------------------------------------------------

from contextlib import nullcontext
from model import LatentThoughtModel, LTMConfig, Attention
from optimizer import PosteriorOptimizer

# On CPU, SDPA flash kernel doesn't support create_graph=True (needed for BPTT).
# Disable it so all attention falls back to the math kernel.
USE_CPU = not torch.cuda.is_available()
if USE_CPU:
    torch.backends.cuda.enable_flash_sdp(False) if hasattr(torch.backends.cuda, 'enable_flash_sdp') else None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY_DIM = 64
TINY_LAYERS = 2
TINY_HEADS = 2
TINY_SEQ = 32
TINY_VOCAB = 256
TINY_Z_LEN = TINY_LAYERS * 8  # 16
TINY_WINDOW = 16
BSZ = 2
NUM_STEPS = 4


def create_tiny_model(device):
    cfg = LTMConfig(
        dim=TINY_DIM,
        n_layers=TINY_LAYERS,
        n_heads=TINY_HEADS,
        n_kv_heads=TINY_HEADS,
        vocab_size=TINY_VOCAB,
        multiple_of=32,
        max_seq_len=TINY_SEQ,
        dropout=0.0,
        window_size=TINY_WINDOW,
        use_liger=False,
        max_z_len=TINY_Z_LEN,
        use_z_pos_emb=True,
    )
    model = LatentThoughtModel(cfg)
    # On CPU, flash SDPA doesn't support create_graph=True (needed for BPTT).
    # Fall back to manual math-mode attention.
    if USE_CPU:
        for m in model.modules():
            if isinstance(m, Attention):
                m.flash = False
    model.to(device)
    return model


def make_fake_batch(device):
    X = torch.randint(0, TINY_VOCAB, (BSZ, TINY_SEQ), device=device)
    Y = torch.randint(0, TINY_VOCAB, (BSZ, TINY_SEQ), device=device)
    Z = None
    return X, Y, Z


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

# has_learnable is a property of the STRATEGY, not the eval_mode.
# test_grad_flow is only meaningful when eval_mode=False and the strategy is learnable.
TEST_CONFIGS = [
    # (name, inference_method, extra_kwargs, eval_mode, has_learnable, test_grad_flow)
    ("adamVI",                       "adamVI",                  {},                                          True,  False, False),
    ("meta_learned (eval)",          "meta_learned",            {"meta_hidden_dim": 32, "meta_bptt_depth": 2}, True,  True,  False),
    ("meta_learned (train+BPTT)",    "meta_learned",            {"meta_hidden_dim": 32, "meta_bptt_depth": 2}, False, True,  True),
    ("delta_momentum",               "delta_momentum",          {"delta_alpha": 0.9},                        True,  False, False),
    ("precond_shampoo",              "preconditioned_manifold", {"precond_type": "shampoo", "precond_beta": 0.9}, True, False, False),
    ("precond_learned (eval)",       "preconditioned_manifold", {"precond_type": "learned", "precond_rank": 8, "meta_bptt_depth": 2}, True, True, False),
    ("precond_learned (train+BPTT)", "preconditioned_manifold", {"precond_type": "learned", "precond_rank": 8, "meta_bptt_depth": 2}, False, True, True),
    ("langevin_fixed",               "underdamped_langevin",    {"langevin_gamma": 0.1, "langevin_sigma": 0.01, "langevin_learn_schedule": False}, True, False, False),
    ("langevin_learned (eval)",      "underdamped_langevin",    {"langevin_gamma": 0.1, "langevin_sigma": 0.01, "langevin_learn_schedule": True, "meta_bptt_depth": 2}, True, True, False),
    ("langevin_learned (train+BPTT)","underdamped_langevin",    {"langevin_gamma": 0.1, "langevin_sigma": 0.01, "langevin_learn_schedule": True, "meta_bptt_depth": 2}, False, True, True),
]


def run_test(name, method, extra_kwargs, eval_mode, has_learnable, test_grad_flow, device):
    """Run a single optimizer test.  Returns (passed: bool, message: str)."""
    errors = []

    try:
        # Use a fixed global seed for reproducible model init and input data
        torch.manual_seed(0)
        model = create_tiny_model(device)
        ctx = nullcontext()

        kwargs = dict(
            num_steps=NUM_STEPS,
            max_z_len=TINY_Z_LEN,
            z_dim=TINY_DIM,
            lr=0.1,
            eval_mode=eval_mode,
            **extra_kwargs,
        )

        opt = PosteriorOptimizer(model=model, inference_method=method, **kwargs)

        # -- Check learnable parameters --
        lp = opt.get_learnable_parameters()
        if has_learnable and len(lp) == 0:
            errors.append("Expected learnable params but got none")
        if not has_learnable and len(lp) > 0:
            errors.append(f"Expected no learnable params but got {len(lp)}")

        # -- Run step --
        torch.manual_seed(99)
        X, Y, Z = make_fake_batch(device)
        z, ppl, kl, nlkhd = opt.step(
            data=[X, Y, Z], ctx=ctx, scaler=None, steps=NUM_STEPS, seed=42, lr=0.1
        )

        # -- Check output shapes --
        expected_z_shape = (BSZ, TINY_Z_LEN, TINY_DIM)
        if z.shape != expected_z_shape:
            errors.append(f"z shape {z.shape} != expected {expected_z_shape}")
        for tensor_name, t in [("ppl", ppl), ("kl", kl), ("nlkhd", nlkhd)]:
            if t.dim() != 0:
                errors.append(f"{tensor_name} should be scalar, got shape {t.shape}")

        # -- Check NaN / Inf --
        if torch.isnan(z).any():
            errors.append("z contains NaN")
        if torch.isinf(z).any():
            errors.append("z contains Inf")
        for tensor_name, t in [("ppl", ppl), ("kl", kl), ("nlkhd", nlkhd)]:
            if torch.isnan(t) or torch.isinf(t):
                errors.append(f"{tensor_name} is NaN or Inf: {t.item()}")

        # -- Determinism: re-run with same model weights & same data & same seed --
        torch.manual_seed(0)
        model2 = create_tiny_model(device)
        opt2 = PosteriorOptimizer(model=model2, inference_method=method, **kwargs)
        # Copy learnable params from opt to opt2
        if lp:
            lp2_dict = dict(opt2.get_learnable_parameters())
            for pname, p in opt.get_learnable_parameters():
                if pname in lp2_dict:
                    lp2_dict[pname].data.copy_(p.data)

        torch.manual_seed(99)
        X2, Y2, Z2 = make_fake_batch(device)
        z2, _, _, _ = opt2.step(
            data=[X2, Y2, Z2], ctx=ctx, scaler=None, steps=NUM_STEPS, seed=42, lr=0.1
        )
        if not torch.allclose(z.detach(), z2.detach(), atol=1e-5):
            max_diff = (z.detach() - z2.detach()).abs().max().item()
            errors.append(f"Non-deterministic: max diff = {max_diff}")

        # -- Gradient flow for learnable optimizers --
        if test_grad_flow:
            torch.manual_seed(0)
            model3 = create_tiny_model(device)
            kwargs3 = dict(kwargs)
            kwargs3["eval_mode"] = False
            opt3 = PosteriorOptimizer(model=model3, inference_method=method, **kwargs3)
            lp3 = opt3.get_learnable_parameters()

            torch.manual_seed(99)
            X3, Y3, Z3 = make_fake_batch(device)
            z3, _, _, _ = opt3.step(
                data=[X3, Y3, Z3], ctx=ctx, scaler=None, steps=NUM_STEPS, seed=42, lr=0.1
            )

            if not z3.requires_grad:
                errors.append("z from learnable strategy does not require grad (detached?)")
            else:
                outer_loss = z3.sum()
                outer_loss.backward()

                grads_found = 0
                for pname, p in lp3:
                    if p.grad is not None and p.grad.abs().max().item() > 0:
                        grads_found += 1
                if grads_found == 0:
                    errors.append("No learnable params received gradients after outer backward")

    except Exception as e:
        import traceback
        errors.append(f"EXCEPTION: {e}\n{traceback.format_exc()}")

    passed = len(errors) == 0
    msg = "PASS" if passed else "FAIL: " + "; ".join(errors)
    return passed, msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    args = parser.parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    print(f"Running optimizer sweep on device: {device}\n")

    results = []
    all_passed = True

    for name, method, extra_kwargs, eval_mode, has_learnable, test_grad_flow in TEST_CONFIGS:
        passed, msg = run_test(name, method, extra_kwargs, eval_mode, has_learnable, test_grad_flow, device)
        results.append((name, passed, msg))
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            for line in msg.split("; "):
                print(f"         {line}")

    print(f"\n{'='*60}")
    passed_count = sum(1 for _, p, _ in results if p)
    total = len(results)
    print(f"Results: {passed_count}/{total} passed")

    if all_passed:
        print("All optimizer tests passed!")
        sys.exit(0)
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()

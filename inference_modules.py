"""
Neural modules for learnable posterior inference strategies.

Contains:
- MetaOptimizerModule: MLP-based meta-learned optimizer for Deep Momentum (Strategy 2)
- LearnedPreconditioner: Low-rank learned preconditioner P = I + AB^T (Strategy 4B)
- LangevinSchedule: Per-step learnable friction/temperature for Langevin dynamics (Strategy 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaOptimizerModule(nn.Module):
    """
    Neural network phi that maps (gradient, hidden_state) -> (delta_z, new_hidden_state).
    Operates per-position with shared weights across all positions in z.

    Args:
        z_dim: Dimension of latent vectors (default 768)
        hidden_dim: Dimension of the hidden memory state (default 128)
    """
    def __init__(self, z_dim=768, hidden_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.backbone = nn.Sequential(
            nn.Linear(z_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.delta_head = nn.Linear(hidden_dim, z_dim)
        self.state_head = nn.Linear(hidden_dim, hidden_dim)

        # Initialize delta_head near zero so initial updates are small
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, g, s):
        """
        Args:
            g: gradient [bsz, max_z_len, z_dim]
            s: hidden state [bsz, max_z_len, hidden_dim]
        Returns:
            delta_z: [bsz, max_z_len, z_dim]
            s_new: [bsz, max_z_len, hidden_dim]
        """
        g_norm = g / (g.norm(dim=-1, keepdim=True) + 1e-8)
        x = torch.cat([g_norm, s], dim=-1)
        h = self.backbone(x)
        delta_z = self.delta_head(h)
        s_new = self.state_head(h) + s  # residual on state
        return delta_z, s_new


class LearnedPreconditioner(nn.Module):
    """
    Low-rank preconditioner P = I + A @ B^T.
    Applies: P @ g = g + A @ (B^T @ g)

    Args:
        z_dim: Dimension of latent vectors (default 768)
        rank: Rank of the low-rank correction (default 32)
    """
    def __init__(self, z_dim=768, rank=32):
        super().__init__()
        self.A = nn.Parameter(torch.randn(z_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(z_dim, rank) * 0.01)

    def forward(self, g):
        """
        Args:
            g: gradient [bsz, max_z_len, z_dim]
        Returns:
            preconditioned gradient [bsz, max_z_len, z_dim]
        """
        Btg = torch.einsum('...d, dr -> ...r', g, self.B)
        ABtg = torch.einsum('...r, dr -> ...d', Btg, self.A)
        return g + ABtg


class LangevinSchedule(nn.Module):
    """
    Learnable per-step friction (gamma) and temperature (sigma) for
    underdamped Langevin dynamics. Uses softplus to ensure positivity.

    Args:
        num_steps: Number of inner optimization steps (default 16)
    """
    def __init__(self, num_steps=16):
        super().__init__()
        self.gamma_raw = nn.Parameter(torch.full((num_steps,), -2.0))   # softplus(-2) ≈ 0.13
        self.sigma_raw = nn.Parameter(torch.full((num_steps,), -4.0))   # softplus(-4) ≈ 0.018

    def forward(self, step_idx):
        """Return (gamma, sigma) for the given step index."""
        gamma = F.softplus(self.gamma_raw[step_idx])
        sigma = F.softplus(self.sigma_raw[step_idx])
        return gamma, sigma

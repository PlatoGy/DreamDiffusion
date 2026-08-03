import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_progress_embedding(progress, dim):
    progress = progress.reshape(-1).float()
    half = dim // 2
    if half <= 0:
        return progress[:, None]
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=progress.device, dtype=progress.dtype)
        / max(half - 1, 1)
    )
    args = progress[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


def entropy(weights, eps=1e-8):
    return -(weights * torch.log(weights + eps)).sum(dim=-1)


class DynamicBoundaryPredictor(nn.Module):
    """Predict differentiable sample-specific chunk lengths from full EEG tokens H."""

    def __init__(self, input_dim, hidden_dim=256, min_chunk_length=8, temperature=1.0):
        super().__init__()
        self.min_chunk_length = float(min_chunk_length)
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_tokens):
        if h_tokens.ndim != 3:
            raise ValueError(f"h_tokens must have shape [B,N,d], got {list(h_tokens.shape)}")
        batch, token_count, _ = h_tokens.shape
        min_total = 3.0 * self.min_chunk_length
        if token_count <= min_total:
            raise ValueError(
                f"token_count={token_count} must be larger than 3 * min_chunk_length={min_total}."
            )
        pooled = h_tokens.mean(dim=1)
        logits = self.net(pooled) / max(self.temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        lengths = self.min_chunk_length + (float(token_count) - min_total) * probs
        b0 = torch.zeros(batch, 1, device=h_tokens.device, dtype=h_tokens.dtype)
        b1 = lengths[:, :1]
        b2 = lengths[:, :2].sum(dim=-1, keepdim=True)
        b3 = torch.full((batch, 1), float(token_count), device=h_tokens.device, dtype=h_tokens.dtype)
        boundaries = torch.cat([b0, b1, b2, b3], dim=-1)
        return lengths, boundaries, logits


def soft_membership(boundaries, token_count, rho=1.0, eps=1e-8):
    """Differentiable membership m[B,3,N] using token centers 0.5,...,N-0.5.

    Boundaries are defined on the continuous interval [0, N]. A token centered at
    n + 0.5 can partially belong to neighboring chunks near a soft boundary.
    """

    if boundaries.ndim != 2 or boundaries.shape[-1] != 4:
        raise ValueError(f"boundaries must have shape [B,4], got {list(boundaries.shape)}")
    positions = torch.arange(token_count, device=boundaries.device, dtype=boundaries.dtype) + 0.5
    starts = boundaries[:, :-1].unsqueeze(-1)
    ends = boundaries[:, 1:].unsqueeze(-1)
    pos = positions.view(1, 1, token_count)
    raw = torch.sigmoid((pos - starts) / rho) - torch.sigmoid((pos - ends) / rho)
    raw = torch.clamp(raw, min=eps)
    return raw / (raw.sum(dim=1, keepdim=True) + eps)


class DiffusionTimeRouter(nn.Module):
    """Dataset-level routing trajectory G_psi(e(p)); input is progress only."""

    def __init__(self, time_dim=64, hidden_dim=256, temperature=1.0, weight_floor=0.0):
        super().__init__()
        self.time_dim = int(time_dim)
        self.temperature = float(temperature)
        self.weight_floor = float(weight_floor)
        self.net = nn.Sequential(
            nn.Linear(self.time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, progress):
        emb = sinusoidal_progress_embedding(progress, self.time_dim).to(
            device=progress.device, dtype=progress.dtype
        )
        logits = self.net(emb)
        weights = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        delta = self.weight_floor
        if delta:
            weights = delta / 3.0 + (1.0 - delta) * weights
        return weights, logits


def token_prior(weights, membership, eps=1e-8):
    pi = torch.einsum("bk,bkn->bn", weights, membership)
    return torch.clamp(pi, min=eps)


def routing_regularization(router, grid_size=250, lambda_smooth=0.0, lambda_spec=0.0, eps=1e-8, device=None):
    device = device or next(router.parameters()).device
    progress = torch.linspace(0.0, 1.0, grid_size, device=device)
    weights, _ = router(progress)
    smooth = ((weights[1:] - weights[:-1]) ** 2).sum(dim=-1).mean()
    mean_weight = weights.mean(dim=0)
    spec = entropy(weights, eps=eps).mean() - entropy(mean_weight.unsqueeze(0), eps=eps).mean()
    loss = float(lambda_smooth) * smooth + float(lambda_spec) * spec
    return loss, {
        "dynamic/L_smooth": smooth,
        "dynamic/L_spec": spec,
        "dynamic/routing_curve_entropy_mean": entropy(weights, eps=eps).mean(),
        "dynamic/routing_curve_w1_mean": weights[:, 0].mean(),
        "dynamic/routing_curve_w2_mean": weights[:, 1].mean(),
        "dynamic/routing_curve_w3_mean": weights[:, 2].mean(),
    }, weights

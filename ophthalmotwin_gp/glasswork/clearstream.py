from __future__ import annotations

import importlib.util
import math

import torch
import torch.nn.functional as F
from torch import nn


def _gabor_bank(n_orient: int, kernel: int) -> torch.Tensor:
    coords = torch.arange(kernel, dtype=torch.float32) - (kernel - 1) / 2.0
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    filters = []
    for o in range(n_orient):
        theta = math.pi * o / n_orient
        rot_x = grid_x * math.cos(theta) + grid_y * math.sin(theta)
        rot_y = -grid_x * math.sin(theta) + grid_y * math.cos(theta)
        for freq in (0.25, 0.5):
            envelope = torch.exp(-(rot_x**2 + rot_y**2) / (2.0 * (kernel / 4.0) ** 2))
            carrier = torch.cos(2.0 * math.pi * freq * rot_x)
            kern = envelope * carrier
            kern = kern - kern.mean()
            filters.append(kern)
    stack = torch.stack(filters, dim=0).unsqueeze(1)
    return stack


class RetfoundDinov2Adapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        in_channels: int = 1,
        flat_dim: int = 1024,
        n_orient: int = 8,
        kernel: int = 7,
        proj_dim: int = 256,
        use_real_backbone: bool = False,
        seed: int = 20260617,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.use_real_backbone = use_real_backbone
        gen = torch.Generator().manual_seed(seed)

        bank = _gabor_bank(n_orient, kernel)
        self.register_buffer("gabor", bank)
        self.gabor: torch.Tensor
        self.gabor_pad = kernel // 2
        n_filters = bank.shape[0]

        proj = torch.randn(flat_dim, proj_dim, generator=gen)
        proj = proj / math.sqrt(flat_dim)
        self.register_buffer("plane_proj", proj)
        self.plane_proj: torch.Tensor

        feat_dim = 2 * n_filters + proj_dim
        mix = torch.randn(feat_dim, embed_dim, generator=gen) / math.sqrt(feat_dim)
        self.register_buffer("mixer", mix)
        self.mixer: torch.Tensor
        self.register_buffer("bias", torch.zeros(embed_dim))
        self.bias: torch.Tensor

    def _assert_real(self) -> None:
        if self.use_real_backbone:
            if importlib.util.find_spec("timm") is None:
                raise RuntimeError(
                    "use_real_backbone=True requires timm and RETFound-DINOv2 weights"
                )
            raise RuntimeError(
                "RETFound-DINOv2 weights are not bundled; provide them or set "
                "use_real_backbone=False"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self._assert_real()
        with torch.no_grad():
            n = images.shape[0]
            gray = images.mean(dim=1, keepdim=True)
            responses = F.conv2d(gray, self.gabor, padding=self.gabor_pad)
            activated = F.relu(responses)
            avg = activated.mean(dim=(2, 3))
            peak = activated.amax(dim=(2, 3))
            flat = images.reshape(n, -1)
            projected = flat @ self.plane_proj
            feats = torch.cat([avg, peak, projected], dim=1)
            out = feats @ self.mixer + self.bias
            return torch.tanh(out)


class TemporalPositional(nn.Module):
    def __init__(self, embed_dim: int, n_freqs: int = 8, width: int = 256) -> None:
        super().__init__()
        self.omega = nn.Parameter(torch.logspace(-2.0, 0.0, n_freqs))
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, width),
            nn.GELU(),
            nn.Linear(width, embed_dim),
        )

    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        scaled = deltas.unsqueeze(-1) * self.omega
        feats = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.mlp(feats)


class CrossVisitPool(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim) / math.sqrt(embed_dim))
        self.project = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, tokens: torch.Tensor, visit_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.project(tokens) @ self.query
        scores = scores / math.sqrt(tokens.shape[-1])
        scores = scores.masked_fill(~visit_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * tokens).sum(dim=1)
        return pooled, weights


class LongitudinalStream(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        in_channels: int = 1,
        flat_dim: int = 1024,
        n_freqs: int = 8,
        tpe_width: int = 256,
        use_real_backbone: bool = False,
    ) -> None:
        super().__init__()
        proj_dim = min(256, flat_dim)
        self.backbone = RetfoundDinov2Adapter(
            embed_dim=embed_dim,
            in_channels=in_channels,
            flat_dim=flat_dim,
            proj_dim=proj_dim,
            use_real_backbone=use_real_backbone,
        )
        self.tpe = TemporalPositional(embed_dim, n_freqs=n_freqs, width=tpe_width)
        self.pool = CrossVisitPool(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def encode(
        self, images: torch.Tensor, deltas: torch.Tensor, visit_mask: torch.Tensor
    ) -> torch.Tensor:
        b, v = images.shape[0], images.shape[1]
        merged = images.reshape(b * v, *images.shape[2:])
        embedded = self.backbone(merged).reshape(b, v, -1)
        positioned = embedded + self.tpe(deltas)
        return positioned.masked_fill(~visit_mask.unsqueeze(-1), 0.0)

    def forward(
        self, images: torch.Tensor, deltas: torch.Tensor, visit_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positioned = self.encode(images, deltas, visit_mask)
        pooled, weights = self.pool(positioned, visit_mask)
        return self.norm(pooled), weights

    def pool_prefix(self, tokens: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.pool(tokens, prefix_mask)
        return self.norm(pooled)

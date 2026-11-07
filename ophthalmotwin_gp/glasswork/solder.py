from __future__ import annotations

import torch
from torch import nn


class TwinState(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def advance(self, previous: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        return self.norm(previous + fused)


class CrossModalSolder(nn.Module):
    def __init__(
        self,
        img_dim: int,
        gen_dim: int,
        twin_dim: int,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.img_proj = nn.Linear(img_dim, twin_dim)
        self.gen_proj = nn.Linear(gen_dim, twin_dim)
        self.img_to_gen = nn.MultiheadAttention(twin_dim, heads, dropout=dropout, batch_first=True)
        self.gen_to_img = nn.MultiheadAttention(twin_dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(twin_dim, twin_dim),
            nn.GELU(),
            nn.Linear(twin_dim, twin_dim),
        )
        self.concat_proj = nn.Linear(2 * twin_dim, twin_dim)
        self.norm = nn.LayerNorm(twin_dim)
        self.twin = TwinState(twin_dim)

    def fuse(self, z_img: torch.Tensor, z_gen: torch.Tensor) -> torch.Tensor:
        img = self.img_proj(z_img).unsqueeze(1)
        gen = self.gen_proj(z_gen).unsqueeze(1)
        forward, _ = self.img_to_gen(img, gen, gen)
        backward, _ = self.gen_to_img(gen, img, img)
        merged = (forward + backward).squeeze(1)
        return self.norm(merged + self.ffn(merged))

    def fuse_concat(self, z_img: torch.Tensor, z_gen: torch.Tensor) -> torch.Tensor:
        img = self.img_proj(z_img)
        gen = self.gen_proj(z_gen)
        merged = self.concat_proj(torch.cat([img, gen], dim=-1))
        return self.norm(merged + self.ffn(merged))

    def forward(
        self,
        z_img: torch.Tensor,
        z_gen: torch.Tensor,
        previous: torch.Tensor,
        mode: str = "cross",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.fuse_concat(z_img, z_gen) if mode == "concat" else self.fuse(z_img, z_gen)
        state = self.twin.advance(previous, fused)
        return state, fused

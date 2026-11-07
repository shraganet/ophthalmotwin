from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ophthalmotwin_gp.glasswork.panel import TwinOutput
from ophthalmotwin_gp.selection.panes import VisitBatch


def cross_entropy_term(gene_logits: torch.Tensor, gene: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(gene_logits, gene)


def pinball_term(
    quantiles: torch.Tensor, target: torch.Tensor, levels: tuple[float, ...]
) -> torch.Tensor:
    total = quantiles.new_zeros(())
    for column, level in enumerate(levels):
        error = target - quantiles[:, column]
        total = total + torch.maximum(level * error, (level - 1.0) * error).mean()
    return total / len(levels)


def focal_bce_term(logit: torch.Tensor, label: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(logit)
    pt = torch.where(label > 0.5, prob, 1.0 - prob).clamp_min(1e-6)
    weight = (1.0 - pt) ** gamma
    return (-weight * torch.log(pt)).mean()


def l2_reg_term(z_img: torch.Tensor, z_gen: torch.Tensor) -> torch.Tensor:
    return z_img.pow(2).mean() + z_gen.pow(2).mean()


@dataclass
class JointObjective:
    lambda_prog: float = 0.5
    lambda_early: float = 0.3
    lambda_reg: float = 1e-4
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)

    def __call__(
        self, output: TwinOutput, batch: VisitBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gene = cross_entropy_term(output.gene_logits, batch.gene)
        prog = pinball_term(output.prog_quantiles, batch.prog_target, self.quantile_levels)
        early = focal_bce_term(output.early_logit, batch.early_label)
        reg = l2_reg_term(output.z_img, output.z_gen)
        total = gene + self.lambda_prog * prog + self.lambda_early * early + self.lambda_reg * reg
        parts = {"gene": gene, "prog": prog, "early": early, "reg": reg, "total": total}
        return total, parts

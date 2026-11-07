from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def raps_scores(probs: torch.Tensor, labels: torch.Tensor, lam: float, kreg: int) -> torch.Tensor:
    sorted_p, order = torch.sort(probs, dim=1, descending=True)
    cumulative = sorted_p.cumsum(dim=1)
    positions = (order == labels.unsqueeze(1)).float().argmax(dim=1)
    cum_at_true = cumulative.gather(1, positions.unsqueeze(1)).squeeze(1)
    rank = positions.float() + 1.0
    regular = lam * torch.clamp(rank - kreg, min=0.0)
    return cum_at_true + regular


def _quantile_level(n: int, alpha: float) -> float:
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    return min(level, 1.0)


def prediction_set(probs: torch.Tensor, qhat: float, lam: float, kreg: int) -> list[list[int]]:
    sorted_p, order = torch.sort(probs, dim=1, descending=True)
    cumulative = sorted_p.cumsum(dim=1)
    ranks = torch.arange(1, probs.shape[1] + 1, device=probs.device).float()
    regular = lam * torch.clamp(ranks - kreg, min=0.0)
    scores = cumulative + regular.unsqueeze(0)
    include = scores <= qhat
    include[:, 0] = True
    sets: list[list[int]] = []
    for row in range(probs.shape[0]):
        chosen = order[row][include[row]]
        sets.append([int(c) for c in chosen.tolist()])
    return sets


@dataclass
class RapsCalibrator:
    alpha: float = 0.10
    lam: float = 0.05
    kreg: int = 5
    qhat: float = 1.0

    def calibrate(self, cal_probs: torch.Tensor, cal_labels: torch.Tensor) -> float:
        scores = raps_scores(cal_probs, cal_labels, self.lam, self.kreg)
        n = scores.shape[0]
        level = _quantile_level(n, self.alpha)
        self.qhat = float(torch.quantile(scores, level).item())
        return self.qhat

    def predict(self, probs: torch.Tensor) -> list[list[int]]:
        return prediction_set(probs, self.qhat, self.lam, self.kreg)

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Patient:
    patient_id: str
    gene: int
    inheritance: int
    variant_indices: list[int]
    images: torch.Tensor
    deltas: torch.Tensor
    prog_target: float
    early_label: int
    sex: int
    age_band: int
    ancestry: int
    fold: str



def fold_indices(patients: list[Patient], fold: str) -> list[int]:
    return [i for i, p in enumerate(patients) if p.fold == fold]

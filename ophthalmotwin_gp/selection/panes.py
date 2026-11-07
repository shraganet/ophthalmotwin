from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from ophthalmotwin_gp.selection.stock import Patient


@dataclass
class VisitBatch:
    images: torch.Tensor
    deltas: torch.Tensor
    visit_mask: torch.Tensor
    variant_lists: list[list[int]]
    gene: torch.Tensor
    inheritance: torch.Tensor
    prog_target: torch.Tensor
    early_label: torch.Tensor
    sex: torch.Tensor
    age_band: torch.Tensor
    ancestry: torch.Tensor
    patient_ids: list[str]

    def to(self, device: torch.device) -> VisitBatch:
        return VisitBatch(
            images=self.images.to(device),
            deltas=self.deltas.to(device),
            visit_mask=self.visit_mask.to(device),
            variant_lists=self.variant_lists,
            gene=self.gene.to(device),
            inheritance=self.inheritance.to(device),
            prog_target=self.prog_target.to(device),
            early_label=self.early_label.to(device),
            sex=self.sex.to(device),
            age_band=self.age_band.to(device),
            ancestry=self.ancestry.to(device),
            patient_ids=self.patient_ids,
        )

    def __len__(self) -> int:
        return self.gene.shape[0]


class VisitStreamDataset(Dataset[Patient]):
    def __init__(self, patients: list[Patient], indices: list[int] | None = None) -> None:
        self._patients = patients
        self._indices = list(range(len(patients))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, item: int) -> Patient:
        return self._patients[self._indices[item]]


def collate_streams(items: list[Patient]) -> VisitBatch:
    batch = len(items)
    max_visits = max(p.images.shape[0] for p in items)
    sample = items[0].images
    channels, height, width = sample.shape[1], sample.shape[2], sample.shape[3]

    images = torch.zeros(batch, max_visits, channels, height, width)
    deltas = torch.zeros(batch, max_visits)
    visit_mask = torch.zeros(batch, max_visits, dtype=torch.bool)
    for row, patient in enumerate(items):
        v = patient.images.shape[0]
        images[row, :v] = patient.images
        deltas[row, :v] = patient.deltas
        visit_mask[row, :v] = True

    return VisitBatch(
        images=images,
        deltas=deltas,
        visit_mask=visit_mask,
        variant_lists=[list(p.variant_indices) for p in items],
        gene=torch.tensor([p.gene for p in items], dtype=torch.long),
        inheritance=torch.tensor([p.inheritance for p in items], dtype=torch.long),
        prog_target=torch.tensor([p.prog_target for p in items], dtype=torch.float32),
        early_label=torch.tensor([float(p.early_label) for p in items], dtype=torch.float32),
        sex=torch.tensor([p.sex for p in items], dtype=torch.long),
        age_band=torch.tensor([p.age_band for p in items], dtype=torch.long),
        ancestry=torch.tensor([p.ancestry for p in items], dtype=torch.long),
        patient_ids=[p.patient_id for p in items],
    )

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ophthalmotwin_gp.bench import atomic_save, count_parameters, get_logger, pick_device
from ophthalmotwin_gp.cartoon import Run
from ophthalmotwin_gp.flux import JointObjective
from ophthalmotwin_gp.glasswork.panel import OphthalmoTwinGP
from ophthalmotwin_gp.selection.leadlines import VariantGraph
from ophthalmotwin_gp.selection.panes import VisitBatch, VisitStreamDataset, collate_streams
from ophthalmotwin_gp.selection.stock import Patient, fold_indices


@dataclass
class Assembly:
    model: OphthalmoTwinGP
    patients: list[Patient]
    graph: VariantGraph


@dataclass
class Predictions:
    gene_probs: torch.Tensor
    gene_labels: torch.Tensor
    early_prob: torch.Tensor
    early_labels: torch.Tensor
    prog_quantiles: torch.Tensor
    prog_target: torch.Tensor
    sex: torch.Tensor
    age_band: torch.Tensor
    ancestry: torch.Tensor


def assemble(run: Run, use_real_backbone: bool = False) -> Assembly:
    raise NotImplementedError(
        "Data integration is required: provide patient records and a variant knowledge graph."
    )


def _loader(
    patients: list[Patient], indices: list[int], batch_size: int, shuffle: bool
) -> DataLoader[Patient]:
    dataset = VisitStreamDataset(patients, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_streams,
        drop_last=False,
    )


def _schedule(
    optimizer: torch.optim.Optimizer, warmup: int, total: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if total <= warmup:
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Trainer:
    def __init__(self, model: OphthalmoTwinGP, run: Run) -> None:
        self.run = run
        self.device = pick_device(run.firing.device)
        self.model = model.to(self.device)
        self.model.longitudinal.backbone.requires_grad_(False)
        self.objective = JointObjective(
            lambda_prog=run.firing.lambda_prog,
            lambda_early=run.firing.lambda_early,
            lambda_reg=run.firing.lambda_reg,
            quantile_levels=run.firing.quantile_levels,
        )
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=run.firing.lr,
            weight_decay=run.firing.weight_decay,
        )
        self.logger = get_logger()

    def fit(self, patients: list[Patient], graph: VariantGraph) -> list[float]:
        graph = graph.to(self.device)
        indices = fold_indices(patients, "train")
        loader = _loader(patients, indices, self.run.firing.batch_size, shuffle=True)
        steps_per_epoch = max(1, len(loader))
        total = self.run.firing.epochs * steps_per_epoch
        if self.run.firing.max_steps > 0:
            total = min(total, self.run.firing.max_steps)
        scheduler = _schedule(self.optimizer, self.run.firing.warmup_steps, total)
        use_amp = self.run.firing.amp and self.device.type == "cuda"
        scaler = torch.amp.grad_scaler.GradScaler(enabled=use_amp)

        history: list[float] = []
        step = 0
        self.model.train()
        for epoch in range(self.run.firing.epochs):
            for batch in loader:
                moved = batch.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast_mode.autocast(self.device.type, enabled=use_amp):
                    output = self.model(moved, graph)
                    loss, parts = self.objective(output, moved)
                scaler.scale(loss).backward()
                if self.run.firing.grad_clip > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.run.firing.grad_clip
                    )
                scaler.step(self.optimizer)
                scaler.update()
                scheduler.step()
                history.append(float(loss.item()))
                if step % self.run.firing.log_every == 0:
                    self.logger.info(
                        "epoch %d step %d loss %.4f gene %.4f",
                        epoch,
                        step,
                        float(parts["total"].item()),
                        float(parts["gene"].item()),
                    )
                step += 1
                if self.run.firing.max_steps > 0 and step >= self.run.firing.max_steps:
                    self.save_checkpoint(step)
                    return history
            self.save_checkpoint(step)
        return history

    def save_checkpoint(self, step: int) -> str:
        path = os.path.join(self.run.firing.out_dir, "twin.pt")
        atomic_save(
            {
                "model": self.model.state_dict(),
                "step": step,
                "seed": self.run.firing.seed,
                "params": count_parameters(self.model),
            },
            path,
        )
        return path


@torch.no_grad()
def gather_predictions(
    model: OphthalmoTwinGP,
    graph: VariantGraph,
    patients: list[Patient],
    fold: str,
    batch_size: int,
    device: torch.device,
) -> Predictions:
    model.eval()
    graph = graph.to(device)
    indices = fold_indices(patients, fold)
    loader = _loader(patients, indices, batch_size, shuffle=False)
    gene_probs: list[torch.Tensor] = []
    gene_labels: list[torch.Tensor] = []
    early_prob: list[torch.Tensor] = []
    early_labels: list[torch.Tensor] = []
    prog_q: list[torch.Tensor] = []
    prog_t: list[torch.Tensor] = []
    sex: list[torch.Tensor] = []
    age: list[torch.Tensor] = []
    ancestry: list[torch.Tensor] = []
    for batch in loader:
        moved: VisitBatch = batch.to(device)
        output = model(moved, graph)
        gene_probs.append(torch.softmax(output.gene_logits, dim=1).cpu())
        gene_labels.append(moved.gene.cpu())
        early_prob.append(torch.sigmoid(output.early_logit).cpu())
        early_labels.append(moved.early_label.cpu())
        prog_q.append(output.prog_quantiles.cpu())
        prog_t.append(moved.prog_target.cpu())
        sex.append(moved.sex.cpu())
        age.append(moved.age_band.cpu())
        ancestry.append(moved.ancestry.cpu())
    return Predictions(
        gene_probs=torch.cat(gene_probs),
        gene_labels=torch.cat(gene_labels),
        early_prob=torch.cat(early_prob),
        early_labels=torch.cat(early_labels),
        prog_quantiles=torch.cat(prog_q),
        prog_target=torch.cat(prog_t),
        sex=torch.cat(sex),
        age_band=torch.cat(age),
        ancestry=torch.cat(ancestry),
    )

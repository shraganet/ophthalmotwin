from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ophthalmotwin_gp.cartoon import Panel
from ophthalmotwin_gp.glasswork.clearstream import LongitudinalStream
from ophthalmotwin_gp.glasswork.leadnet import GenotypeStream
from ophthalmotwin_gp.glasswork.solder import CrossModalSolder
from ophthalmotwin_gp.selection.leadlines import VariantGraph
from ophthalmotwin_gp.selection.panes import VisitBatch


@dataclass
class TwinOutput:
    gene_logits: torch.Tensor
    prog_quantiles: torch.Tensor
    early_logit: torch.Tensor
    twin_state: torch.Tensor
    z_img: torch.Tensor
    z_gen: torch.Tensor
    visit_weights: torch.Tensor
    edge_mass: dict[str, torch.Tensor]


class OphthalmoTwinGP(nn.Module):
    def __init__(
        self,
        config: Panel,
        in_channels: int = 1,
        flat_dim: int = 1024,
        use_real_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.longitudinal = LongitudinalStream(
            embed_dim=config.img_embed_dim,
            in_channels=in_channels,
            flat_dim=flat_dim,
            n_freqs=config.tpe_freqs,
            tpe_width=config.tpe_width,
            use_real_backbone=use_real_backbone,
        )
        self.genotype = GenotypeStream(
            node_dim=config.node_dim,
            hidden=config.gen_embed_dim,
            layers=config.hgt_layers,
            heads=config.hgt_heads,
        )
        self.solder = CrossModalSolder(
            img_dim=config.img_embed_dim,
            gen_dim=config.gen_embed_dim,
            twin_dim=config.twin_dim,
            heads=config.fusion_heads,
            dropout=config.dropout,
        )
        self.initial_state = nn.Parameter(torch.zeros(config.twin_dim))
        self.gene_head = nn.Linear(config.twin_dim, config.n_genes)
        self.prog_head = nn.Linear(config.twin_dim, config.n_quantiles)
        self.early_head = nn.Linear(config.twin_dim, 1)

    def visit_pass(
        self, batch: VisitBatch, graph: VariantGraph
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        z_gen, masses = self.genotype(graph, batch.variant_lists)
        if not self.config.use_genotype:
            z_gen = torch.zeros_like(z_gen)
        tokens = self.longitudinal.encode(batch.images, batch.deltas, batch.visit_mask)
        batch_size, visits = batch.visit_mask.shape
        state = self.initial_state.unsqueeze(0).expand(batch_size, -1).contiguous()
        arange = torch.arange(visits, device=batch.visit_mask.device)
        for step in range(visits):
            active = batch.visit_mask[:, step]
            if not bool(active.any()):
                continue
            prefix = batch.visit_mask & (arange.unsqueeze(0) <= step)
            z_img = self.longitudinal.pool_prefix(tokens, prefix)
            if not self.config.use_longitudinal:
                z_img = torch.zeros_like(z_img)
            advanced, _ = self.solder(z_img, z_gen, state, mode=self.config.fusion_mode)
            gate = active.unsqueeze(-1).to(state.dtype)
            state = gate * advanced + (1.0 - gate) * state
        return state, z_gen, masses

    def forward(self, batch: VisitBatch, graph: VariantGraph) -> TwinOutput:
        state, z_gen, masses = self.visit_pass(batch, graph)
        z_img, weights = self.longitudinal(batch.images, batch.deltas, batch.visit_mask)
        return TwinOutput(
            gene_logits=self.gene_head(state),
            prog_quantiles=self.prog_head(state),
            early_logit=self.early_head(state).squeeze(-1),
            twin_state=state,
            z_img=z_img,
            z_gen=z_gen,
            visit_weights=weights,
            edge_mass=masses,
        )

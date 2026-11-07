from __future__ import annotations

from dataclasses import dataclass

import torch

VARIANT, GENE, DISEASE, DOMAIN, FREQBIN = 0, 1, 2, 3, 4
NODE_TYPES = 5

RELATIONS = (
    "affects",
    "causes",
    "located_in",
    "population_frequency",
    "vcep_classified",
)

VARIANT_CATEGORIES = ("missense", "splice", "lof", "cnv")


@dataclass
class VariantGraph:
    node_type: torch.Tensor
    node_feat: torch.Tensor
    gene_node_id: torch.Tensor
    variant_node_id: torch.Tensor
    variant_gene: torch.Tensor
    variant_is_vcep: torch.Tensor
    variant_category: torch.Tensor
    edges: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    num_nodes: int
    n_genes: int

    def to(self, device: torch.device) -> VariantGraph:
        moved = {
            rel: (s.to(device), d.to(device), w.to(device)) for rel, (s, d, w) in self.edges.items()
        }
        return VariantGraph(
            node_type=self.node_type.to(device),
            node_feat=self.node_feat.to(device),
            gene_node_id=self.gene_node_id.to(device),
            variant_node_id=self.variant_node_id.to(device),
            variant_gene=self.variant_gene.to(device),
            variant_is_vcep=self.variant_is_vcep.to(device),
            variant_category=self.variant_category.to(device),
            edges=moved,
            num_nodes=self.num_nodes,
            n_genes=self.n_genes,
        )



def identify_gene_from_vcep(graph: VariantGraph, variant_indices: list[int]) -> int | None:
    tally: dict[int, int] = {}
    for idx in variant_indices:
        if bool(graph.variant_is_vcep[idx]):
            gene = int(graph.variant_gene[idx])
            tally[gene] = tally.get(gene, 0) + 1
    if not tally:
        return None
    best = max(tally.values())
    winners = [gene for gene, count in tally.items() if count == best]
    if len(winners) != 1:
        return None
    return winners[0]

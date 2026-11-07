from __future__ import annotations

import math

import torch
from torch import nn

from ophthalmotwin_gp.selection.leadlines import NODE_TYPES, RELATIONS, VariantGraph


def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    heads = scores.shape[1]
    expanded = index.unsqueeze(1).expand(-1, heads)
    max_buf = scores.new_full((num_nodes, heads), float("-inf"))
    max_buf = max_buf.scatter_reduce(0, expanded, scores, reduce="amax", include_self=True)
    gathered_max = max_buf.index_select(0, index)
    gathered_max = torch.nan_to_num(gathered_max, neginf=0.0)
    exp = torch.exp(scores - gathered_max)
    sum_buf = scores.new_zeros(num_nodes, heads)
    sum_buf.index_add_(0, index, exp)
    gathered_sum = sum_buf.index_select(0, index).clamp_min(1e-12)
    return exp / gathered_sum


class HGTLayer(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.query = nn.ModuleDict({rel: nn.Linear(hidden, hidden) for rel in RELATIONS})
        self.key = nn.ModuleDict({rel: nn.Linear(hidden, hidden) for rel in RELATIONS})
        self.message = nn.ModuleDict({rel: nn.Linear(hidden, hidden) for rel in RELATIONS})
        self.skip = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        nodes: torch.Tensor,
        graph: VariantGraph,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        num_nodes = nodes.shape[0]
        aggregate = nodes.new_zeros(num_nodes, self.heads, self.head_dim)
        masses: dict[str, torch.Tensor] = {}
        for rel in RELATIONS:
            src, dst, weight = graph.edges[rel]
            if src.numel() == 0:
                masses[rel] = nodes.new_zeros(())
                continue
            q = self.query[rel](nodes).view(num_nodes, self.heads, self.head_dim)
            k = self.key[rel](nodes).view(num_nodes, self.heads, self.head_dim)
            m = self.message[rel](nodes).view(num_nodes, self.heads, self.head_dim)
            q_dst = q.index_select(0, dst)
            k_src = k.index_select(0, src)
            m_src = m.index_select(0, src)
            score = (q_dst * k_src).sum(-1) / math.sqrt(self.head_dim)
            score = score + torch.log(weight.clamp_min(1e-6)).unsqueeze(1)
            alpha = _segment_softmax(score, dst, num_nodes)
            weighted = alpha.unsqueeze(-1) * m_src
            aggregate.index_add_(0, dst, weighted)
            masses[rel] = alpha.mean()
        gathered = aggregate.reshape(num_nodes, self.hidden)
        out = self.norm(torch.relu(gathered) + self.skip(nodes))
        return out, masses


class HeteroGraphTransformer(nn.Module):
    def __init__(self, node_dim: int, hidden: int, layers: int, heads: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.ModuleList([nn.Linear(node_dim, hidden) for _ in range(NODE_TYPES)])
        self.layers = nn.ModuleList([HGTLayer(hidden, heads) for _ in range(layers)])

    def forward(self, graph: VariantGraph) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        nodes = graph.node_feat.new_zeros(graph.num_nodes, self.hidden)
        for type_id, proj in enumerate(self.input_proj):
            mask = graph.node_type == type_id
            if bool(mask.any()):
                nodes[mask] = proj(graph.node_feat[mask])
        masses: dict[str, torch.Tensor] = {}
        for layer in self.layers:
            nodes, masses = layer(nodes, graph)
        return nodes, masses


class GatedReadout(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, 1)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        variant_node_id: torch.Tensor,
        variant_lists: list[list[int]],
    ) -> torch.Tensor:
        outputs = []
        for indices in variant_lists:
            ids = variant_node_id.index_select(0, variant_node_id.new_tensor(indices))
            picked = node_embeddings.index_select(0, ids)
            gates = torch.sigmoid(self.gate(picked))
            outputs.append((gates * picked).sum(dim=0))
        return torch.stack(outputs, dim=0)

    def edge_mass(self, masses: dict[str, torch.Tensor]) -> dict[str, float]:
        total = sum(float(v) for v in masses.values())
        if total <= 0.0:
            return {rel: 0.0 for rel in masses}
        return {rel: float(v) / total for rel, v in masses.items()}


class GenotypeStream(nn.Module):
    def __init__(self, node_dim: int, hidden: int, layers: int, heads: int) -> None:
        super().__init__()
        self.hgt = HeteroGraphTransformer(node_dim, hidden, layers, heads)
        self.readout = GatedReadout(hidden)

    def forward(
        self, graph: VariantGraph, variant_lists: list[list[int]]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        node_embeddings, masses = self.hgt(graph)
        z_gen = self.readout(node_embeddings, graph.variant_node_id, variant_lists)
        return z_gen, masses

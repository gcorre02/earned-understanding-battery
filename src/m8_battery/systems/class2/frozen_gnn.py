"""System 2B — Frozen GNN (GAT).

A Class 2 system: trained on domain A, then frozen. Has learned structure
but does not learn during operation.

Architecture: 2-layer GAT (4 heads, 16 hidden per head) trained on
SBM domain for node classification (community detection). Then frozen.

Expected battery result: some instruments may pass (has structure),
but developmental trajectory should FAIL (no learning during operation).
"""

from __future__ import annotations

import pickle
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from m8_battery.core.test_system import TestSystem


class FrozenGAT(TestSystem):
    """Frozen Graph Attention Network for graph navigation.

    Trained on domain A for node classification, then weights frozen.
    Uses attention entropy as structure metric. Ablation via head zeroing.
    """

    def __init__(
        self,
        n_features: int = 8,
        n_classes: int = 4,
        hidden_dim: int = 16,
        n_heads: int = 4,
        seed: int = 42,
    ) -> None:
        self._seed = seed
        self._n_features = n_features
        self._n_classes = n_classes
        self._hidden_dim = hidden_dim
        self._n_heads = n_heads

        torch.manual_seed(seed)

        # 2-layer GAT
        from torch_geometric.nn import GATConv

        self._conv1 = GATConv(n_features, hidden_dim, heads=n_heads, concat=True)
        self._conv2 = GATConv(hidden_dim * n_heads, n_classes, heads=1, concat=False)

        self._graph: nx.DiGraph | None = None
        self._pyg_data: Any = None
        self._current_node: int | None = None
        self._step_count = 0
        self._visit_counts: dict[int, int] = {}
        self._is_frozen = False

    def train_on_domain(self, graph: nx.DiGraph, epochs: int = 50, lr: float = 0.01) -> float:
        """Train GAT on the domain graph for node classification, then freeze.

        Labels = community assignments. Returns final accuracy.
        """
        self._graph = graph
        self._pyg_data = self._graph_to_pyg(graph)

        optimizer = torch.optim.Adam(
            list(self._conv1.parameters()) + list(self._conv2.parameters()),
            lr=lr,
        )

        self._conv1.train()
        self._conv2.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self._forward(self._pyg_data.x, self._pyg_data.edge_index)
            loss = F.cross_entropy(out, self._pyg_data.y)
            loss.backward()
            optimizer.step()

        # Freeze
        self._conv1.eval()
        self._conv2.eval()
        for param in self._conv1.parameters():
            param.requires_grad = False
        for param in self._conv2.parameters():
            param.requires_grad = False
        self._is_frozen = True

        # Compute accuracy
        with torch.no_grad():
            pred = self._forward(self._pyg_data.x, self._pyg_data.edge_index).argmax(dim=1)
            accuracy = (pred == self._pyg_data.y).float().mean().item()

        return accuracy

    def _forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GAT."""
        h = self._conv1(x, edge_index)
        h = F.elu(h)
        out = self._conv2(h, edge_index)
        return out

    def _graph_to_pyg(self, G: nx.DiGraph) -> Any:
        """Convert networkx graph to PyG Data object."""
        from torch_geometric.data import Data

        nodes = sorted(G.nodes())
        node_map = {n: i for i, n in enumerate(nodes)}

        # Node features
        features = []
        labels = []
        for n in nodes:
            feats = G.nodes[n].get("features", {})
            feat_vec = [feats.get(f"feat_{i}", 0.0) for i in range(self._n_features)]
            features.append(feat_vec)
            labels.append(feats.get("community", 0))

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)

        # Edges
        edge_list = []
        for u, v in G.edges():
            if u in node_map and v in node_map:
                edge_list.append([node_map[u], node_map[v]])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)

    def reset(self) -> None:
        self._current_node = None
        self._step_count = 0
        self._visit_counts = {}

    def step(self, input_data: Any) -> Any:
        if self._graph is None or self._pyg_data is None:
            return {"error": "no graph — call train_on_domain first"}

        nodes = sorted(self._graph.nodes())
        if not nodes:
            return {"error": "empty graph"}

        if self._current_node is None:
            self._current_node = nodes[0]

        # Forward pass (frozen — no gradient)
        with torch.no_grad():
            out = self._forward(self._pyg_data.x, self._pyg_data.edge_index)
            node_idx = {n: i for i, n in enumerate(nodes)}

            # Use softmax probabilities to select next node
            if self._current_node in node_idx:
                probs = F.softmax(out[node_idx[self._current_node]], dim=0)

        # Navigate based on classifier confidence
        successors = list(self._graph.successors(self._current_node))
        if successors and input_data is not None and input_data in self._graph:
            try:
                path = nx.shortest_path(self._graph, self._current_node, input_data)
                if len(path) > 1:
                    self._current_node = path[1]
            except nx.NetworkXNoPath:
                if successors:
                    self._current_node = successors[0]
        elif successors:
            self._current_node = successors[0]

        self._visit_counts[self._current_node] = (
            self._visit_counts.get(self._current_node, 0) + 1
        )
        self._step_count += 1

        return {
            "current_node": self._current_node,
            "step": self._step_count,
        }

    def get_state(self) -> bytes:
        return pickle.dumps({
            "current_node": self._current_node,
            "step_count": self._step_count,
            "visit_counts": self._visit_counts,
            "conv1_state": self._conv1.state_dict(),
            "conv2_state": self._conv2.state_dict(),
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._current_node = state["current_node"]
        self._step_count = state["step_count"]
        self._visit_counts = state["visit_counts"]
        self._conv1.load_state_dict(state["conv1_state"])
        self._conv2.load_state_dict(state["conv2_state"])

    def get_structure_metric(self) -> float:
        """Mean attention entropy across nodes — CONSTANT when frozen."""
        if self._pyg_data is None:
            return 0.0

        with torch.no_grad():
            # Get attention weights from first GAT layer
            h, (edge_index_out, alpha) = self._conv1(
                self._pyg_data.x, self._pyg_data.edge_index,
                return_attention_weights=True,
            )
            # Compute per-node attention entropy
            # alpha shape: (num_edges, num_heads)
            if alpha is not None and alpha.numel() > 0:
                # Clamp to avoid log(0)
                alpha_clamped = alpha.clamp(min=1e-10)
                entropy = -(alpha_clamped * alpha_clamped.log()).sum(dim=-1).mean()
                return float(entropy.item())

        return 0.0

    def get_structure_distribution(self) -> dict[str, float]:
        if self._graph is None:
            return {}
        communities: dict[int, list[int]] = {}
        for node in self._graph.nodes():
            block = self._graph.nodes[node].get("features", {}).get("community", 0)
            communities.setdefault(block, []).append(node)
        return {
            f"community_{c}": float(nx.density(self._graph.subgraph(ns)))
            for c, ns in communities.items()
            if self._graph.subgraph(ns).number_of_edges() > 0
        }

    def get_engagement_distribution(self) -> dict[str, float]:
        if self._graph is None:
            return {}
        communities: dict[int, list[int]] = {}
        for node in self._graph.nodes():
            block = self._graph.nodes[node].get("features", {}).get("community", 0)
            communities.setdefault(block, []).append(node)
        total = sum(self._visit_counts.values()) or 1
        return {
            f"community_{c}": sum(self._visit_counts.get(n, 0) for n in ns) / total
            for c, ns in communities.items()
        }

    def ablate(self, region_id: str) -> TestSystem:
        """Ablation: zero out a GAT attention head."""
        new = FrozenGAT(
            n_features=self._n_features, n_classes=self._n_classes,
            hidden_dim=self._hidden_dim, n_heads=self._n_heads, seed=self._seed,
        )
        # Copy weights
        new._conv1.load_state_dict(self._conv1.state_dict())
        new._conv2.load_state_dict(self._conv2.state_dict())
        new._graph = self._graph.copy()
        new._pyg_data = self._pyg_data
        new._is_frozen = True

        # Zero the attention head corresponding to the region
        head_idx = int(region_id.split("_")[-1]) % self._n_heads
        with torch.no_grad():
            # Zero the linear projection for this head
            h_dim = self._hidden_dim
            start = head_idx * h_dim
            end = start + h_dim
            new._conv1.lin.weight[start:end, :] = 0

        return new

    def perturb(self, region_id: str, method: str = "shuffle_weights") -> TestSystem:
        new = FrozenGAT(
            n_features=self._n_features, n_classes=self._n_classes,
            hidden_dim=self._hidden_dim, n_heads=self._n_heads, seed=self._seed,
        )
        new._conv1.load_state_dict(self._conv1.state_dict())
        new._conv2.load_state_dict(self._conv2.state_dict())
        new._graph = self._graph.copy()
        new._pyg_data = self._pyg_data
        new._is_frozen = True

        head_idx = int(region_id.split("_")[-1]) % self._n_heads
        with torch.no_grad():
            h_dim = self._hidden_dim
            start = head_idx * h_dim
            end = start + h_dim
            rng = torch.Generator().manual_seed(self._seed + 999)
            new._conv1.lin.weight[start:end, :] = torch.randn(
                h_dim, self._n_features, generator=rng
            ) * 0.1

        return new

    def get_regions(self) -> list[str]:
        """Regions = GAT attention heads."""
        return [f"head_{i}" for i in range(self._n_heads)]

    def clone(self) -> TestSystem:
        new = FrozenGAT(
            n_features=self._n_features, n_classes=self._n_classes,
            hidden_dim=self._hidden_dim, n_heads=self._n_heads, seed=self._seed,
        )
        new._conv1.load_state_dict(self._conv1.state_dict())
        new._conv2.load_state_dict(self._conv2.state_dict())
        new._graph = self._graph.copy()
        new._pyg_data = self._pyg_data
        new._is_frozen = self._is_frozen
        return new

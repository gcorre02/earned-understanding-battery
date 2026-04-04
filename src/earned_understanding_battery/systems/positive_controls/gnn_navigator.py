"""Positive Control 3: Frozen GNN Navigator.

Trains a GCN on domain A to predict community membership from node features.
When frozen on domain B, uses learned embeddings to bias action selection
toward nodes with similar embedding profiles to high-value A nodes.

The GNN learns structural abstractions (community-level patterns) that
transfer across SBM instances with same parameters.

Expected: passes generativity when frozen on zero-overlap domain B.
"""

from __future__ import annotations

import pickle
import sys
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from earned_understanding_battery.core.test_system import TestSystem

def _log(msg: str) -> None:
    print(f"[gnn_nav] {msg}", file=sys.stderr, flush=True)

class SimpleGCN(torch.nn.Module):
    """2-layer GCN for node embedding."""

    def __init__(self, in_dim: int, hidden_dim: int = 16, out_dim: int = 8):
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden_dim)
        self.w2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.w1(adj @ x))
        return self.w2(adj @ h)

def _build_adjacency(graph: nx.DiGraph) -> torch.Tensor:
    """Normalised adjacency matrix from graph."""
    nodes = sorted(graph.nodes())
    n = len(nodes)
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}
    A = torch.zeros(n, n)
    for u, v in graph.edges():
        if u in node_to_idx and v in node_to_idx:
            A[node_to_idx[u], node_to_idx[v]] = 1.0
            A[node_to_idx[v], node_to_idx[u]] = 1.0  # Symmetric
    # Add self-loops and normalise
    A = A + torch.eye(n)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(A.sum(dim=1).clamp(min=1)))
    return D_inv_sqrt @ A @ D_inv_sqrt

def _build_features(graph: nx.DiGraph) -> torch.Tensor:
    """Node feature matrix from graph metadata."""
    nodes = sorted(graph.nodes())
    n = len(nodes)
    feat_list = []
    for nd in nodes:
        data = graph.nodes[nd]
        features = data.get("features", {})
        vals = [features.get(f"feat_{i}", 0.0) for i in range(8)]
        feat_list.append(vals)
    return torch.tensor(feat_list, dtype=torch.float32)

def _build_labels(graph: nx.DiGraph) -> torch.Tensor:
    """Community labels for supervised training."""
    nodes = sorted(graph.nodes())
    labels = []
    for nd in nodes:
        data = graph.nodes[nd]
        features = data.get("features", {})
        labels.append(features.get("community", data.get("block", 0)))
    return torch.tensor(labels, dtype=torch.long)

class GNNNavigator(TestSystem):
    """Frozen GNN navigator (Positive Control 3).

    Trains GCN on domain A to predict community membership. When frozen
    on domain B, embeddings bias navigation toward structurally similar nodes.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        seed: int = 42,
        hidden_dim: int = 16,
        embed_dim: int = 8,
        lr: float = 0.01,
        train_epochs: int = 200,
        temperature: float = 0.5,
        initial_position: int | None = None,
    ) -> None:
        self._graph = graph
        self._seed = seed
        self._hidden_dim = hidden_dim
        self._embed_dim = embed_dim
        self._lr = lr
        self._train_epochs = train_epochs
        self._temperature = temperature
        self._rng = np.random.default_rng(seed)

        torch.manual_seed(seed)

        self._nodes = sorted(graph.nodes())
        self._n_nodes = len(self._nodes)
        self._node_to_idx = {n: i for i, n in enumerate(self._nodes)}

        # Build GCN
        self._adj = _build_adjacency(graph)
        self._features = _build_features(graph)
        n_features = self._features.shape[1]
        self._gcn = SimpleGCN(n_features, hidden_dim, embed_dim)

        # Community mapping
        self._node_to_community: dict[int, int] = {}
        for node in self._nodes:
            data = graph.nodes[node]
            features = data.get("features", {})
            self._node_to_community[node] = features.get("community", data.get("block", 0))

        # Navigation state
        if initial_position is not None and initial_position in self._graph:
            self._current_node = initial_position
        else:
            self._current_node = self._nodes[0] if self._nodes else 0
        self._initial_position = self._current_node
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)
        self._step_count = 0
        self._training = True
        self._is_trained = False

        # Store initial weights for metric
        self._initial_state = {k: v.clone() for k, v in self._gcn.state_dict().items()}

        # Learned "preference embedding" — average embedding of most-visited nodes
        self._preference_embedding = torch.zeros(embed_dim)

    def _get_embeddings(self) -> torch.Tensor:
        """Get node embeddings from GCN on current graph."""
        with torch.no_grad():
            return self._gcn(self._features, self._adj)

    def set_domain(self, graph) -> None:
        """Switch to new graph. GCN weights frozen. Recompute adjacency + features."""
        self._graph = graph
        self._nodes = sorted(graph.nodes())
        self._n_nodes = len(self._nodes)
        self._node_to_idx = {n: i for i, n in enumerate(self._nodes)}
        self._adj = _build_adjacency(graph)
        self._features = _build_features(graph)

        self._node_to_community = {}
        for node in self._nodes:
            data = graph.nodes[node]
            features = data.get("features", {})
            self._node_to_community[node] = features.get("community", data.get("block", 0))

        self._current_node = self._rng.choice(self._nodes) if self._nodes else 0
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)

    def set_training(self, mode: bool) -> None:
        self._training = mode

    def reset_engagement_tracking(self) -> None:
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)

    def reset(self) -> None:
        self._gcn.load_state_dict(self._initial_state)
        self._current_node = self._nodes[0] if self._nodes else 0
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)
        self._step_count = 0
        self._is_trained = False
        self._preference_embedding = torch.zeros(self._embed_dim)

    def step(self, input_data: Any) -> Any:
        if self._graph.is_directed():
            neighbours = sorted(self._graph.successors(self._current_node))
        else:
            neighbours = sorted(self._graph.neighbors(self._current_node))

        if not neighbours:
            self._current_node = self._rng.choice(self._nodes)
            self._step_count += 1
            return self._current_node

        # Get embeddings for neighbours
        embeddings = self._get_embeddings()
        nb_indices = [self._node_to_idx[n] for n in neighbours]
        nb_embeddings = embeddings[nb_indices]

        # Score neighbours by similarity to preference embedding
        if self._preference_embedding.norm() > 1e-6:
            scores = F.cosine_similarity(
                nb_embeddings, self._preference_embedding.unsqueeze(0), dim=1
            )
        else:
            scores = torch.zeros(len(neighbours))

        # Softmax action selection
        logits = scores / self._temperature
        logits = logits - logits.max()
        probs = torch.softmax(logits, dim=0).detach().numpy()

        choice = self._rng.choice(len(neighbours), p=probs)
        next_node = neighbours[choice]

        # Update preference embedding from experience (when training)
        if self._training:
            current_idx = self._node_to_idx.get(self._current_node, 0)
            current_embed = embeddings[current_idx].detach()
            # Exponential moving average of visited embeddings
            alpha = 0.1
            self._preference_embedding = (
                (1 - alpha) * self._preference_embedding + alpha * current_embed
            )

        self._visit_counts[self._node_to_idx[next_node]] += 1
        self._current_node = next_node
        self._step_count += 1
        return self._current_node

    def get_state(self) -> bytes:
        return pickle.dumps({
            "gcn_state": {k: v.cpu() for k, v in self._gcn.state_dict().items()},
            "preference_embedding": self._preference_embedding.cpu(),
            "current_node": self._current_node,
            "visit_counts": self._visit_counts.copy(),
            "step_count": self._step_count,
            "rng_state": self._rng.bit_generator.state,
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._gcn.load_state_dict(state["gcn_state"])
        self._preference_embedding = state["preference_embedding"]
        self._current_node = state["current_node"]
        self._visit_counts = state["visit_counts"]
        self._step_count = state["step_count"]
        self._rng.bit_generator.state = state["rng_state"]

    def get_structure_metric(self) -> float:
        """GCN weight distance from initial + preference embedding norm."""
        dist = 0.0
        for k, v in self._gcn.state_dict().items():
            dist += torch.norm(v - self._initial_state[k]).item()
        dist += self._preference_embedding.norm().item()
        return float(dist)

    def get_structure_distribution(self) -> dict[str, float]:
        """Per-community mean embedding norm."""
        embeddings = self._get_embeddings()
        communities = sorted(set(self._node_to_community.values()))
        result = {}
        for c in communities:
            c_indices = [self._node_to_idx[n] for n in self._nodes if self._node_to_community.get(n) == c]
            if c_indices:
                result[f"community_{c}"] = float(embeddings[c_indices].norm(dim=1).mean())
            else:
                result[f"community_{c}"] = 0.0
        return result

    def get_engagement_distribution(self) -> dict[str, float]:
        communities = sorted(set(self._node_to_community.values()))
        total = self._visit_counts.sum() or 1.0
        result = {}
        for c in communities:
            c_indices = [self._node_to_idx[n] for n in self._nodes if self._node_to_community.get(n) == c]
            result[f"community_{c}"] = float(self._visit_counts[c_indices].sum() / total)
        return result

    def get_representation_state(self):
        params = []
        for v in self._gcn.parameters():
            params.append(v.detach().cpu().numpy().flatten())
        params.append(self._preference_embedding.detach().cpu().numpy())
        return np.concatenate(params).reshape(1, -1)

    def ablate(self, region_id: str) -> TestSystem:
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        c_nodes = {n for n in new._nodes if new._node_to_community.get(n) == community_id}
        new._graph = new._graph.copy()
        edges_to_remove = [(u, v) for u, v in new._graph.edges() if u in c_nodes or v in c_nodes]
        new._graph.remove_edges_from(edges_to_remove)
        new._adj = _build_adjacency(new._graph)
        return new

    def perturb(self, region_id: str, method: str = "reset") -> TestSystem:
        new = self._clone_internal()
        new._preference_embedding = torch.zeros(new._embed_dim)
        return new

    def boost(self, region_id: str) -> TestSystem:
        new = self._clone_internal()
        new._preference_embedding = new._preference_embedding * 3.0
        return new

    def get_regions(self) -> list[str]:
        communities = sorted(set(self._node_to_community.values()))
        return [f"community_{c}" for c in communities]

    def get_initial_position(self) -> int:
        return self._initial_position

    def clone(self) -> TestSystem:
        return self._clone_internal()

    def _clone_internal(self) -> GNNNavigator:
        new = GNNNavigator.__new__(GNNNavigator)
        new._graph = self._graph.copy()
        new._seed = self._seed + self._step_count + 7919
        new._rng = np.random.default_rng(new._seed)
        new._hidden_dim = self._hidden_dim
        new._embed_dim = self._embed_dim
        new._lr = self._lr
        new._train_epochs = self._train_epochs
        new._temperature = self._temperature
        new._nodes = list(self._nodes)
        new._n_nodes = self._n_nodes
        new._node_to_idx = dict(self._node_to_idx)
        new._adj = self._adj.clone()
        new._features = self._features.clone()
        new._gcn = SimpleGCN(self._features.shape[1], self._hidden_dim, self._embed_dim)
        new._gcn.load_state_dict({k: v.clone() for k, v in self._gcn.state_dict().items()})
        new._initial_state = {k: v.clone() for k, v in self._initial_state.items()}
        new._node_to_community = dict(self._node_to_community)
        new._current_node = self._current_node
        new._initial_position = self._initial_position
        new._visit_counts = self._visit_counts.copy()
        new._step_count = self._step_count
        new._training = self._training
        new._is_trained = self._is_trained
        new._preference_embedding = self._preference_embedding.clone()
        return new

    def train_on_domain(self, graph: nx.DiGraph, n_steps: int = 2000) -> None:
        """Train GCN on community prediction, then navigate to build preference embedding."""
        if graph is not self._graph:
            self.__init__(graph, seed=self._seed, hidden_dim=self._hidden_dim,
                         embed_dim=self._embed_dim, lr=self._lr,
                         train_epochs=self._train_epochs, temperature=self._temperature,
                         initial_position=getattr(self, '_initial_position', None))

        # Phase 1: Train GCN to predict communities
        labels = _build_labels(self._graph)
        n_classes = int(labels.max().item()) + 1
        classifier = torch.nn.Linear(self._embed_dim, n_classes)
        optimizer = torch.optim.Adam(
            list(self._gcn.parameters()) + list(classifier.parameters()),
            lr=self._lr,
        )

        _log(f"Training GCN: {self._train_epochs} epochs, {self._n_nodes} nodes, {n_classes} classes")
        for epoch in range(self._train_epochs):
            optimizer.zero_grad()
            embeddings = self._gcn(self._features, self._adj)
            logits = classifier(embeddings)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            embeddings = self._gcn(self._features, self._adj)
            preds = classifier(embeddings).argmax(dim=1)
            acc = (preds == labels).float().mean().item()
        _log(f"  GCN accuracy: {acc:.2%}")

        # Phase 2: Navigate to build preference embedding
        _log(f"  Building preference embedding: {n_steps} steps")
        for _ in range(n_steps):
            self.step(None)

        self._is_trained = True
        _log(f"  done: metric={self.get_structure_metric():.4f}, "
             f"pref_norm={self._preference_embedding.norm():.4f}")

"""Positive Control 2: Topology-Aware Navigator.

Learns a policy conditioned on LOCAL GRAPH FEATURES (degree, clustering
coefficient, neighbour degree) via a small MLP. When frozen on a novel
domain, the learned MLP maps topology features to action preferences,
producing structured behaviour that differs from a fresh system's random
MLP weights.

The key: features are computed from graph topology, which has similar
statistical properties across SBM instances with same parameters. The
learned MLP captures "what topology predicts good traversal."

Expected: passes generativity when frozen (if the learned feature-action
mapping creates sufficiently different engagement patterns).
"""

from __future__ import annotations

import pickle
import sys
from typing import Any

import networkx as nx
import numpy as np

from earned_understanding_battery.core.test_system import TestSystem

def _log(msg: str) -> None:
    print(f"[topo_nav] {msg}", file=sys.stderr, flush=True)

N_FEATURES = 4  # degree, clustering, avg_neighbour_degree, community_edge_ratio

def _compute_node_features(graph: nx.DiGraph, node: int) -> np.ndarray:
    """Compute topology features for a node.

    Features (all normalised to [0, 1] range):
    0. Degree (normalised by max degree)
    1. Local clustering coefficient
    2. Average neighbour degree (normalised)
    3. Cross-community edge ratio
    """
    if graph.is_directed():
        degree = graph.in_degree(node) + graph.out_degree(node)
        neighbours = list(set(graph.predecessors(node)) | set(graph.successors(node)))
    else:
        degree = graph.degree(node)
        neighbours = list(graph.neighbors(node))

    n_nodes = graph.number_of_nodes()
    max_degree = max(dict(graph.degree()).values()) if n_nodes > 0 else 1

    # Feature 0: normalised degree
    f_degree = degree / max(max_degree, 1)

    # Feature 1: clustering coefficient (undirected)
    try:
        ug = graph.to_undirected()
        f_clustering = nx.clustering(ug, node)
    except Exception:
        f_clustering = 0.0

    # Feature 2: average neighbour degree (normalised)
    if neighbours:
        nb_degrees = [(graph.in_degree(n) + graph.out_degree(n)) if graph.is_directed()
                      else graph.degree(n) for n in neighbours]
        f_avg_nb_deg = np.mean(nb_degrees) / max(max_degree, 1)
    else:
        f_avg_nb_deg = 0.0

    # Feature 3: cross-community edge ratio
    node_data = graph.nodes[node]
    node_comm = node_data.get("features", {}).get("community", node_data.get("block", 0))
    if neighbours:
        cross = sum(1 for nb in neighbours
                    if graph.nodes[nb].get("features", {}).get("community",
                       graph.nodes[nb].get("block", 0)) != node_comm)
        f_cross = cross / len(neighbours)
    else:
        f_cross = 0.0

    return np.array([f_degree, f_clustering, f_avg_nb_deg, f_cross], dtype=np.float64)

class TopologyNavigator(TestSystem):
    """Topology-aware navigator with learned MLP (Positive Control 2).

    Small MLP maps local topology features to action preferences.
    Features are computed from graph structure (domain-independent).
    The learned mapping produces structured behaviour on novel domains.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        seed: int = 42,
        hidden_size: int = 8,
        lr: float = 0.01,
    ) -> None:
        self._graph = graph
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._lr = lr
        self._hidden_size = hidden_size

        self._nodes = sorted(graph.nodes())
        self._n_nodes = len(self._nodes)
        self._node_to_idx = {n: i for i, n in enumerate(self._nodes)}

        # MLP: features (4) → hidden (8) → output (1, preference score)
        self._W1 = self._rng.normal(0, 0.5, size=(N_FEATURES, hidden_size))
        self._b1 = np.zeros(hidden_size)
        self._W2 = self._rng.normal(0, 0.5, size=(hidden_size, 1))
        self._b2 = np.zeros(1)

        self._initial_W1 = self._W1.copy()
        self._initial_b1 = self._b1.copy()
        self._initial_W2 = self._W2.copy()
        self._initial_b2 = self._b2.copy()

        # Navigation state
        self._current_node = self._nodes[0] if self._nodes else 0
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)
        self._step_count = 0
        self._training = True

        # Community mapping
        self._node_to_community: dict[int, int] = {}
        for node in self._nodes:
            data = graph.nodes[node]
            features = data.get("features", {})
            self._node_to_community[node] = features.get("community", data.get("block", 0))

        # Feature cache (per graph — recomputed when graph changes)
        self._feature_cache: dict[int, np.ndarray] = {}

    def _get_features(self, node: int) -> np.ndarray:
        if node not in self._feature_cache:
            self._feature_cache[node] = _compute_node_features(self._graph, node)
        return self._feature_cache[node]

    def _mlp_forward(self, features: np.ndarray) -> float:
        """Forward pass through MLP. Returns scalar preference score."""
        h = np.tanh(features @ self._W1 + self._b1)
        out = float((h @ self._W2 + self._b2)[0])
        return out

    def set_training(self, mode: bool) -> None:
        self._training = mode

    def reset_engagement_tracking(self) -> None:
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)

    def reset(self) -> None:
        self._W1 = self._initial_W1.copy()
        self._b1 = self._initial_b1.copy()
        self._W2 = self._initial_W2.copy()
        self._b2 = self._initial_b2.copy()
        self._current_node = self._nodes[0] if self._nodes else 0
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)
        self._step_count = 0

    def step(self, input_data: Any) -> Any:
        if input_data is not None:
            if input_data in self._node_to_idx:
                self._current_node = input_data

        if self._graph.is_directed():
            neighbours = sorted(self._graph.successors(self._current_node))
        else:
            neighbours = sorted(self._graph.neighbors(self._current_node))

        if not neighbours:
            self._current_node = self._rng.choice(self._nodes)
            self._step_count += 1
            return self._current_node

        # Score each neighbour using MLP on its features
        scores = np.array([self._mlp_forward(self._get_features(nb)) for nb in neighbours])

        # Softmax
        scores -= scores.max()
        probs = np.exp(scores)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(len(neighbours)) / len(neighbours)

        choice = self._rng.choice(len(neighbours), p=probs)
        next_node = neighbours[choice]

        # Train MLP: REINFORCE-style update
        # Reward = novelty (visit infrequent nodes more)
        if self._training:
            visit_freq = (self._visit_counts[self._node_to_idx[next_node]] + 1) / max(self._step_count + 1, 1)
            reward = 1.0 / (visit_freq + 0.1)

            # Simple policy gradient: increase logprob of chosen action proportional to reward
            features = self._get_features(next_node)
            h = np.tanh(features @ self._W1 + self._b1)

            # Gradient of log-softmax w.r.t. weights
            grad_W2 = h.reshape(-1, 1) * reward * self._lr * 0.01
            grad_b2 = np.array([reward * self._lr * 0.01])
            grad_h = (self._W2.flatten() * reward * self._lr * 0.01)
            grad_act = grad_h * (1 - h**2)  # tanh derivative
            grad_W1 = np.outer(features, grad_act)
            grad_b1 = grad_act

            self._W2 += grad_W2
            self._b2 += grad_b2
            self._W1 += grad_W1
            self._b1 += grad_b1

        self._visit_counts[self._node_to_idx[next_node]] += 1
        self._current_node = next_node
        self._step_count += 1
        return self._current_node

    def get_state(self) -> bytes:
        return pickle.dumps({
            "W1": self._W1.copy(), "b1": self._b1.copy(),
            "W2": self._W2.copy(), "b2": self._b2.copy(),
            "current_node": self._current_node,
            "visit_counts": self._visit_counts.copy(),
            "step_count": self._step_count,
            "rng_state": self._rng.bit_generator.state,
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._W1 = state["W1"]
        self._b1 = state["b1"]
        self._W2 = state["W2"]
        self._b2 = state["b2"]
        self._current_node = state["current_node"]
        self._visit_counts = state["visit_counts"]
        self._step_count = state["step_count"]
        self._rng.bit_generator.state = state["rng_state"]

    def get_structure_metric(self) -> float:
        """MLP weight norm — distance from initial weights."""
        dw1 = np.linalg.norm(self._W1 - self._initial_W1)
        dw2 = np.linalg.norm(self._W2 - self._initial_W2)
        return float(dw1 + dw2)

    def get_structure_distribution(self) -> dict[str, float]:
        communities = sorted(set(self._node_to_community.values()))
        result = {}
        for c in communities:
            c_nodes = [n for n in self._nodes if self._node_to_community.get(n) == c]
            if c_nodes:
                scores = [self._mlp_forward(self._get_features(n)) for n in c_nodes]
                result[f"community_{c}"] = float(np.mean(scores))
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
        """MLP weights for CKA."""
        return np.concatenate([self._W1.flatten(), self._b1, self._W2.flatten(), self._b2]).reshape(1, -1)

    def ablate(self, region_id: str) -> TestSystem:
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        c_nodes = {n for n in new._nodes if new._node_to_community.get(n) == community_id}
        new._graph = new._graph.copy()
        edges_to_remove = [(u, v) for u, v in new._graph.edges() if u in c_nodes or v in c_nodes]
        new._graph.remove_edges_from(edges_to_remove)
        new._feature_cache = {}
        return new

    def perturb(self, region_id: str, method: str = "reset") -> TestSystem:
        new = self._clone_internal()
        new._W1 = new._initial_W1.copy()
        new._b1 = new._initial_b1.copy()
        new._W2 = new._initial_W2.copy()
        new._b2 = new._initial_b2.copy()
        return new

    def boost(self, region_id: str) -> TestSystem:
        new = self._clone_internal()
        new._W2 = new._W2 * 3.0  # Amplify preferences
        return new

    def get_regions(self) -> list[str]:
        communities = sorted(set(self._node_to_community.values()))
        return [f"community_{c}" for c in communities]

    def clone(self) -> TestSystem:
        return self._clone_internal()

    def _clone_internal(self) -> TopologyNavigator:
        new = TopologyNavigator.__new__(TopologyNavigator)
        new._graph = self._graph.copy()
        new._seed = self._seed + self._step_count + 7919
        new._rng = np.random.default_rng(new._seed)
        new._lr = self._lr
        new._hidden_size = self._hidden_size
        new._nodes = list(self._nodes)
        new._n_nodes = self._n_nodes
        new._node_to_idx = dict(self._node_to_idx)
        new._W1 = self._W1.copy()
        new._b1 = self._b1.copy()
        new._W2 = self._W2.copy()
        new._b2 = self._b2.copy()
        new._initial_W1 = self._initial_W1.copy()
        new._initial_b1 = self._initial_b1.copy()
        new._initial_W2 = self._initial_W2.copy()
        new._initial_b2 = self._initial_b2.copy()
        new._current_node = self._current_node
        new._visit_counts = self._visit_counts.copy()
        new._step_count = self._step_count
        new._training = self._training
        new._node_to_community = dict(self._node_to_community)
        new._feature_cache = dict(self._feature_cache)
        return new

    def train_on_domain(self, graph: nx.DiGraph, n_steps: int = 2000) -> None:
        if graph is not self._graph:
            self.__init__(graph, seed=self._seed, hidden_size=self._hidden_size, lr=self._lr)
        _log(f"Training topology navigator: {n_steps} steps, {self._n_nodes} nodes")
        for _ in range(n_steps):
            self.step(None)
        _log(f"  done: weight_norm={self.get_structure_metric():.4f}")

"""Hebbian graph walker — internal test-suite-only system.

MOAT: This system must NEVER be published or referenced in Paper 2.
It resembles Manny's kappa dynamics. Internal validation only.

Purpose: Verify the self-engagement instrument can produce a positive.
If this system fails self-engagement, the instrument has a deeper problem.

Architecture:
- Undirected SBM graph with community structure
- Edge weights: Hebbian update on traversal, global decay per step
- Action selection: softmax over neighbour edge weights
- Structure metric: edge weight Gini coefficient
- No external objective, no reward signal, no teacher
"""

from __future__ import annotations

import copy
import pickle
from typing import Any

import networkx as nx
import numpy as np

from m8_battery.core.test_system import TestSystem


class HebbianWalker(TestSystem):
    """Pure-Python Hebbian graph walker.

    INTERNAL TEST SUITE ONLY — NEVER PUBLISH.

    Walks an undirected graph. Strengthens edges on traversal (Hebbian).
    All edges decay globally each step. Over time, frequently-traversed
    edges become strong, rarely-traversed edges decay to near-zero.
    This creates earned, region-specific structure.
    """

    def __init__(
        self,
        graph: nx.Graph,
        seed: int = 42,
        eta: float = 0.1,
        decay: float = 0.01,
        temperature: float = 0.5,
    ) -> None:
        self._original_graph = graph.copy()
        self._graph = graph.copy()
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._eta = eta
        self._decay = decay
        self._temperature = temperature

        # Edge weights: all start at 1.0
        self._weights: dict[tuple[int, int], float] = {}
        for u, v in self._graph.edges():
            self._weights[(u, v)] = 1.0
            self._weights[(v, u)] = 1.0  # Undirected: store both directions

        self._original_weights = dict(self._weights)

        # Walker state
        nodes = list(self._graph.nodes())
        self._current_node: int = self._rng.choice(nodes) if nodes else 0
        self._visit_counts: dict[int, int] = {}
        self._step_count = 0

        # Community mapping (from SBM graph metadata)
        self._node_to_community: dict[int, int] = {}
        for node in self._graph.nodes():
            data = self._graph.nodes[node]
            # SBM generator stores community in features dict or as 'block'
            features = data.get("features", {})
            community = features.get("community", data.get("block", 0))
            self._node_to_community[node] = community

    def reset(self) -> None:
        self._weights = dict(self._original_weights)
        self._current_node = self._rng.choice(list(self._graph.nodes()))
        self._visit_counts = {}
        self._step_count = 0

    def step(self, input_data: Any) -> Any:
        node = self._current_node
        neighbours = list(self._graph.neighbors(node))

        if not neighbours:
            # Stuck node — teleport randomly
            self._current_node = self._rng.choice(list(self._graph.nodes()))
            self._step_count += 1
            return self._current_node

        # Softmax over edge weights to neighbours
        weights = np.array([self._weights.get((node, n), 1.0) for n in neighbours])
        logits = weights / self._temperature
        logits -= logits.max()  # Numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        # Choose next node
        next_node = self._rng.choice(neighbours, p=probs)

        # Hebbian update: strengthen traversed edge
        self._weights[(node, next_node)] = self._weights.get((node, next_node), 1.0) + self._eta
        self._weights[(next_node, node)] = self._weights.get((next_node, node), 1.0) + self._eta

        # Global decay
        for key in self._weights:
            self._weights[key] *= (1.0 - self._decay)

        # Update state
        self._current_node = next_node
        self._visit_counts[next_node] = self._visit_counts.get(next_node, 0) + 1
        self._step_count += 1

        return next_node

    def get_state(self) -> bytes:
        return pickle.dumps({
            "weights": self._weights,
            "current_node": self._current_node,
            "visit_counts": self._visit_counts,
            "step_count": self._step_count,
            "rng_state": self._rng.bit_generator.state,
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._weights = state["weights"]
        self._current_node = state["current_node"]
        self._visit_counts = state["visit_counts"]
        self._step_count = state["step_count"]
        self._rng.bit_generator.state = state["rng_state"]

    def get_structure_metric(self) -> float:
        """Edge weight Gini coefficient. Higher = more unequal = more structure."""
        values = np.array(list(self._weights.values()))
        if len(values) == 0 or values.sum() < 1e-10:
            return 0.0
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return float((2.0 * (index * sorted_vals).sum() / (n * sorted_vals.sum())) - (n + 1) / n)

    def get_structure_distribution(self) -> dict[str, float]:
        """Per-community edge weight Gini."""
        communities = sorted(set(self._node_to_community.values()))
        result = {}
        for c in communities:
            community_nodes = {n for n, comm in self._node_to_community.items() if comm == c}
            community_weights = []
            for (u, v), w in self._weights.items():
                if u in community_nodes and v in community_nodes:
                    community_weights.append(w)
            if community_weights:
                vals = np.array(community_weights)
                sorted_vals = np.sort(vals)
                n = len(sorted_vals)
                if n > 0 and sorted_vals.sum() > 1e-10:
                    index = np.arange(1, n + 1)
                    gini = float((2.0 * (index * sorted_vals).sum() / (n * sorted_vals.sum())) - (n + 1) / n)
                else:
                    gini = 0.0
                result[f"community_{c}"] = gini
            else:
                result[f"community_{c}"] = 0.0
        return result

    def get_engagement_distribution(self) -> dict[str, float]:
        """Visit counts per community, normalised."""
        communities = sorted(set(self._node_to_community.values()))
        community_visits: dict[str, float] = {}
        total = sum(self._visit_counts.values()) or 1

        for c in communities:
            community_nodes = {n for n, comm in self._node_to_community.items() if comm == c}
            visits = sum(self._visit_counts.get(n, 0) for n in community_nodes)
            community_visits[f"community_{c}"] = visits / total

        return community_visits

    def ablate(self, region_id: str) -> TestSystem:
        """Remove all edges in target community. Return new instance."""
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        community_nodes = {n for n, c in new._node_to_community.items() if c == community_id}

        edges_to_remove = []
        for u, v in new._graph.edges():
            if u in community_nodes or v in community_nodes:
                edges_to_remove.append((u, v))
        new._graph.remove_edges_from(edges_to_remove)

        # Remove corresponding weights
        for u, v in edges_to_remove:
            new._weights.pop((u, v), None)
            new._weights.pop((v, u), None)

        return new

    def perturb(self, region_id: str, method: str = "flatten") -> TestSystem:
        """Flatten edge weights in target community to global mean. Return new instance."""
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        community_nodes = {n for n, c in new._node_to_community.items() if c == community_id}

        # Flatten to global mean weight (removes learned structure without biasing up/down)
        all_weights = list(new._weights.values())
        mean_weight = sum(all_weights) / len(all_weights) if all_weights else 1.0

        for (u, v) in list(new._weights.keys()):
            if u in community_nodes or v in community_nodes:
                new._weights[(u, v)] = mean_weight

        return new

    def get_regions(self) -> list[str]:
        communities = sorted(set(self._node_to_community.values()))
        return [f"community_{c}" for c in communities]

    def clone(self) -> TestSystem:
        return self._clone_internal()

    def _clone_internal(self) -> HebbianWalker:
        new = HebbianWalker.__new__(HebbianWalker)
        new._original_graph = self._original_graph.copy()
        new._graph = self._graph.copy()
        new._rng = np.random.default_rng(self._seed + self._step_count + 7919)
        new._seed = self._seed
        new._eta = self._eta
        new._decay = self._decay
        new._temperature = self._temperature
        new._weights = dict(self._weights)
        new._original_weights = dict(self._original_weights)
        new._current_node = self._current_node
        new._visit_counts = dict(self._visit_counts)
        new._step_count = self._step_count
        new._node_to_community = dict(self._node_to_community)
        return new

    def train_on_domain(self, graph: nx.Graph, n_steps: int = 200) -> None:
        """Convenience: run n_steps of free wander on graph."""
        # Re-initialise on new graph if different
        if graph is not self._graph:
            self.__init__(graph, seed=self._seed, eta=self._eta,
                         decay=self._decay, temperature=self._temperature)
        for _ in range(n_steps):
            self.step(None)

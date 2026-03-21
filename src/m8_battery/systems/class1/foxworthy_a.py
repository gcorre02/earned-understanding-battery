"""System 1C — Foxworthy Variant A (Stateless MLP).

A Class 1 system: fixed-weight MLP, no learning, no state.
Source: Foxworthy (2026) section 2.7.

Expected battery result: NEGATIVE on all instruments.
Passes 0/4 Foxworthy diagnostics.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

from m8_battery.core.test_system import TestSystem


class FoxworthyA(TestSystem):
    """Stateless frozen MLP navigator.

    Architecture: input_dim → 20 → 5 output.
    Random init then frozen. No learning, no state updates.
    Output = argmax of MLP forward pass on node features.
    """

    def __init__(self, n_features: int = 8, seed: int = 42) -> None:
        self._seed = seed
        self._n_features = n_features
        rng = np.random.default_rng(seed)

        # Fixed random weights (never updated)
        self._w1 = rng.standard_normal((n_features, 20)) * 0.1
        self._b1 = np.zeros(20)
        self._w2 = rng.standard_normal((20, 5)) * 0.1
        self._b2 = np.zeros(5)

        self._graph: Any = None
        self._current_node: int | None = None
        self._step_count = 0
        self._visit_counts: dict[int, int] = {}

    def set_graph(self, graph) -> None:
        """Attach a graph for navigation. Not part of TestSystem ABC."""
        import networkx as nx
        self._graph = graph

    def set_domain(self, graph) -> None:
        """Switch to a new graph domain (near re-init for Class 1).

        Delegates to set_graph() and resets navigation state.
        No learned state to preserve (MLP weights are frozen at init).
        """
        self.set_graph(graph)
        self._current_node = None
        self._step_count = 0
        self._visit_counts = {}

    def reset(self) -> None:
        self._current_node = None
        self._step_count = 0
        self._visit_counts = {}

    def step(self, input_data: Any) -> Any:
        if self._graph is None:
            return {"error": "no graph attached"}

        nodes = list(self._graph.nodes())
        if not nodes:
            return {"error": "empty graph"}

        if self._current_node is None:
            self._current_node = nodes[0]

        # Get current node features
        features = self._graph.nodes[self._current_node].get("features", {})
        x = np.array([features.get(f"feat_{i}", 0.0) for i in range(self._n_features)])

        # Forward pass (frozen MLP)
        h = np.maximum(0, x @ self._w1 + self._b1)  # ReLU
        out = h @ self._w2 + self._b2

        # Use output to select next node
        successors = list(self._graph.successors(self._current_node))
        if successors:
            # Map output to successor selection
            idx = int(np.argmax(out)) % len(successors)
            self._current_node = successors[idx]

        self._visit_counts[self._current_node] = (
            self._visit_counts.get(self._current_node, 0) + 1
        )
        self._step_count += 1

        return {
            "current_node": self._current_node,
            "step": self._step_count,
            "mlp_output": out.tolist(),
        }

    def get_state(self) -> bytes:
        return pickle.dumps({
            "current_node": self._current_node,
            "step_count": self._step_count,
            "visit_counts": self._visit_counts,
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._current_node = state["current_node"]
        self._step_count = state["step_count"]
        self._visit_counts = state["visit_counts"]

    def get_structure_metric(self) -> float:
        # Constant — MLP weights never change
        return float(np.linalg.norm(self._w1) + np.linalg.norm(self._w2))

    def get_structure_distribution(self) -> dict[str, float]:
        if self._graph is None:
            return {}
        communities: dict[int, list[int]] = {}
        for node in self._graph.nodes():
            block = self._graph.nodes[node].get("features", {}).get("community", 0)
            communities.setdefault(block, []).append(node)
        # Weight norm contribution per community (constant)
        return {f"community_{c}": float(len(ns)) for c, ns in communities.items()}

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
        new = FoxworthyA(n_features=self._n_features, seed=self._seed)
        if self._graph is not None:
            comm_id = int(region_id.split("_")[-1])
            new_graph = self._graph.copy()
            nodes_to_remove = [
                n for n in new_graph.nodes()
                if new_graph.nodes[n].get("features", {}).get("community", -1) == comm_id
            ]
            new_graph.remove_nodes_from(nodes_to_remove)
            new.set_graph(new_graph)
        return new

    def perturb(self, region_id: str, method: str = "shuffle_weights") -> TestSystem:
        new = FoxworthyA(n_features=self._n_features, seed=self._seed)
        if self._graph is not None:
            new_graph = self._graph.copy()
            rng = np.random.default_rng(self._seed + 999)
            comm_id = int(region_id.split("_")[-1])
            comm_nodes = set(
                n for n in new_graph.nodes()
                if new_graph.nodes[n].get("features", {}).get("community", -1) == comm_id
            )
            for u, v in new_graph.edges():
                if u in comm_nodes or v in comm_nodes:
                    new_graph.edges[u, v]["weight"] = rng.uniform(0.1, 1.0)
            new.set_graph(new_graph)
        return new

    def get_regions(self) -> list[str]:
        if self._graph is None:
            return []
        communities = set()
        for node in self._graph.nodes():
            block = self._graph.nodes[node].get("features", {}).get("community", 0)
            communities.add(block)
        return [f"community_{c}" for c in sorted(communities)]

    def clone(self) -> TestSystem:
        new = FoxworthyA(n_features=self._n_features, seed=self._seed)
        if self._graph is not None:
            new.set_graph(self._graph)
        return new

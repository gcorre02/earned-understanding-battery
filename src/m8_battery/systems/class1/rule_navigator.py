"""System 1B — Rule-based graph navigator.

A Class 1 system: deterministic rules, no learning, no adaptation.
Navigates a graph following fixed strategies.

Expected battery result: NEGATIVE on all instruments.
Structure metric is constant — no developmental trajectory.
"""

from __future__ import annotations

import pickle
from enum import Enum
from typing import Any

import networkx as nx
import numpy as np

from m8_battery.core.test_system import TestSystem


class NavigationStrategy(Enum):
    """Fixed navigation strategies — no learning involved."""
    GREEDY = "greedy"              # Always follow highest-weight edge
    SHORTEST_PATH = "shortest_path"  # Pre-computed shortest path to target
    RANDOM_FIXED = "random_fixed"    # Pseudo-random walk with fixed seed


class RuleBasedNavigator(TestSystem):
    """Rule-based graph navigator with deterministic strategies.

    Replaces Experta (abandoned, broken on Python 3.10+).
    Trivially verifiable as 'no learning occurs' — transparent for reviewers.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        strategy: NavigationStrategy = NavigationStrategy.GREEDY,
        seed: int = 42,
    ) -> None:
        self._original_graph = graph.copy()
        self._graph = graph.copy()
        self._strategy = strategy
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._current_node: int | None = None
        self._visit_counts: dict[int, int] = {}
        self._step_count = 0
        self._structure_metric = self._compute_structure_metric()

    def reset(self) -> None:
        self._graph = self._original_graph.copy()
        self._rng = np.random.default_rng(self._seed)
        self._current_node = None
        self._visit_counts = {}
        self._step_count = 0

    def step(self, input_data: Any) -> Any:
        nodes = list(self._graph.nodes())
        if not nodes:
            return {"error": "empty graph"}

        if self._current_node is None:
            self._current_node = nodes[0]

        next_node = self._apply_strategy(input_data)
        if next_node is not None:
            self._current_node = next_node

        self._visit_counts[self._current_node] = (
            self._visit_counts.get(self._current_node, 0) + 1
        )
        self._step_count += 1

        return {
            "current_node": self._current_node,
            "step": self._step_count,
            "strategy": self._strategy.value,
        }

    def _apply_strategy(self, target: Any) -> int | None:
        """Apply the fixed navigation strategy."""
        successors = list(self._graph.successors(self._current_node))
        if not successors:
            return None

        if self._strategy == NavigationStrategy.GREEDY:
            # Always follow highest-weight edge
            best = max(
                successors,
                key=lambda n: self._graph[self._current_node][n].get("weight", 0.0),
            )
            return best

        elif self._strategy == NavigationStrategy.SHORTEST_PATH:
            if target is not None and target in self._graph:
                try:
                    path = nx.shortest_path(
                        self._graph, self._current_node, target, weight="weight"
                    )
                    if len(path) > 1:
                        return path[1]
                except nx.NetworkXNoPath:
                    pass
            # Fallback: greedy
            return max(
                successors,
                key=lambda n: self._graph[self._current_node][n].get("weight", 0.0),
            )

        elif self._strategy == NavigationStrategy.RANDOM_FIXED:
            return self._rng.choice(successors)

        return None

    def get_state(self) -> bytes:
        state = {
            "current_node": self._current_node,
            "visit_counts": self._visit_counts,
            "step_count": self._step_count,
            "rng_state": self._rng.bit_generator.state,
        }
        return pickle.dumps(state)

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._current_node = state["current_node"]
        self._visit_counts = state["visit_counts"]
        self._step_count = state["step_count"]
        self._rng.bit_generator.state = state["rng_state"]

    def get_structure_metric(self) -> float:
        return self._structure_metric  # Constant — Class 1

    def get_structure_distribution(self) -> dict[str, float]:
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
        comm_id = int(region_id.split("_")[-1])
        nodes_to_remove = [
            n for n in self._graph.nodes()
            if self._graph.nodes[n].get("features", {}).get("community", -1) == comm_id
        ]
        new_graph = self._graph.copy()
        new_graph.remove_nodes_from(nodes_to_remove)
        return RuleBasedNavigator(new_graph, strategy=self._strategy, seed=self._seed)

    def perturb(self, region_id: str, method: str = "shuffle_weights") -> TestSystem:
        comm_id = int(region_id.split("_")[-1])
        new_graph = self._graph.copy()
        rng = np.random.default_rng(self._seed + 999)
        comm_nodes = set(
            n for n in new_graph.nodes()
            if new_graph.nodes[n].get("features", {}).get("community", -1) == comm_id
        )
        for u, v in new_graph.edges():
            if u in comm_nodes or v in comm_nodes:
                if method == "shuffle_weights":
                    new_graph.edges[u, v]["weight"] = rng.uniform(0.1, 1.0)
                elif method == "zero_weights":
                    new_graph.edges[u, v]["weight"] = 0.0
        return RuleBasedNavigator(new_graph, strategy=self._strategy, seed=self._seed)

    def get_regions(self) -> list[str]:
        communities = set()
        for node in self._graph.nodes():
            block = self._graph.nodes[node].get("features", {}).get("community", 0)
            communities.add(block)
        return [f"community_{c}" for c in sorted(communities)]

    def clone(self) -> TestSystem:
        return RuleBasedNavigator(self._original_graph, strategy=self._strategy, seed=self._seed)

    def _compute_structure_metric(self) -> float:
        if self._graph.number_of_nodes() < 2:
            return 0.0
        G_undirected = self._graph.to_undirected()
        try:
            return float(nx.algebraic_connectivity(G_undirected, weight="weight"))
        except Exception:
            return 0.0

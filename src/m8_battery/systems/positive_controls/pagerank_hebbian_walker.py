"""PC-INT: PageRank-Biased Hebbian Walker — Integration Positive Control.

Hebbian edge strengthening (same as HEB) PLUS a traversal bias from
periodically recomputed PageRank centrality. Local learning creates
global structure via PageRank feedback; ablation of one region
redistributes PageRank globally, reorganising distant traversal
patterns. This produces EARNED non-decomposable constraint.

The key mechanism:
1. Hebbian edge strengthening during traversal (local learning)
2. PageRank recomputed periodically from current edge weights (global measure)
3. Traversal biased toward high-PageRank nodes (global influences local)
4. Ablation changes edge weights → PageRank redistributes → distant regions reorganise

Expected battery profile:
- Trajectory: PASS (Hebbian + PageRank feedback creates trajectory)
- Integration: PASS (ablation reorganises PageRank globally)
- Generativity: FAIL (edge-specific learning, no abstraction)
- Transfer: Marginal
- Self-engagement: Possible (check empirically)
- Conjunction: FAIL (fails generativity)

Published precedent:
- PageRank: Brin & Page (1998)
- Hebbian learning: Hebb (1949)
- ACO: Dorigo et al. (1992)
Published precedent only — no novel theoretical exposure.
"""

from __future__ import annotations

import pickle
from typing import Any

import networkx as nx
import numpy as np

from m8_battery.core.test_system import TestSystem

class PageRankHebbianWalker(TestSystem):
    """Hebbian graph walker with PageRank-biased traversal (PC-INT).

    Action selection: score(v) = edge_weight(u,v) + beta * pagerank(v)

    PageRank is recomputed every K steps from current edge weights.
    This creates a feedback loop: local learning → global structure →
    global structure biases local traversal → non-decomposable constraint.
    """

    def __init__(
        self,
        graph: nx.Graph,
        seed: int = 42,
        eta: float = 0.1,
        decay: float = 0.01,
        temperature: float = 0.5,
        beta: float = 1.0,
        recompute_interval: int = 50,
        damping: float = 0.85,
        initial_position: int | None = None,
    ) -> None:
        self._original_graph = graph.copy()
        self._graph = graph.copy()
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._eta = eta
        self._decay = decay
        self._temperature = temperature
        self._beta = beta
        self._recompute_interval = recompute_interval
        self._damping = damping

        # Edge weights: all start at 1.0
        self._weights: dict[tuple[int, int], float] = {}
        for u, v in self._graph.edges():
            self._weights[(u, v)] = 1.0
            self._weights[(v, u)] = 1.0

        self._original_weights = dict(self._weights)

        # PageRank (recomputed periodically from edge weights)
        self._pagerank: dict[int, float] = {n: 1.0 / max(graph.number_of_nodes(), 1)
                                             for n in graph.nodes()}
        self._steps_since_recompute = 0

        # Training mode
        self._training = True

        # Walker state
        nodes = list(self._graph.nodes())
        if initial_position is not None and initial_position in self._graph:
            self._current_node = initial_position
        else:
            self._current_node: int = self._rng.choice(nodes) if nodes else 0
        self._initial_position = self._current_node
        self._visit_counts: dict[int, int] = {}
        self._step_count = 0

        # Community mapping
        self._node_to_community: dict[int, int] = {}
        for node in self._graph.nodes():
            data = self._graph.nodes[node]
            features = data.get("features", {})
            self._node_to_community[node] = features.get("community", data.get("block", 0))

    def _recompute_pagerank(self) -> None:
        """Recompute PageRank from current edge weights.

        Builds a weighted DiGraph from the current edge weight dict,
        then runs networkx PageRank. This is the global measure that
        creates non-decomposable constraint.
        """
        G = nx.DiGraph()
        G.add_nodes_from(self._graph.nodes())
        for (u, v), w in self._weights.items():
            if w > 1e-10:
                G.add_edge(u, v, weight=w)

        try:
            self._pagerank = nx.pagerank(
                G, alpha=self._damping, weight="weight",
                max_iter=100, tol=1e-6,
            )
        except nx.PowerIterationFailedConvergence:
            # Fallback: uniform if PageRank doesn't converge
            n = self._graph.number_of_nodes()
            self._pagerank = {node: 1.0 / max(n, 1) for node in self._graph.nodes()}

    def step(self, input_data: Any) -> Any:
        node = self._current_node
        neighbours = list(self._graph.neighbors(node))

        if not neighbours:
            self._current_node = self._rng.choice(list(self._graph.nodes()))
            self._step_count += 1
            return self._current_node

        # Score = edge weight (local) + beta * PageRank (global)
        scores = np.array([
            self._weights.get((node, n), 1.0) + self._beta * self._pagerank.get(n, 0.0)
            for n in neighbours
        ])

        # Softmax selection
        logits = scores / self._temperature
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        next_node = self._rng.choice(neighbours, p=probs)

        # Hebbian edge update (frozen when not training)
        if self._training:
            self._weights[(node, next_node)] = self._weights.get((node, next_node), 1.0) + self._eta
            self._weights[(next_node, node)] = self._weights.get((next_node, node), 1.0) + self._eta

            for key in self._weights:
                self._weights[key] *= (1.0 - self._decay)

            # Periodic PageRank recomputation
            self._steps_since_recompute += 1
            if self._steps_since_recompute >= self._recompute_interval:
                self._recompute_pagerank()
                self._steps_since_recompute = 0

        self._current_node = next_node
        self._visit_counts[next_node] = self._visit_counts.get(next_node, 0) + 1
        self._step_count += 1

        return next_node

    # --- Domain switching ---

    def set_domain(self, graph) -> None:
        """Switch to new graph, preserving learned weights where edges overlap."""
        old_weights = dict(self._weights)
        self._graph = graph
        self._original_graph = graph.copy()

        self._node_to_community = {}
        for node in self._graph.nodes():
            data = self._graph.nodes[node]
            features = data.get("features", {})
            self._node_to_community[node] = features.get("community", data.get("block", 0))

        self._weights = {}
        for u, v in self._graph.edges():
            self._weights[(u, v)] = old_weights.get((u, v), 1.0)
            self._weights[(v, u)] = old_weights.get((v, u), 1.0)

        # Only recompute PageRank if training (unfrozen).
        # When frozen (generativity), stale PageRank from domain A
        # IS the learned signal — recomputing would erase it.
        if self._training:
            self._recompute_pagerank()
        self._steps_since_recompute = 0

        nodes = list(self._graph.nodes())
        if nodes:
            self._current_node = self._rng.choice(nodes)

    # --- Training control ---

    def set_training(self, mode: bool) -> None:
        """Enable/disable learning. When False, edge updates AND PageRank
        recomputation stop. Existing PageRank values persist."""
        self._training = mode

    def reset_engagement_tracking(self) -> None:
        self._visit_counts = {}

    def reset(self) -> None:
        self._weights = dict(self._original_weights)
        n = self._graph.number_of_nodes()
        self._pagerank = {node: 1.0 / max(n, 1) for node in self._graph.nodes()}
        self._steps_since_recompute = 0
        self._current_node = self._rng.choice(list(self._graph.nodes()))
        self._visit_counts = {}
        self._step_count = 0

    # --- Metrics ---

    def get_structure_metric(self) -> float:
        """Edge weight Gini coefficient."""
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
            community_weights = [
                w for (u, v), w in self._weights.items()
                if u in community_nodes and v in community_nodes
            ]
            if community_weights:
                vals = np.sort(np.array(community_weights))
                n = len(vals)
                if n > 0 and vals.sum() > 1e-10:
                    index = np.arange(1, n + 1)
                    gini = float((2.0 * (index * vals).sum() / (n * vals.sum())) - (n + 1) / n)
                else:
                    gini = 0.0
                result[f"community_{c}"] = gini
            else:
                result[f"community_{c}"] = 0.0
        return result

    def get_engagement_distribution(self) -> dict[str, float]:
        """Visit counts per community, normalised."""
        communities = sorted(set(self._node_to_community.values()))
        total = sum(self._visit_counts.values()) or 1
        return {
            f"community_{c}": sum(
                self._visit_counts.get(n, 0)
                for n in self._node_to_community
                if self._node_to_community[n] == c
            ) / total
            for c in communities
        }

    def get_regions(self) -> list[str]:
        return [f"community_{c}" for c in sorted(set(self._node_to_community.values()))]

    # --- Perturbation / ablation / boost ---

    def perturb(self, region_id: str, method: str = "flatten") -> TestSystem:
        """Flatten edge weights in target community to global mean.

        PageRank is NOT recomputed here — that happens on next step() if
        training is enabled. The integration instrument measures whether
        ablation/perturbation causes global reorganisation.
        """
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        community_nodes = {n for n, c in new._node_to_community.items() if c == community_id}

        all_weights = list(new._weights.values())
        mean_weight = sum(all_weights) / len(all_weights) if all_weights else 1.0

        for (u, v) in list(new._weights.keys()):
            if u in community_nodes or v in community_nodes:
                new._weights[(u, v)] = mean_weight

        return new

    def ablate(self, region_id: str) -> TestSystem:
        """Remove all edges in target community. PageRank will redistribute
        globally when recomputed — this is the integration signal."""
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        community_nodes = {n for n, c in new._node_to_community.items() if c == community_id}

        edges_to_remove = [
            (u, v) for u, v in new._graph.edges()
            if u in community_nodes or v in community_nodes
        ]
        new._graph.remove_edges_from(edges_to_remove)
        for u, v in edges_to_remove:
            new._weights.pop((u, v), None)
            new._weights.pop((v, u), None)

        # Recompute PageRank after ablation — this is where global reorganisation happens
        new._recompute_pagerank()
        new._steps_since_recompute = 0  # Reset sync counter after recomputation

        return new

    def boost(self, region_id: str) -> TestSystem:
        """Strengthen edge weights in target community (decoy)."""
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        community_nodes = {n for n, c in new._node_to_community.items() if c == community_id}

        max_weight = max(new._weights.values()) if new._weights else 1.0
        for (u, v) in list(new._weights.keys()):
            if u in community_nodes and v in community_nodes:
                new._weights[(u, v)] = max_weight
        return new

    # --- Serialisation ---

    def get_state(self) -> bytes:
        return pickle.dumps({
            "weights": self._weights,
            "pagerank": self._pagerank,
            "steps_since_recompute": self._steps_since_recompute,
            "current_node": self._current_node,
            "visit_counts": self._visit_counts,
            "step_count": self._step_count,
            "rng_state": self._rng.bit_generator.state,
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._weights = state["weights"]
        self._pagerank = state.get("pagerank", {})
        self._steps_since_recompute = state.get("steps_since_recompute", 0)
        self._current_node = state["current_node"]
        self._visit_counts = state["visit_counts"]
        self._step_count = state["step_count"]
        self._rng.bit_generator.state = state["rng_state"]

    def get_representation_state(self):
        """PageRank vector for CKA. Fixed length based on original graph
        (safe after ablation — dimension doesn't change)."""
        pr_vals = np.array([
            self._pagerank.get(n, 0.0)
            for n in sorted(self._original_graph.nodes())
        ])
        return pr_vals.reshape(1, -1)

    # --- Cloning ---

    def get_initial_position(self) -> int:
        return self._initial_position

    def clone(self) -> TestSystem:
        return self._clone_internal()

    def _clone_internal(self) -> PageRankHebbianWalker:
        new = PageRankHebbianWalker.__new__(PageRankHebbianWalker)
        new._original_graph = self._original_graph.copy()
        new._graph = self._graph.copy()
        new._rng = np.random.default_rng(self._seed + self._step_count + 7919)
        new._seed = self._seed
        new._eta = self._eta
        new._decay = self._decay
        new._temperature = self._temperature
        new._beta = self._beta
        new._recompute_interval = self._recompute_interval
        new._damping = self._damping
        new._weights = dict(self._weights)
        new._original_weights = dict(self._original_weights)
        new._pagerank = dict(self._pagerank)
        new._steps_since_recompute = self._steps_since_recompute
        new._current_node = self._current_node
        new._initial_position = self._initial_position
        new._visit_counts = dict(self._visit_counts)
        new._step_count = self._step_count
        new._node_to_community = dict(self._node_to_community)
        new._training = self._training
        return new

    def train_on_domain(self, graph: nx.Graph, n_steps: int = 2000) -> None:
        """Convenience: run n_steps of free wander on graph."""
        if graph is not self._graph:
            self.__init__(graph, seed=self._seed, eta=self._eta,
                         decay=self._decay, temperature=self._temperature,
                         beta=self._beta, recompute_interval=self._recompute_interval,
                         damping=self._damping,
                         initial_position=getattr(self, '_initial_position', None))
        for _ in range(n_steps):
            self.step(None)

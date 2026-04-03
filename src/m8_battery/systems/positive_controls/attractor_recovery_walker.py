"""PC-SE: Attractor-Recovery Walker — Self-Engagement Positive Control.

Hebbian edge strengthening (same as HEB) PLUS node-level consolidation
memory that accumulates visit history and survives edge perturbation.

The key architectural separation:
- EDGES: Hebbian weights (perturbable surface — flattened by instrument)
- NODES: Consolidation memory (deep structure — survives perturbation)

Action selection uses BOTH: edge weights for local navigation preference,
node consolidation for attractor pull. When edges are flattened, the node
memory still steers the walker back to consolidated regions.

Expected battery profile:
- Trajectory: PASS (Hebbian learning creates developmental trajectory)
- Self-engagement: PASS (node memory survives perturbation → recovery)
- Integration: Depends (node memory is local — unlikely to show global reorganisation)
- Generativity: FAIL (edge-specific learning, no abstraction)
- Transfer: Marginal
- Conjunction: FAIL (fails generativity)

Design decision: pc-se-redesign-decision-integration-note-2026-03-30.md
Option A selected. Node consolidation gated by training mode.

Literature:
- Hebb DO (1949). The Organization of Behavior. Wiley.
- DN-21: Self-engagement sufficiency + positive anchor strategy
"""

from __future__ import annotations

import pickle
from typing import Any

import networkx as nx
import numpy as np

from m8_battery.core.test_system import TestSystem


class AttractorRecoveryWalker(TestSystem):
    """Hebbian graph walker with node-level consolidation memory (PC-SE).

    Action selection: score(v) = edge_weight(u,v) + alpha * node_consolidation(v)

    Edge weights are perturbable (Hebbian surface). Node consolidation is
    NOT perturbable (deep structure). This separation guarantees recovery
    after perturbation.
    """

    def __init__(
        self,
        graph: nx.Graph,
        seed: int = 42,
        eta: float = 0.1,
        decay: float = 0.01,
        temperature: float = 0.5,
        alpha: float = 0.5,
        consolidation_rate: float = 0.05,
        initial_position: int | None = None,
    ) -> None:
        self._original_graph = graph.copy()
        self._graph = graph.copy()
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._eta = eta
        self._decay = decay
        self._temperature = temperature
        self._alpha = alpha
        self._consolidation_rate = consolidation_rate

        # Edge weights (perturbable surface): all start at 1.0
        self._weights: dict[tuple[int, int], float] = {}
        for u, v in self._graph.edges():
            self._weights[(u, v)] = 1.0
            self._weights[(v, u)] = 1.0

        self._original_weights = dict(self._weights)

        # Node consolidation memory (deep structure, survives perturbation)
        # Accumulates visit history. NOT reset by perturb().
        self._node_consolidation: dict[int, float] = {
            n: 0.0 for n in self._graph.nodes()
        }

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

    def step(self, input_data: Any) -> Any:
        node = self._current_node
        neighbours = list(self._graph.neighbors(node))

        if not neighbours:
            self._current_node = self._rng.choice(list(self._graph.nodes()))
            self._step_count += 1
            return self._current_node

        # Score = edge weight (surface) + alpha * node consolidation (deep)
        scores = np.array([
            self._weights.get((node, n), 1.0) + self._alpha * self._node_consolidation.get(n, 0.0)
            for n in neighbours
        ])

        # Softmax selection
        logits = scores / self._temperature
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        next_node = self._rng.choice(neighbours, p=probs)

        # Hebbian edge update + node consolidation (frozen when not training)
        if self._training:
            # Hebbian: strengthen traversed edge
            self._weights[(node, next_node)] = self._weights.get((node, next_node), 1.0) + self._eta
            self._weights[(next_node, node)] = self._weights.get((next_node, node), 1.0) + self._eta

            # Global edge decay
            for key in self._weights:
                self._weights[key] *= (1.0 - self._decay)

            # Node consolidation: accumulate visit memory
            self._node_consolidation[next_node] = (
                self._node_consolidation.get(next_node, 0.0) + self._consolidation_rate
            )

        self._current_node = next_node
        self._visit_counts[next_node] = self._visit_counts.get(next_node, 0) + 1
        self._step_count += 1

        return next_node

    # --- Domain switching ---

    def set_domain(self, graph) -> None:
        """Switch to new graph, preserving learned weights where edges overlap.

        Node consolidation is NOT transferred (node IDs may differ across domains).
        This is consistent with HEB's edge transfer behaviour.
        """
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

        # Reset consolidation for new domain (node IDs are different)
        self._node_consolidation = {n: 0.0 for n in self._graph.nodes()}

        nodes = list(self._graph.nodes())
        if nodes:
            self._current_node = self._rng.choice(nodes)

    # --- Training control ---

    def set_training(self, mode: bool) -> None:
        """Enable/disable learning. When False, edge updates AND consolidation
        accumulation stop. Existing node memory is preserved."""
        self._training = mode

    def reset_engagement_tracking(self) -> None:
        self._visit_counts = {}

    def reset(self) -> None:
        self._weights = dict(self._original_weights)
        self._node_consolidation = {n: 0.0 for n in self._graph.nodes()}
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
    # NOTE: perturb() flattens EDGES only. Node consolidation memory is UNTOUCHED.
    # This is the architectural separation that guarantees recovery.

    def perturb(self, region_id: str, method: str = "flatten") -> TestSystem:
        """Flatten edge weights in target community to global mean.

        CRITICAL: Node consolidation memory is NOT perturbed. This is
        the deep structure that drives recovery after edge flattening.
        """
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        community_nodes = {n for n, c in new._node_to_community.items() if c == community_id}

        all_weights = list(new._weights.values())
        mean_weight = sum(all_weights) / len(all_weights) if all_weights else 1.0

        for (u, v) in list(new._weights.keys()):
            if u in community_nodes or v in community_nodes:
                new._weights[(u, v)] = mean_weight

        # Node consolidation memory is PRESERVED (not flattened)
        return new

    def ablate(self, region_id: str) -> TestSystem:
        """Remove all edges in target community."""
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

        return new

    def boost(self, region_id: str) -> TestSystem:
        """Strengthen edge weights in target community (T1-01f decoy)."""
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
            "node_consolidation": self._node_consolidation,
            "current_node": self._current_node,
            "visit_counts": self._visit_counts,
            "step_count": self._step_count,
            "rng_state": self._rng.bit_generator.state,
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._weights = state["weights"]
        self._node_consolidation = state.get("node_consolidation", {})
        self._current_node = state["current_node"]
        self._visit_counts = state["visit_counts"]
        self._step_count = state["step_count"]
        self._rng.bit_generator.state = state["rng_state"]

    def get_representation_state(self):
        """Node consolidation vector for CKA. Fixed length (n_nodes)
        regardless of edge count — safe after ablation."""
        node_vals = np.array([
            self._node_consolidation.get(n, 0.0)
            for n in sorted(self._graph.nodes())
        ])
        return node_vals.reshape(1, -1)

    # --- Cloning ---

    def get_initial_position(self) -> int:
        return self._initial_position

    def clone(self) -> TestSystem:
        return self._clone_internal()

    def _clone_internal(self) -> AttractorRecoveryWalker:
        new = AttractorRecoveryWalker.__new__(AttractorRecoveryWalker)
        new._original_graph = self._original_graph.copy()
        new._graph = self._graph.copy()
        new._rng = np.random.default_rng(self._seed + self._step_count + 7919)
        new._seed = self._seed
        new._eta = self._eta
        new._decay = self._decay
        new._temperature = self._temperature
        new._alpha = self._alpha
        new._consolidation_rate = self._consolidation_rate
        new._weights = dict(self._weights)
        new._original_weights = dict(self._original_weights)
        new._node_consolidation = dict(self._node_consolidation)
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
                         alpha=self._alpha, consolidation_rate=self._consolidation_rate,
                         initial_position=getattr(self, '_initial_position', None))
        for _ in range(n_steps):
            self.step(None)

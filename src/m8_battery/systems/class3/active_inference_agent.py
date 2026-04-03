"""System 3E: Active inference agent (numpy implementation).

Discrete-state agent that minimises expected free energy on SBM graph.
Learns transition model from experience. Selects actions to balance
epistemic value (information gain) and pragmatic value (preferences).

Dual interpretation (Paper 2):
- Interpretation A: designer specified generative model + free energy.
  Agent minimises designer-chosen quantity → Class 3.
- Interpretation B: dynamics naturally described by free energy minimisation.
  Matrices parameterise dynamics, don't direct them → open question.

Expected battery result: fails conjunction.

Literature:
- Friston K (2010). The free-energy principle: a unified brain theory?
  Nat Rev Neurosci 11:127-138.
- Da Costa L et al. (2020). Active inference on discrete state-spaces.
  J Math Psych 99:102447.

Note: uses numpy implementation rather than pymdp to avoid JAX immutability
issues with TestSystem's mutable state pattern.
"""

from __future__ import annotations

import pickle
import sys
from typing import Any

import networkx as nx
import numpy as np

from m8_battery.core.test_system import TestSystem

def _log(msg: str) -> None:
    print(f"[active_inference] {msg}", file=sys.stderr, flush=True)

class ActiveInferenceAgent(TestSystem):
    """Active inference agent on discrete SBM graph (System 3E).

    Maintains a learned transition model (Dirichlet counts). Selects
    actions to minimise expected free energy = epistemic value +
    pragmatic value. No external reward — pure surprise minimisation.

    Structure metric: Frobenius distance of learned transition model
    from initial uniform prior.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        seed: int = 42,
        alpha: float = 1.0,  # Dirichlet prior concentration
        initial_position: int | None = None,
    ) -> None:
        self._graph = graph
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._alpha = alpha

        self._nodes = sorted(graph.nodes())
        self._n_nodes = len(self._nodes)
        self._node_to_idx = {n: i for i, n in enumerate(self._nodes)}

        # Build adjacency-based action mapping
        self._max_deg = max(dict(graph.degree()).values()) if self._n_nodes > 0 else 1
        self._max_deg = max(self._max_deg, 1)

        # Neighbour lookup: neighbours[s] = list of (action_idx, next_state_idx)
        self._neighbours: dict[int, list[tuple[int, int]]] = {}
        for node in self._nodes:
            s = self._node_to_idx[node]
            if graph.is_directed():
                nbs = sorted(graph.successors(node))
            else:
                nbs = sorted(graph.neighbors(node))
            self._neighbours[s] = [(i, self._node_to_idx[nb]) for i, nb in enumerate(nbs)]

        # Learned transition model: Dirichlet counts pB[s', s, a]
        # Initial: uniform prior (alpha for all transitions)
        self._pB = np.ones((self._n_nodes, self._n_nodes, self._max_deg)) * alpha
        self._initial_pB = self._pB.copy()

        # Navigation state
        if initial_position is not None and initial_position in self._graph:
            self._current_node = initial_position
        else:
            self._current_node = self._nodes[0] if self._nodes else 0
        self._initial_position = self._current_node
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)
        self._step_count = 0
        self._training = True

        # Community mapping
        self._node_to_community: dict[int, int] = {}
        for node in self._nodes:
            data = graph.nodes[node]
            features = data.get("features", {})
            self._node_to_community[node] = features.get("community", data.get("block", 0))

    def _get_transition_probs(self, s: int, a: int) -> np.ndarray:
        """Get normalised transition probabilities P(s'|s,a) from Dirichlet counts."""
        counts = self._pB[:, s, a]
        total = counts.sum()
        if total > 0:
            return counts / total
        return np.ones(self._n_nodes) / self._n_nodes

    def _expected_free_energy(self, s: int) -> np.ndarray:
        """Compute expected free energy for each action from state s.

        G(a) = -epistemic_value(a)
        Epistemic value = expected information gain about transition model.
        Lower G = better action (more informative).
        """
        G = np.zeros(self._max_deg)

        for a in range(self._max_deg):
            p_s_prime = self._get_transition_probs(s, a)

            # Epistemic value: expected KL divergence between posterior and prior
            # Approximated by entropy of transition distribution
            # High entropy = uncertain = high epistemic value
            entropy = -np.sum(p_s_prime * np.log(p_s_prime + 1e-12))
            G[a] = -entropy  # Negative because we minimise G

        return G

    def set_domain(self, graph) -> None:
        """Switch to new graph, resetting topology-dependent state."""
        # Reinitialise on new graph
        self._graph = graph
        self._nodes = sorted(graph.nodes())
        self._n_nodes = len(self._nodes)
        self._node_to_idx = {n: i for i, n in enumerate(self._nodes)}

        self._max_deg = max(dict(graph.degree()).values()) if self._n_nodes > 0 else 1
        self._max_deg = max(self._max_deg, 1)

        # Rebuild neighbour lookup
        self._neighbours = {}
        for node in self._nodes:
            s = self._node_to_idx[node]
            if graph.is_directed():
                nbs = sorted(graph.successors(node))
            else:
                nbs = sorted(graph.neighbors(node))
            self._neighbours[s] = [(i, self._node_to_idx[nb]) for i, nb in enumerate(nbs)]

        # Rebuild community mapping
        self._node_to_community = {}
        for node in self._nodes:
            data = graph.nodes[node]
            features = data.get("features", {})
            self._node_to_community[node] = features.get("community", data.get("block", 0))

        # Always reset pB to uniform prior when switching domains. The
        # transition model learned on graph A is specific to A's topology
        # (which edges exist, which transitions are possible). Even if
        # dimensions happen to match, the learned counts encode A's
        # structure and cannot transfer meaningfully to B (different
        # edges = different transition dynamics). This is by design.
        n = self._n_nodes
        md = self._max_deg
        self._pB = np.ones((n, n, md)) * self._alpha
        self._initial_pB = np.ones((n, n, md)) * self._alpha

        # Reset position
        self._current_node = self._nodes[0] if self._nodes else 0
        self._visit_counts = np.zeros(n, dtype=np.float64)

    def set_training(self, mode: bool) -> None:
        self._training = mode

    def reset_engagement_tracking(self) -> None:
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)

    def reset(self) -> None:
        self._pB = self._initial_pB.copy()
        self._current_node = self._nodes[0] if self._nodes else 0
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)
        self._step_count = 0

    def step(self, input_data: Any) -> Any:
        if input_data is not None:
            if input_data in self._node_to_idx:
                self._current_node = input_data
            elif isinstance(input_data, int) and input_data < self._n_nodes:
                self._current_node = self._nodes[input_data]

        s = self._node_to_idx.get(self._current_node, 0)
        nbs = self._neighbours.get(s, [])

        if not nbs:
            # Dead-end — teleport
            self._current_node = self._rng.choice(self._nodes)
            self._step_count += 1
            return self._current_node

        # Compute expected free energy for available actions
        G = self._expected_free_energy(s)

        # Action selection: softmax over negative G (lower G = better)
        available_actions = [a for a, _ in nbs]
        G_available = G[available_actions]
        logits = -G_available  # Negate: we want to MAXIMISE negative free energy
        logits -= logits.max()
        probs = np.exp(logits)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(len(nbs)) / len(nbs)

        choice = self._rng.choice(len(nbs), p=probs)
        action_idx, next_s = nbs[choice]

        # Update transition model (Dirichlet count)
        if self._training:
            self._pB[next_s, s, action_idx] += 1.0

        next_node = self._nodes[next_s]
        self._visit_counts[next_s] += 1
        self._current_node = next_node
        self._step_count += 1

        return self._current_node

    def get_state(self) -> bytes:
        return pickle.dumps({
            "pB": self._pB.copy(),
            "current_node": self._current_node,
            "visit_counts": self._visit_counts.copy(),
            "step_count": self._step_count,
            "rng_state": self._rng.bit_generator.state,
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._pB = state["pB"]
        self._current_node = state["current_node"]
        self._visit_counts = state["visit_counts"]
        self._step_count = state["step_count"]
        self._rng.bit_generator.state = state["rng_state"]

    def get_structure_metric(self) -> float:
        """Frobenius distance between current and initial pB (Dirichlet counts)."""
        diff = self._pB.flatten() - self._initial_pB.flatten()
        return float(np.sqrt(np.sum(diff ** 2)))

    def get_structure_distribution(self) -> dict[str, float]:
        """Per-community pB distance."""
        communities = sorted(set(self._node_to_community.values()))
        result = {}
        for c in communities:
            c_indices = [self._node_to_idx[n] for n in self._nodes if self._node_to_community.get(n) == c]
            if c_indices:
                diff = self._pB[:, c_indices, :].flatten() - self._initial_pB[:, c_indices, :].flatten()
                result[f"community_{c}"] = float(np.sqrt(np.sum(diff ** 2)))
            else:
                result[f"community_{c}"] = 0.0
        return result

    def get_engagement_distribution(self) -> dict[str, float]:
        """Visit counts per community, normalised."""
        communities = sorted(set(self._node_to_community.values()))
        total = self._visit_counts.sum() or 1.0
        result = {}
        for c in communities:
            c_indices = [self._node_to_idx[n] for n in self._nodes if self._node_to_community.get(n) == c]
            result[f"community_{c}"] = float(self._visit_counts[c_indices].sum() / total)
        return result

    def get_representation_state(self):
        """pB reshaped for CKA. Uses (n_states, n_states*n_actions) layout."""
        return self._pB.reshape(self._n_nodes, -1)

    def ablate(self, region_id: str) -> TestSystem:
        """Zero out transitions involving target community."""
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        c_indices = [new._node_to_idx[n] for n in new._nodes if new._node_to_community.get(n) == community_id]

        for s in c_indices:
            new._pB[:, s, :] = new._alpha
            new._pB[s, :, :] = new._alpha
            new._pB[s, s, :] = new._alpha * 10  # Self-loop bias

        return new

    def perturb(self, region_id: str, method: str = "reset") -> TestSystem:
        """Reset pB for target community to initial values."""
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        c_indices = [new._node_to_idx[n] for n in new._nodes if new._node_to_community.get(n) == community_id]

        for s in c_indices:
            new._pB[:, s, :] = new._initial_pB[:, s, :]

        return new

    def get_regions(self) -> list[str]:
        communities = sorted(set(self._node_to_community.values()))
        return [f"community_{c}" for c in communities]

    def get_initial_position(self) -> int:
        return self._initial_position

    def clone(self) -> TestSystem:
        return self._clone_internal()

    def _clone_internal(self) -> ActiveInferenceAgent:
        new = ActiveInferenceAgent.__new__(ActiveInferenceAgent)
        new._graph = self._graph.copy()
        new._seed = self._seed + self._step_count + 7919
        new._rng = np.random.default_rng(new._seed)
        new._alpha = self._alpha
        new._nodes = list(self._nodes)
        new._n_nodes = self._n_nodes
        new._node_to_idx = dict(self._node_to_idx)
        new._max_deg = self._max_deg
        new._neighbours = {k: list(v) for k, v in self._neighbours.items()}
        new._pB = self._pB.copy()
        new._initial_pB = self._initial_pB.copy()
        new._current_node = self._current_node
        new._initial_position = self._initial_position
        new._visit_counts = self._visit_counts.copy()
        new._step_count = self._step_count
        new._training = self._training
        new._node_to_community = dict(self._node_to_community)
        return new

    def train_on_domain(self, graph: nx.DiGraph, n_steps: int = 500) -> None:
        """Train by exploring the graph under active inference."""
        if graph is not self._graph:
            self.__init__(graph, seed=self._seed, alpha=self._alpha,
                         initial_position=getattr(self, '_initial_position', None))
        _log(f"Training active inference agent: {n_steps} steps, {self._n_nodes} nodes")
        for _ in range(n_steps):
            self.step(None)
        _log(f"  done: pB_distance={self.get_structure_metric():.4f}")

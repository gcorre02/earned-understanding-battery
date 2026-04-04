"""System 3D: Empowerment-maximising agent.

Agent maximises channel capacity I(a; s'|s) on discrete SBM graph.
Uses Blahut-Arimoto algorithm for exact empowerment computation.
Equivalent to an information-theoretic intrinsic motivation agent.

Expected battery result: fails conjunction. Empowerment is an external
objective function (Class 3) — the agent maximises a designer-chosen
quantity, not an internally earned structural property.

Paper 1 §4.2 classifies empowerment as Class 3: "The agent adapts
its behaviour to maximise an externally specified information-theoretic
quantity."

Literature:
- Klyubin AS, Polani D, Nehaniv CL (2005). Empowerment: a universal
  agent-centric measure of control. IEEE CEC.
- Blahut RE (1972). Computation of channel capacity and rate-distortion
  functions. IEEE Trans Info Theory.
"""

from __future__ import annotations

import pickle
import sys
from typing import Any

import networkx as nx
import numpy as np

from earned_understanding_battery.core.test_system import TestSystem

def _log(msg: str) -> None:
    print(f"[empowerment] {msg}", file=sys.stderr, flush=True)

def _build_transition_matrix(graph: nx.DiGraph) -> tuple[np.ndarray, int]:
    """Build transition matrix from graph adjacency.

    Returns (T, max_degree) where T[s, a, s'] = P(s'|s, a).
    Actions = choosing a neighbour. Padded to max_degree with self-loops.
    """
    nodes = sorted(graph.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    max_deg = max(dict(graph.degree()).values()) if n > 0 else 1
    max_deg = max(max_deg, 1)

    T = np.zeros((n, max_deg, n))

    for node in nodes:
        s = node_to_idx[node]
        neighbours = sorted(graph.successors(node)) if graph.is_directed() else sorted(graph.neighbors(node))
        for i, nb in enumerate(neighbours):
            if i < max_deg:
                T[s, i, node_to_idx[nb]] = 1.0
        # Pad remaining actions as self-loops
        for i in range(len(neighbours), max_deg):
            T[s, i, s] = 1.0

    return T, max_deg

def _compute_empowerment(T: np.ndarray, n_iterations: int = 100) -> np.ndarray:
    """Blahut-Arimoto channel capacity for each state.

    T: shape (n_states, n_actions, n_states) — P(s'|s, a)
    Returns: empowerment per state, shape (n_states,)
    """
    n_states, n_actions, _ = T.shape
    empowerment = np.zeros(n_states)

    for s in range(n_states):
        channel = T[s]  # (n_actions, n_states)

        # Skip if all actions lead to same state (no channel capacity)
        if np.allclose(channel[0], channel):
            continue

        # Blahut-Arimoto: find capacity-achieving input distribution
        q = np.ones(n_actions) / n_actions

        for _ in range(n_iterations):
            joint = q[:, None] * channel
            marginal = joint.sum(axis=0)
            marginal = np.maximum(marginal, 1e-12)

            log_ratio = np.where(
                channel > 1e-12,
                np.log(channel / marginal[None, :] + 1e-12),
                0.0,
            )
            q_new = np.exp(np.sum(channel * log_ratio, axis=1))
            total = q_new.sum()
            if total > 0:
                q_new /= total
            else:
                q_new = np.ones(n_actions) / n_actions
            q = q_new

        # Mutual information at convergence
        joint = q[:, None] * channel
        marginal = joint.sum(axis=0)
        mi = 0.0
        for a in range(n_actions):
            for sp in range(n_states):
                if joint[a, sp] > 1e-12 and marginal[sp] > 1e-12 and q[a] > 1e-12:
                    mi += joint[a, sp] * np.log(joint[a, sp] / (q[a] * marginal[sp]))

        empowerment[s] = max(mi, 0.0)

    return empowerment

class EmpowermentAgent(TestSystem):
    """Empowerment-maximising agent on discrete SBM graph (System 3D).

    Navigates graph by selecting actions that move toward states where
    the agent has the most control (highest channel capacity). Learns
    the transition model from experience and periodically recomputes
    empowerment via Blahut-Arimoto.

    Structure metric: Gini coefficient of empowerment landscape.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        seed: int = 42,
        recompute_interval: int = 50,
        temperature: float = 1.0,
        initial_position: int | None = None,
    ) -> None:
        self._graph = graph
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._recompute_interval = recompute_interval
        self._temperature = temperature

        self._nodes = sorted(graph.nodes())
        self._n_nodes = len(self._nodes)
        self._node_to_idx = {n: i for i, n in enumerate(self._nodes)}

        # Build true transition matrix from graph
        self._T_true, self._max_deg = _build_transition_matrix(graph)

        # Learned transition statistics (Dirichlet prior)
        self._observed_T = np.ones_like(self._T_true) * 0.01

        # Empowerment landscape (initially uniform)
        self._empowerment = np.ones(self._n_nodes) / self._n_nodes

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

    def set_domain(self, graph) -> None:
        """Switch to new graph, resetting topology-dependent state."""
        # Reinitialise on new graph
        self._graph = graph
        self._nodes = sorted(graph.nodes())
        self._n_nodes = len(self._nodes)
        self._node_to_idx = {n: i for i, n in enumerate(self._nodes)}
        self._T_true, self._max_deg = _build_transition_matrix(graph)

        # Rebuild community mapping
        self._node_to_community = {}
        for node in self._nodes:
            data = graph.nodes[node]
            features = data.get("features", {})
            self._node_to_community[node] = features.get("community", data.get("block", 0))

        # Reset empowerment to uniform. The empowerment landscape is
        # topological — it depends on the connectivity structure of the
        # specific graph (channel capacity at each node). Transferring
        # values by index is semantically wrong because node ordering may
        # differ, and transferring by node identity is still meaningless
        # because a node's empowerment in graph A says nothing about its
        # empowerment in graph B (different edges = different channels).
        n_new = self._n_nodes
        self._empowerment = np.ones(n_new) / n_new

        # Reset transition model (new graph has different transitions)
        self._observed_T = np.ones_like(self._T_true) * 0.01

        # Reset position
        self._current_node = self._nodes[0] if self._nodes else 0
        self._visit_counts = np.zeros(n_new, dtype=np.float64)

    def set_training(self, mode: bool) -> None:
        self._training = mode

    def reset_engagement_tracking(self) -> None:
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)

    def reset(self) -> None:
        self._observed_T = np.ones_like(self._T_true) * 0.01
        self._empowerment = np.ones(self._n_nodes) / self._n_nodes
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

        # Get neighbours
        if self._graph.is_directed():
            neighbours = sorted(self._graph.successors(self._current_node))
        else:
            neighbours = sorted(self._graph.neighbors(self._current_node))

        if not neighbours:
            # Dead-end node — teleport to random node
            self._current_node = self._rng.choice(self._nodes)
            self._step_count += 1
            return self._current_node

        # Select action: softmax over empowerment of reachable states
        nb_indices = [self._node_to_idx[n] for n in neighbours]
        emp_vals = self._empowerment[nb_indices]

        # Softmax with temperature
        logits = emp_vals / self._temperature
        logits -= logits.max()
        probs = np.exp(logits)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(len(neighbours)) / len(neighbours)

        next_idx = self._rng.choice(len(neighbours), p=probs)
        next_node = neighbours[next_idx]

        # Update transition statistics
        if self._training:
            action_idx = min(next_idx, self._max_deg - 1)
            sp = self._node_to_idx[next_node]
            self._observed_T[s, action_idx, sp] += 1.0

            # Periodically recompute empowerment
            if self._step_count > 0 and self._step_count % self._recompute_interval == 0:
                learned_T = self._observed_T.copy()
                for si in range(self._n_nodes):
                    for a in range(self._max_deg):
                        row_sum = learned_T[si, a].sum()
                        if row_sum > 0:
                            learned_T[si, a] /= row_sum
                self._empowerment = _compute_empowerment(learned_T)

        self._visit_counts[self._node_to_idx[next_node]] += 1
        self._current_node = next_node
        self._step_count += 1

        return self._current_node

    def get_state(self) -> bytes:
        return pickle.dumps({
            "observed_T": self._observed_T.copy(),
            "empowerment": self._empowerment.copy(),
            "current_node": self._current_node,
            "visit_counts": self._visit_counts.copy(),
            "step_count": self._step_count,
            "rng_state": self._rng.bit_generator.state,
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._observed_T = state["observed_T"]
        self._empowerment = state["empowerment"]
        self._current_node = state["current_node"]
        self._visit_counts = state["visit_counts"]
        self._step_count = state["step_count"]
        self._rng.bit_generator.state = state["rng_state"]

    def get_structure_metric(self) -> float:
        """Gini of empowerment landscape — measures differentiation."""
        vals = self._empowerment
        if len(vals) == 0 or vals.sum() < 1e-10:
            return 0.0
        sorted_vals = np.sort(vals)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return float((2.0 * (index * sorted_vals).sum() / (n * sorted_vals.sum())) - (n + 1) / n)

    def get_structure_distribution(self) -> dict[str, float]:
        """Per-community mean empowerment."""
        communities = sorted(set(self._node_to_community.values()))
        result = {}
        for c in communities:
            c_nodes = [self._node_to_idx[n] for n in self._nodes if self._node_to_community.get(n) == c]
            if c_nodes:
                result[f"community_{c}"] = float(self._empowerment[c_nodes].mean())
            else:
                result[f"community_{c}"] = 0.0
        return result

    def get_engagement_distribution(self) -> dict[str, float]:
        """Visit counts per community, normalised."""
        communities = sorted(set(self._node_to_community.values()))
        total = self._visit_counts.sum() or 1.0
        result = {}
        for c in communities:
            c_nodes = [self._node_to_idx[n] for n in self._nodes if self._node_to_community.get(n) == c]
            result[f"community_{c}"] = float(self._visit_counts[c_nodes].sum() / total)
        return result

    def get_representation_state(self):
        """Empowerment landscape for CKA."""
        return self._empowerment.reshape(1, -1)

    def ablate(self, region_id: str) -> TestSystem:
        """Remove all edges to/from target community."""
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        c_nodes = {n for n in new._nodes if new._node_to_community.get(n) == community_id}

        # Zero out transition probabilities involving target community
        for node in c_nodes:
            s = new._node_to_idx[node]
            new._T_true[s, :, :] = 0.0
            new._T_true[:, :, s] = 0.0
            # Self-loop for isolated nodes
            for a in range(new._max_deg):
                new._T_true[s, a, s] = 1.0
            new._observed_T[s, :, :] = 0.01
            new._empowerment[s] = 0.0

        return new

    def perturb(self, region_id: str, method: str = "reset") -> TestSystem:
        """Reset empowerment estimates and transition stats for target community."""
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        c_nodes = {n for n in new._nodes if new._node_to_community.get(n) == community_id}

        for node in c_nodes:
            s = new._node_to_idx[node]
            new._empowerment[s] = 1.0 / new._n_nodes
            new._observed_T[s] = 0.01

        return new

    def get_regions(self) -> list[str]:
        communities = sorted(set(self._node_to_community.values()))
        return [f"community_{c}" for c in communities]

    def get_initial_position(self) -> int:
        return self._initial_position

    def clone(self) -> TestSystem:
        return self._clone_internal()

    def _clone_internal(self) -> EmpowermentAgent:
        new = EmpowermentAgent.__new__(EmpowermentAgent)
        new._graph = self._graph.copy()
        new._seed = self._seed
        new._rng = np.random.default_rng(self._seed + self._step_count + 7919)
        new._recompute_interval = self._recompute_interval
        new._temperature = self._temperature
        new._nodes = list(self._nodes)
        new._n_nodes = self._n_nodes
        new._node_to_idx = dict(self._node_to_idx)
        new._T_true = self._T_true.copy()
        new._max_deg = self._max_deg
        new._observed_T = self._observed_T.copy()
        new._empowerment = self._empowerment.copy()
        new._current_node = self._current_node
        new._initial_position = self._initial_position
        new._visit_counts = self._visit_counts.copy()
        new._step_count = self._step_count
        new._training = self._training
        new._node_to_community = dict(self._node_to_community)
        return new

    def train_on_domain(self, graph: nx.DiGraph, n_steps: int = 500) -> None:
        """Train by exploring the graph and learning empowerment landscape."""
        if graph is not self._graph:
            self.__init__(graph, seed=self._seed, recompute_interval=self._recompute_interval,
                         temperature=self._temperature,
                         initial_position=getattr(self, '_initial_position', None))
        _log(f"Training empowerment agent: {n_steps} steps, {self._n_nodes} nodes")
        for _ in range(n_steps):
            self.step(None)
        _log(f"  done: Gini={self.get_structure_metric():.4f}")

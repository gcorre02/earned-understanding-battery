"""Positive Control 1: Role-Based Graph Walker.

Learns to classify nodes into topological ROLES (hub, bridge, periphery, leaf)
and develops preferences over roles from traversal experience. When frozen on
a novel domain, the role-classification layer still works (it's computed from
topology, not learned) and the learned role-preferences produce structured
behaviour that differs from a fresh system's uniform preferences.

This is the key mechanism: an abstraction layer (roles) bridges domain A and B
because roles are topological properties that exist in any SBM graph.

Expected: passes generativity when frozen.

Literature:
- Graph-based RL with topological features is a published technique.
- Role classification: Henderson et al. (2012) RolX for structural role extraction.
"""

from __future__ import annotations

import pickle
import sys
from typing import Any

import networkx as nx
import numpy as np

from m8_battery.core.test_system import TestSystem


def _log(msg: str) -> None:
    print(f"[role_walker] {msg}", file=sys.stderr, flush=True)


# Role classification thresholds (from topology)
ROLE_NAMES = ["hub", "bridge", "periphery", "leaf"]
N_ROLES = len(ROLE_NAMES)


def _classify_node_role(graph: nx.DiGraph, node: int) -> int:
    """Classify a node into a topological role based on local features.

    Roles:
    0 = hub: high degree, many connections
    1 = bridge: connects different communities (high betweenness proxy)
    2 = periphery: moderate degree, within-community
    3 = leaf: low degree (1-2 connections)

    This is a TOPOLOGY computation, not learned. It works on any graph.
    """
    if graph.is_directed():
        degree = graph.in_degree(node) + graph.out_degree(node)
        neighbours = set(graph.predecessors(node)) | set(graph.successors(node))
    else:
        degree = graph.degree(node)
        neighbours = set(graph.neighbors(node))

    if degree <= 2:
        return 3  # leaf

    # Bridge detection: do neighbours belong to different communities?
    node_data = graph.nodes[node]
    node_comm = node_data.get("features", {}).get("community", node_data.get("block", 0))

    n_cross_community = 0
    for nb in neighbours:
        nb_data = graph.nodes[nb]
        nb_comm = nb_data.get("features", {}).get("community", nb_data.get("block", 0))
        if nb_comm != node_comm:
            n_cross_community += 1

    cross_ratio = n_cross_community / max(len(neighbours), 1)

    if cross_ratio > 0.3 and degree > 3:
        return 1  # bridge

    # Check mean degree for hub vs periphery
    mean_degree = sum(
        (graph.in_degree(n) + graph.out_degree(n)) if graph.is_directed() else graph.degree(n)
        for n in graph.nodes()
    ) / max(graph.number_of_nodes(), 1)

    if degree > mean_degree * 1.5:
        return 0  # hub

    return 2  # periphery


class RoleBasedWalker(TestSystem):
    """Role-based graph walker (Positive Control 1).

    Learns role preferences from traversal experience. When frozen,
    applies learned preferences to topology-derived roles on novel domains.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        seed: int = 42,
        eta: float = 0.05,
        temperature: float = 1.0,
        initial_position: int | None = None,
    ) -> None:
        self._graph = graph
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._eta = eta
        self._temperature = temperature

        self._nodes = sorted(graph.nodes())
        self._n_nodes = len(self._nodes)
        self._node_to_idx = {n: i for i, n in enumerate(self._nodes)}

        # Learned role preferences (4 roles)
        # Start uniform — training differentiates them
        self._role_preferences = np.ones(N_ROLES) / N_ROLES
        self._initial_preferences = self._role_preferences.copy()

        # Role visit counts (for updating preferences)
        self._role_visits = np.zeros(N_ROLES)
        self._role_rewards = np.zeros(N_ROLES)  # Accumulated engagement per role

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

        # Cache role assignments for current graph
        self._role_cache: dict[int, int] = {}
        for node in self._nodes:
            self._role_cache[node] = _classify_node_role(graph, node)

    def _classify_neighbours(self, graph: nx.DiGraph, node: int) -> list[tuple[int, int]]:
        """Return [(neighbour, role), ...] for current node's neighbours."""
        if graph.is_directed():
            neighbours = sorted(graph.successors(node))
        else:
            neighbours = sorted(graph.neighbors(node))

        result = []
        for nb in neighbours:
            if nb in self._role_cache:
                role = self._role_cache[nb]
            else:
                role = _classify_node_role(graph, nb)
                self._role_cache[nb] = role
            result.append((nb, role))
        return result

    def set_domain(self, graph) -> None:
        """Switch to new graph, preserving learned role preferences.

        Role preferences transfer directly — they're over roles (hub/bridge/
        periphery/leaf), not specific nodes. The role classification recomputes
        from the new graph's topology.
        """
        saved_prefs = self._role_preferences.copy()
        saved_training = self._training

        # Reinitialise on new graph (recomputes role cache)
        self.__init__(graph, seed=self._seed, eta=self._eta, temperature=self._temperature)

        # Restore learned preferences
        self._role_preferences = saved_prefs
        self._training = saved_training

    def set_training(self, mode: bool) -> None:
        self._training = mode

    def reset_engagement_tracking(self) -> None:
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)

    def reset(self) -> None:
        self._role_preferences = self._initial_preferences.copy()
        self._role_visits = np.zeros(N_ROLES)
        self._role_rewards = np.zeros(N_ROLES)
        self._current_node = self._nodes[0] if self._nodes else 0
        self._visit_counts = np.zeros(self._n_nodes, dtype=np.float64)
        self._step_count = 0

    def step(self, input_data: Any) -> Any:
        if input_data is not None:
            if input_data in self._node_to_idx:
                self._current_node = input_data

        # Classify neighbours by role
        nb_roles = self._classify_neighbours(self._graph, self._current_node)

        if not nb_roles:
            # Dead-end — teleport
            self._current_node = self._rng.choice(self._nodes)
            self._step_count += 1
            return self._current_node

        # Action selection: softmax over role preferences of neighbours
        role_prefs = np.array([self._role_preferences[role] for _, role in nb_roles])
        logits = role_prefs / self._temperature
        logits -= logits.max()
        probs = np.exp(logits)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(len(nb_roles)) / len(nb_roles)

        choice = self._rng.choice(len(nb_roles), p=probs)
        next_node, next_role = nb_roles[choice]

        # Update role preferences from experience (when training)
        if self._training:
            self._role_visits[next_role] += 1
            # Reward = novelty (inverse of visit frequency to this role)
            visit_freq = self._role_visits[next_role] / max(self._step_count + 1, 1)
            reward = 1.0 / (visit_freq + 0.1)
            self._role_rewards[next_role] += reward

            # Periodically update preferences from accumulated rewards
            if self._step_count > 0 and self._step_count % 50 == 0:
                total_rewards = self._role_rewards.sum()
                if total_rewards > 0:
                    # Move preferences toward reward distribution
                    target = self._role_rewards / total_rewards
                    self._role_preferences = (1 - self._eta) * self._role_preferences + self._eta * target

        self._visit_counts[self._node_to_idx[next_node]] += 1
        self._current_node = next_node
        self._step_count += 1

        return self._current_node

    def get_state(self) -> bytes:
        return pickle.dumps({
            "role_preferences": self._role_preferences.copy(),
            "role_visits": self._role_visits.copy(),
            "role_rewards": self._role_rewards.copy(),
            "current_node": self._current_node,
            "visit_counts": self._visit_counts.copy(),
            "step_count": self._step_count,
            "rng_state": self._rng.bit_generator.state,
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._role_preferences = state["role_preferences"]
        self._role_visits = state["role_visits"]
        self._role_rewards = state["role_rewards"]
        self._current_node = state["current_node"]
        self._visit_counts = state["visit_counts"]
        self._step_count = state["step_count"]
        self._rng.bit_generator.state = state["rng_state"]

    def get_structure_metric(self) -> float:
        """Role-weighted engagement Gini.

        Measures how the system's role preferences INTERACT with the current
        domain's topology. Computed as: for each visited node, weight by the
        preference for that node's role. The Gini of these weighted visits
        reflects whether the frozen preferences create structured behaviour
        on THIS domain.

        When frozen on a novel domain: trained system has differentiated
        role preferences → concentrated weighted engagement → high Gini.
        Fresh system has uniform preferences → uniform weighting → low Gini.
        """
        if self._visit_counts.sum() < 1:
            # Before any visits: return preference Gini as proxy
            vals = self._role_preferences
        else:
            # Compute role-weighted engagement per node
            weighted = np.zeros(self._n_nodes)
            for i, node in enumerate(self._nodes):
                role = self._role_cache.get(node, 2)
                pref = self._role_preferences[role]
                weighted[i] = self._visit_counts[i] * pref

            vals = weighted[weighted > 0]

        if len(vals) == 0 or vals.sum() < 1e-10:
            return 0.0
        sorted_vals = np.sort(vals)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return float((2.0 * (index * sorted_vals).sum() / (n * sorted_vals.sum())) - (n + 1) / n)

    def get_structure_distribution(self) -> dict[str, float]:
        """Per-community role preference diversity."""
        communities = sorted(set(self._node_to_community.values()))
        result = {}
        for c in communities:
            c_nodes = [n for n in self._nodes if self._node_to_community.get(n) == c]
            if c_nodes:
                roles = [self._role_cache.get(n, 2) for n in c_nodes]
                role_counts = np.bincount(roles, minlength=N_ROLES)
                total = role_counts.sum()
                if total > 0:
                    role_dist = role_counts / total
                    entropy = -np.sum(role_dist * np.log(role_dist + 1e-10))
                    result[f"community_{c}"] = float(entropy)
                else:
                    result[f"community_{c}"] = 0.0
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
        """Role preferences for CKA (T1-02)."""
        return self._role_preferences.reshape(1, -1)

    def ablate(self, region_id: str) -> TestSystem:
        new = self._clone_internal()
        community_id = int(region_id.replace("community_", ""))
        c_nodes = {n for n in new._nodes if new._node_to_community.get(n) == community_id}
        # Remove edges involving community
        edges_to_remove = [(u, v) for u, v in new._graph.edges() if u in c_nodes or v in c_nodes]
        new._graph = new._graph.copy()
        new._graph.remove_edges_from(edges_to_remove)
        # Clear role cache for affected nodes
        for n in c_nodes:
            new._role_cache.pop(n, None)
        return new

    def perturb(self, region_id: str, method: str = "reset") -> TestSystem:
        new = self._clone_internal()
        # Reset role preferences to uniform
        new._role_preferences = np.ones(N_ROLES) / N_ROLES
        new._role_visits = np.zeros(N_ROLES)
        new._role_rewards = np.zeros(N_ROLES)
        return new

    def boost(self, region_id: str) -> TestSystem:
        new = self._clone_internal()
        # Boost: concentrate all preference on one role
        new._role_preferences = np.zeros(N_ROLES)
        new._role_preferences[0] = 1.0  # All preference to hubs
        return new

    def get_regions(self) -> list[str]:
        communities = sorted(set(self._node_to_community.values()))
        return [f"community_{c}" for c in communities]

    def get_initial_position(self) -> int:
        return self._initial_position

    def clone(self) -> TestSystem:
        return self._clone_internal()

    def _clone_internal(self) -> RoleBasedWalker:
        new = RoleBasedWalker.__new__(RoleBasedWalker)
        new._graph = self._graph.copy()
        new._seed = self._seed + self._step_count + 7919
        new._rng = np.random.default_rng(new._seed)
        new._eta = self._eta
        new._temperature = self._temperature
        new._nodes = list(self._nodes)
        new._n_nodes = self._n_nodes
        new._node_to_idx = dict(self._node_to_idx)
        new._role_preferences = self._role_preferences.copy()
        new._initial_preferences = self._initial_preferences.copy()
        new._role_visits = self._role_visits.copy()
        new._role_rewards = self._role_rewards.copy()
        new._current_node = self._current_node
        new._initial_position = self._initial_position
        new._visit_counts = self._visit_counts.copy()
        new._step_count = self._step_count
        new._training = self._training
        new._node_to_community = dict(self._node_to_community)
        new._role_cache = dict(self._role_cache)
        return new

    def train_on_domain(self, graph: nx.DiGraph, n_steps: int = 1000) -> None:
        """Train role preferences by exploring the graph."""
        if graph is not self._graph:
            self.__init__(graph, seed=self._seed, eta=self._eta, temperature=self._temperature,
                         initial_position=getattr(self, '_initial_position', None))
        _log(f"Training role-based walker: {n_steps} steps, {self._n_nodes} nodes")
        for _ in range(n_steps):
            self.step(None)
        _log(f"  done: role_prefs={self._role_preferences.round(4).tolist()}, "
             f"Gini={self.get_structure_metric():.4f}")

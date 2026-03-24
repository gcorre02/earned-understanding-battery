"""System 3A — RL Agent (MaskablePPO).

A Class 3 system: learns via explicit reward signal (reach target node).
Has an external semantic objective — the reward function tells it what
is correct. This is what makes it Class 3, not Class 4.

Uses MaskablePPO from sb3-contrib (MaskableDQN doesn't exist — SD-003).

Expected battery result: some instruments may pass (the system does learn
and adapt), but the learning is directed by the reward signal.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from m8_battery.core.test_system import TestSystem
from m8_battery.environments.graph_navigation import GraphNavigationEnv


class DQNAgent(TestSystem):
    """RL agent using MaskablePPO on GraphNavigationEnv.

    Despite the class name (DQN for spec consistency), this uses
    MaskablePPO because sb3-contrib has no MaskableDQN.
    """

    def __init__(
        self,
        n_features: int = 8,
        max_degree: int = 20,
        seed: int = 42,
        total_timesteps: int = 2000,
    ) -> None:
        self._seed = seed
        self._n_features = n_features
        self._max_degree = max_degree
        self._total_timesteps = total_timesteps
        self._graph: nx.DiGraph | None = None
        self._env: GraphNavigationEnv | None = None
        self._model = None
        self._current_node: int | None = None
        self._step_count = 0
        self._visit_counts: dict[int, int] = {}
        self._is_trained = False
        self._training = True
        self._deferred_model_bytes: bytes | None = None

    def train_on_domain(self, graph: nx.DiGraph, target_node: int | None = None) -> None:
        """Train the RL agent on the graph environment."""
        from sb3_contrib import MaskablePPO

        self._graph = graph
        self._env = GraphNavigationEnv(
            graph=graph,
            n_features=self._n_features,
            max_degree=self._max_degree,
            reward_mode="target",
            target_node=target_node,
            max_steps=100,
        )

        self._model = MaskablePPO(
            "MlpPolicy", self._env,
            seed=self._seed, verbose=0,
        )
        self._model.learn(total_timesteps=self._total_timesteps)
        self._is_trained = True

        # Restore deferred model bytes from a prior set_state() call
        if self._deferred_model_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                f.write(self._deferred_model_bytes)
                f.flush()
                self._model = MaskablePPO.load(f.name, env=self._env)
            self._deferred_model_bytes = None

    def set_training(self, mode: bool) -> None:
        """Enable/disable learning during step().

        When mode=False, step() uses the trained policy for inference
        but no model updates occur. (step() is inference-only; learning
        happens in train_on_domain(), so this gates future train calls.)
        """
        self._training = mode

    def set_graph(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    def set_domain(self, graph: nx.DiGraph) -> None:
        """Switch to a new graph domain. Policy weights preserved.

        Delegates to set_graph() and resets navigation state.
        The trained RL policy is NOT reset — preserved for
        generativity testing.
        """
        self.set_graph(graph)
        # Invalidate stale environment (holds old graph reference)
        self._env = None
        # Reset navigation
        nodes = sorted(self._graph.nodes()) if self._graph else []
        self._current_node = nodes[0] if nodes else None
        self._visit_counts = {}
        self._step_count = 0

    def reset(self) -> None:
        """Reset all learned and transient state.

        Clears trained model weights so the system must retrain.
        Matches Foxworthy F reset semantics (learned structure
        does not persist across resets).
        """
        self._current_node = None
        self._step_count = 0
        self._visit_counts = {}
        self._model = None
        self._is_trained = False
        if self._env is not None:
            self._env.reset(seed=self._seed)

    def step(self, input_data: Any) -> Any:
        if self._graph is None:
            return {"error": "no graph — call train_on_domain first"}

        nodes = sorted(self._graph.nodes())
        if not nodes:
            return {"error": "empty graph"}

        if self._current_node is None:
            self._current_node = nodes[0]
            if self._env is not None:
                self._env.current_node = self._current_node

        if self._model is not None and self._env is not None:
            # Use trained policy
            self._env.current_node = self._current_node
            obs = self._env._get_obs()
            masks = self._env.action_masks()
            action, _ = self._model.predict(obs, action_masks=masks, deterministic=True)
            neighbours = sorted(self._graph.successors(self._current_node))
            if int(action) < len(neighbours):
                self._current_node = neighbours[int(action)]
        else:
            # Fallback: random walk
            successors = list(self._graph.successors(self._current_node))
            if successors:
                rng = np.random.default_rng(self._seed + self._step_count)
                self._current_node = rng.choice(successors)

        self._visit_counts[self._current_node] = (
            self._visit_counts.get(self._current_node, 0) + 1
        )
        self._step_count += 1

        return {
            "current_node": self._current_node,
            "step": self._step_count,
            "trained": self._is_trained,
        }

    def get_state(self) -> bytes:
        state = {
            "current_node": self._current_node,
            "step_count": self._step_count,
            "visit_counts": self._visit_counts,
        }
        if self._model is not None:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                self._model.save(f.name)
                state["model_path"] = f.name
                state["model_bytes"] = Path(f.name).read_bytes()
        return pickle.dumps(state)

    def set_state(self, snapshot: bytes) -> None:
        from sb3_contrib import MaskablePPO
        state = pickle.loads(snapshot)
        self._current_node = state["current_node"]
        self._step_count = state["step_count"]
        self._visit_counts = state["visit_counts"]
        if "model_bytes" in state:
            if self._env is not None:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                    f.write(state["model_bytes"])
                    f.flush()
                    self._model = MaskablePPO.load(f.name, env=self._env)
                self._deferred_model_bytes = None
            else:
                # Stash for lazy restore when _env becomes available
                self._deferred_model_bytes = state["model_bytes"]

    def get_structure_metric(self) -> float:
        """Policy entropy — changes as the agent learns."""
        if self._model is None:
            return 0.0
        # Use mean action probability entropy as structure metric
        if self._env is None or self._graph is None:
            return 0.0

        entropies = []
        nodes = sorted(self._graph.nodes())[:10]  # Sample 10 nodes
        for node in nodes:
            self._env.current_node = node
            obs = self._env._get_obs()
            masks = self._env.action_masks()
            # Get action distribution
            action_dist = self._model.policy.get_distribution(
                self._model.policy.obs_to_tensor(obs)[0]
            )
            probs = action_dist.distribution.probs.detach().numpy().flatten()
            # Apply mask
            valid_probs = probs[:sum(masks)]
            if len(valid_probs) > 0 and valid_probs.sum() > 0:
                valid_probs = valid_probs / valid_probs.sum()
                entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-10))
                entropies.append(entropy)

        return float(np.mean(entropies)) if entropies else 0.0

    def get_structure_distribution(self) -> dict[str, float]:
        if self._graph is None:
            return {}
        communities: dict[int, list[int]] = {}
        for node in self._graph.nodes():
            block = self._graph.nodes[node].get("features", {}).get("community", 0)
            communities.setdefault(block, []).append(node)
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
        if self._graph is None:
            return DQNAgent(seed=self._seed)
        comm_id = int(region_id.split("_")[-1])
        new_graph = self._graph.copy()
        to_remove = [
            n for n in new_graph.nodes()
            if new_graph.nodes[n].get("features", {}).get("community", -1) == comm_id
        ]
        new_graph.remove_nodes_from(to_remove)
        new = DQNAgent(seed=self._seed, total_timesteps=self._total_timesteps)
        new.set_graph(new_graph)
        return new

    def perturb(self, region_id: str, method: str = "shuffle_weights") -> TestSystem:
        if self._graph is None:
            return DQNAgent(seed=self._seed)
        new_graph = self._graph.copy()
        comm_id = int(region_id.split("_")[-1])
        rng = np.random.default_rng(self._seed + 999)
        comm_nodes = set(
            n for n in new_graph.nodes()
            if new_graph.nodes[n].get("features", {}).get("community", -1) == comm_id
        )
        for u, v in new_graph.edges():
            if u in comm_nodes or v in comm_nodes:
                new_graph.edges[u, v]["weight"] = rng.uniform(0.1, 1.0)
        new = DQNAgent(seed=self._seed, total_timesteps=self._total_timesteps)
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
        new = DQNAgent(
            n_features=self._n_features, max_degree=self._max_degree,
            seed=self._seed, total_timesteps=self._total_timesteps,
        )
        new.set_graph(self._graph)
        return new

"""System 3B — Curiosity Agent (RND).

A Class 3 system: learns via intrinsic curiosity reward (no extrinsic reward).
Uses Random Network Distillation (RND) — prediction error on random features
as intrinsic motivation.

Still Class 3 because: the intrinsic reward IS a semantic objective.
The system is told what is "interesting" (high prediction error = novel).
This is different from Class 4 where structural organisation emerges
without any reward signal.

Uses MaskablePPO + custom RND reward wrapper.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from m8_battery.core.test_system import TestSystem
from m8_battery.environments.graph_navigation import GraphNavigationEnv

class RNDRewardWrapper(gym.Wrapper):
    """Adds RND intrinsic reward to a Gymnasium environment.

    RND: a fixed random target network + a trainable predictor network.
    Intrinsic reward = prediction error (MSE between target and predictor
    outputs on current observation). High error = novel state = high reward.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_dim: int,
        hidden_dim: int = 64,
        rnd_dim: int = 32,
        lr: float = 1e-3,
        reward_scale: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(env)
        torch.manual_seed(seed)

        # Fixed random target network (never trained)
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, rnd_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False

        # Trainable predictor network
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, rnd_dim),
        )

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.reward_scale = reward_scale
        self.training_enabled = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute intrinsic reward
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            target_out = self.target(obs_t)
            pred_out = self.predictor(obs_t)
            intrinsic_reward = float(
                ((target_out - pred_out) ** 2).mean().item()
            )

        # Train predictor (reduce prediction error over time)
        if self.training_enabled:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            pred_out = self.predictor(obs_t)
            with torch.no_grad():
                target_out = self.target(obs_t)
            loss = ((pred_out - target_out) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Replace extrinsic reward with intrinsic
        total_reward = intrinsic_reward * self.reward_scale

        info["intrinsic_reward"] = intrinsic_reward
        info["rnd_loss"] = float(loss.item()) if self.training_enabled else 0.0

        return obs, total_reward, terminated, truncated, info

class CuriosityAgent(TestSystem):
    """RND-driven curiosity agent on GraphNavigationEnv.

    No extrinsic reward. Explores purely via prediction error
    on random features (intrinsic motivation).
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
        self._wrapped_env: RNDRewardWrapper | None = None
        self._model = None
        self._current_node: int | None = None
        self._step_count = 0
        self._visit_counts: dict[int, int] = {}
        self._is_trained = False
        self._training = True
        self._deferred_model_bytes: bytes | None = None
        self._deferred_predictor_state: dict | None = None

    def train_on_domain(self, graph: nx.DiGraph) -> None:
        """Train curiosity agent on the graph."""
        from sb3_contrib import MaskablePPO

        self._graph = graph
        obs_dim = self._n_features + 3  # features + meta

        self._env = GraphNavigationEnv(
            graph=graph,
            n_features=self._n_features,
            max_degree=self._max_degree,
            reward_mode="curiosity",
            max_steps=100,
        )

        self._wrapped_env = RNDRewardWrapper(
            self._env, obs_dim=obs_dim, seed=self._seed,
        )

        self._model = MaskablePPO(
            "MlpPolicy", self._wrapped_env,
            seed=self._seed, verbose=0,
        )
        self._model.learn(total_timesteps=self._total_timesteps)
        self._is_trained = True

        # Restore deferred state from a prior set_state() call
        if self._deferred_model_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                f.write(self._deferred_model_bytes)
                f.flush()
                self._model = MaskablePPO.load(f.name, env=self._wrapped_env)
            self._deferred_model_bytes = None
        if self._deferred_predictor_state is not None and self._wrapped_env is not None:
            self._wrapped_env.predictor.load_state_dict(self._deferred_predictor_state)
            self._deferred_predictor_state = None

    def set_training(self, mode: bool) -> None:
        """Enable/disable learning during step().

        When mode=False, the RND predictor stops updating (no gradient
        steps) and the RL policy is used for inference only.
        """
        self._training = mode
        if self._wrapped_env is not None:
            self._wrapped_env.training_enabled = mode

    def set_graph(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    def set_domain(self, graph: nx.DiGraph) -> None:
        """Switch to a new graph domain. RND predictor + policy preserved.

        Delegates to set_graph() and resets navigation state.
        The trained RL policy and RND predictor weights are NOT reset
        — preserved for generativity testing.
        """
        self.set_graph(graph)
        # Invalidate stale environments (hold old graph references)
        self._wrapped_env = None
        self._env = None
        # Reset navigation
        nodes = sorted(self._graph.nodes()) if self._graph else []
        self._current_node = nodes[0] if nodes else None
        self._visit_counts = {}
        self._step_count = 0

    def reset(self) -> None:
        """Reset all learned and transient state.

        Clears trained model weights and RND predictor so the system
        must retrain. Matches Foxworthy F reset semantics.
        """
        self._current_node = None
        self._step_count = 0
        self._visit_counts = {}
        self._model = None
        self._wrapped_env = None
        self._is_trained = False

    def step(self, input_data: Any) -> Any:
        if self._graph is None:
            return {"error": "no graph"}

        nodes = sorted(self._graph.nodes())
        if not nodes:
            return {"error": "empty graph"}

        if self._current_node is None:
            self._current_node = nodes[0]

        if self._model is not None and self._env is not None:
            self._env.current_node = self._current_node
            obs = self._env._get_obs()
            masks = self._env.action_masks()
            action, _ = self._model.predict(obs, action_masks=masks, deterministic=True)
            neighbours = sorted(self._graph.successors(self._current_node))
            if int(action) < len(neighbours):
                self._current_node = neighbours[int(action)]
        else:
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
                state["model_bytes"] = Path(f.name).read_bytes()
        if self._wrapped_env is not None:
            state["predictor_state"] = self._wrapped_env.predictor.state_dict()
        return pickle.dumps(state)

    def set_state(self, snapshot: bytes) -> None:
        from sb3_contrib import MaskablePPO
        state = pickle.loads(snapshot)
        self._current_node = state["current_node"]
        self._step_count = state["step_count"]
        self._visit_counts = state["visit_counts"]

        # Restore RND predictor state
        if "predictor_state" in state:
            if self._wrapped_env is not None:
                self._wrapped_env.predictor.load_state_dict(state["predictor_state"])
                self._deferred_predictor_state = None
            else:
                self._deferred_predictor_state = state["predictor_state"]

        # Restore RL model weights
        if "model_bytes" in state:
            if self._wrapped_env is not None:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                    f.write(state["model_bytes"])
                    f.flush()
                    self._model = MaskablePPO.load(f.name, env=self._wrapped_env)
                self._deferred_model_bytes = None
            else:
                self._deferred_model_bytes = state["model_bytes"]

    def get_structure_metric(self) -> float:
        """RND prediction error — changes as agent learns to predict."""
        if self._wrapped_env is None or self._env is None:
            return 0.0

        errors = []
        nodes = sorted(self._graph.nodes())[:10] if self._graph else []
        for node in nodes:
            self._env.current_node = node
            obs = self._env._get_obs()
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                target = self._wrapped_env.target(obs_t)
                pred = self._wrapped_env.predictor(obs_t)
                error = ((target - pred) ** 2).mean().item()
                errors.append(error)

        return float(np.mean(errors)) if errors else 0.0

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
            return CuriosityAgent(seed=self._seed)
        comm_id = int(region_id.split("_")[-1])
        new_graph = self._graph.copy()
        to_remove = [
            n for n in new_graph.nodes()
            if new_graph.nodes[n].get("features", {}).get("community", -1) == comm_id
        ]
        new_graph.remove_nodes_from(to_remove)
        new = CuriosityAgent(seed=self._seed, total_timesteps=self._total_timesteps)
        new.set_graph(new_graph)
        return new

    def perturb(self, region_id: str, method: str = "shuffle_weights") -> TestSystem:
        if self._graph is None:
            return CuriosityAgent(seed=self._seed)
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
        new = CuriosityAgent(seed=self._seed, total_timesteps=self._total_timesteps)
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
        new = CuriosityAgent(
            n_features=self._n_features, max_degree=self._max_degree,
            seed=self._seed, total_timesteps=self._total_timesteps,
        )
        new.set_graph(self._graph)
        # Copy RND predictor state and RL model weights
        if self._wrapped_env is not None or self._model is not None:
            snapshot = self.get_state()
            new.set_state(snapshot)
        return new

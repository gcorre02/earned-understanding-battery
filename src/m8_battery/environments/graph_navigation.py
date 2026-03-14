"""GraphNavigationEnv — Gymnasium environment for RL-based graph navigation.

States: current node features + local neighbourhood.
Actions: traverse to adjacent node (discrete, action-masked).
Supports MaskablePPO from sb3-contrib.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces


class GraphNavigationEnv(gym.Env):
    """Graph navigation environment for RL agents.

    Observation: current node features + neighbour count + visit count.
    Action: index into sorted neighbour list (action-masked).
    Reward modes:
      - "target": reward for reaching specified target node
      - "curiosity": no extrinsic reward (for ICM/RND agents)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        graph: nx.DiGraph,
        n_features: int = 8,
        max_degree: int = 20,
        reward_mode: str = "target",
        target_node: int | None = None,
        max_steps: int = 100,
    ) -> None:
        super().__init__()

        self.graph = graph
        self.n_features = n_features
        self.max_degree = max_degree
        self.reward_mode = reward_mode
        self.target_node = target_node
        self.max_steps = max_steps

        self._nodes = sorted(graph.nodes())
        self._node_to_idx = {n: i for i, n in enumerate(self._nodes)}

        # Observation: node features + meta (visit count, degree, step)
        obs_dim = n_features + 3
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: index into neighbour list (masked)
        self.action_space = spaces.Discrete(max_degree)

        self.current_node: int | None = None
        self.visit_counts: dict[int, int] = {}
        self.step_count = 0
        self._rng = np.random.default_rng()

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Start at a random node
        self.current_node = self._rng.choice(self._nodes)
        self.visit_counts = {}
        self.step_count = 0

        if self.target_node is None and self.reward_mode == "target":
            # Pick a random target different from start
            candidates = [n for n in self._nodes if n != self.current_node]
            if candidates:
                self.target_node = self._rng.choice(candidates)

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        neighbours = self._get_neighbours()

        if action >= len(neighbours):
            # Invalid action — stay in place (shouldn't happen with masking)
            reward = -0.1
        else:
            self.current_node = neighbours[action]
            reward = self._compute_reward()

        self.visit_counts[self.current_node] = (
            self.visit_counts.get(self.current_node, 0) + 1
        )
        self.step_count += 1

        terminated = (
            self.reward_mode == "target"
            and self.current_node == self.target_node
        )
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def action_masks(self) -> np.ndarray:
        """Return binary mask for valid actions. Required by MaskablePPO."""
        mask = np.zeros(self.max_degree, dtype=np.int8)
        neighbours = self._get_neighbours()
        for i in range(min(len(neighbours), self.max_degree)):
            mask[i] = 1
        return mask

    def _get_neighbours(self) -> list[int]:
        """Sorted list of successor nodes from current position."""
        if self.current_node is None:
            return []
        return sorted(self.graph.successors(self.current_node))

    def _get_obs(self) -> np.ndarray:
        """Build observation vector from current node."""
        if self.current_node is None:
            return np.zeros(self.n_features + 3, dtype=np.float32)

        features = self.graph.nodes[self.current_node].get("features", {})
        feat_vec = [features.get(f"feat_{i}", 0.0) for i in range(self.n_features)]

        # Meta features
        visit_count = self.visit_counts.get(self.current_node, 0)
        degree = self.graph.degree(self.current_node)
        step_frac = self.step_count / self.max_steps

        obs = feat_vec + [float(visit_count), float(degree), step_frac]
        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "current_node": self.current_node,
            "step_count": self.step_count,
            "n_visited": len(self.visit_counts),
        }

    def _compute_reward(self) -> float:
        if self.reward_mode == "target":
            if self.current_node == self.target_node:
                return 1.0
            return -0.01  # Small step penalty
        elif self.reward_mode == "curiosity":
            return 0.0  # No extrinsic reward — ICM/RND provides intrinsic
        return 0.0

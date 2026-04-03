"""Tests for Class 3 system adapters.

These tests train RL agents (slower than Class 1/2 tests).
Total timesteps kept low (2000) for test speed.
"""

import pytest

from m8_battery.core.types import SystemClass
from m8_battery.domains.sbm_generator import generate_domain
from m8_battery.domains.presets import SMALL
from m8_battery.environments.graph_navigation import GraphNavigationEnv

class TestGraphNavigationEnv:
    def test_env_creation(self):
        G = generate_domain(SMALL)
        env = GraphNavigationEnv(G, n_features=8, max_degree=20)
        obs, info = env.reset(seed=42)
        assert obs.shape == (11,)  # 8 features + 3 meta
        assert "current_node" in info

    def test_step(self):
        G = generate_domain(SMALL)
        env = GraphNavigationEnv(G, max_degree=20)
        env.reset(seed=42)
        masks = env.action_masks()
        valid_actions = [i for i, m in enumerate(masks) if m == 1]
        if valid_actions:
            obs, reward, term, trunc, info = env.step(valid_actions[0])
            assert obs.shape == (11,)

    def test_action_masking(self):
        G = generate_domain(SMALL)
        env = GraphNavigationEnv(G, max_degree=20)
        env.reset(seed=42)
        masks = env.action_masks()
        assert masks.shape == (20,)
        assert masks.sum() > 0  # At least one valid action

    def test_target_reward(self):
        G = generate_domain(SMALL)
        nodes = sorted(G.nodes())
        env = GraphNavigationEnv(
            G, max_degree=20, reward_mode="target", target_node=nodes[-1]
        )
        env.reset(seed=42)
        env.current_node = nodes[-1]  # Move to target
        masks = env.action_masks()
        valid = [i for i, m in enumerate(masks) if m == 1]
        if valid:
            _, reward, terminated, _, _ = env.step(valid[0])
            # Stepping away from target gives negative reward
            assert reward == -0.01

    def test_curiosity_no_reward(self):
        G = generate_domain(SMALL)
        env = GraphNavigationEnv(G, max_degree=20, reward_mode="curiosity")
        env.reset(seed=42)
        masks = env.action_masks()
        valid = [i for i, m in enumerate(masks) if m == 1]
        if valid:
            _, reward, _, _, _ = env.step(valid[0])
            assert reward == 0.0

class TestDQNAgent:
    def test_train_and_step(self):
        from m8_battery.systems.class3.dqn_agent import DQNAgent

        G = generate_domain(SMALL)
        agent = DQNAgent(seed=42, total_timesteps=500)
        nodes = sorted(G.nodes())
        agent.train_on_domain(G, target_node=nodes[-1])
        assert agent._is_trained

        result = agent.step(None)
        assert result["current_node"] is not None
        assert result["trained"] is True

    def test_regions(self):
        from m8_battery.systems.class3.dqn_agent import DQNAgent

        G = generate_domain(SMALL)
        agent = DQNAgent(seed=42)
        agent.set_graph(G)
        regions = agent.get_regions()
        assert len(regions) == 4

    def test_clone(self):
        from m8_battery.systems.class3.dqn_agent import DQNAgent

        G = generate_domain(SMALL)
        agent = DQNAgent(seed=42, total_timesteps=500)
        agent.train_on_domain(G)
        clone = agent.clone()
        assert clone._is_trained is False  # Clone starts fresh

class TestCuriosityAgent:
    def test_train_and_step(self):
        from m8_battery.systems.class3.curiosity_agent import CuriosityAgent

        G = generate_domain(SMALL)
        agent = CuriosityAgent(seed=42, total_timesteps=500)
        agent.train_on_domain(G)
        assert agent._is_trained

        result = agent.step(None)
        assert result["current_node"] is not None

    def test_structure_metric_exists(self):
        from m8_battery.systems.class3.curiosity_agent import CuriosityAgent

        G = generate_domain(SMALL)
        agent = CuriosityAgent(seed=42, total_timesteps=500)
        agent.train_on_domain(G)
        metric = agent.get_structure_metric()
        assert isinstance(metric, float)

    def test_clone(self):
        from m8_battery.systems.class3.curiosity_agent import CuriosityAgent

        G = generate_domain(SMALL)
        agent = CuriosityAgent(seed=42, total_timesteps=500)
        agent.train_on_domain(G)
        clone = agent.clone()
        assert clone._is_trained is False

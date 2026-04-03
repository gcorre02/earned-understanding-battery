"""Gym encoder — wraps SBM graph as GraphNavigationEnv for RL systems."""

from __future__ import annotations

import networkx as nx

from m8_battery.environments.graph_navigation import GraphNavigationEnv

def encode_gym(
    G: nx.DiGraph,
    n_features: int = 8,
    max_degree: int = 20,
    reward_mode: str = "target",
    target_node: int | None = None,
    max_steps: int = 100,
) -> GraphNavigationEnv:
    """Wrap an SBM graph as a GraphNavigationEnv.

    Convenience function for RL systems (3A DQN, 3B Curiosity) that need
    a Gymnasium environment for training.

    Args:
        G: SBM-generated directed graph
        n_features: number of node features (must match graph)
        max_degree: maximum action space size
        reward_mode: "target" (extrinsic reward) or "curiosity" (no reward)
        target_node: target for "target" reward mode (random if None)
        max_steps: episode truncation limit

    Returns:
        GraphNavigationEnv ready for SB3 training
    """
    if target_node is None and reward_mode == "target":
        nodes = sorted(G.nodes())
        target_node = nodes[len(nodes) // 2] if nodes else 0

    return GraphNavigationEnv(
        graph=G,
        n_features=n_features,
        max_degree=max_degree,
        reward_mode=reward_mode,
        target_node=target_node,
        max_steps=max_steps,
    )

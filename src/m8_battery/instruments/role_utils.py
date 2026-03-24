"""Role classification and role-aggregated transition matrices.

Extracted from role_based_walker.py for use in the generativity instrument.
Roles are topology-derived (not learned) and recomputed per graph.

Four roles:
  0 = hub: degree > 1.5x mean
  1 = bridge: cross-community ratio > 0.3, degree > 3
  2 = periphery: moderate degree, within-community (default)
  3 = leaf: degree <= 2
"""

from __future__ import annotations

import networkx as nx
import numpy as np

ROLE_NAMES = ["hub", "bridge", "periphery", "leaf"]
N_ROLES = len(ROLE_NAMES)


def classify_node_role(graph: nx.DiGraph, node, mean_degree: float) -> int:
    """Classify a single node into a topological role.

    This is a TOPOLOGY computation, not learned. Works on any graph.
    mean_degree is precomputed by classify_all_nodes() to avoid O(n²).
    """
    if graph.is_directed():
        degree = graph.in_degree(node) + graph.out_degree(node)
        neighbours = set(graph.predecessors(node)) | set(graph.successors(node))
    else:
        degree = graph.degree(node)
        neighbours = set(graph.neighbors(node))

    if degree <= 2:
        return 3  # leaf

    node_data = graph.nodes[node]
    node_comm = node_data.get("features", {}).get("community", node_data.get("block", 0))

    n_cross = sum(
        1 for nb in neighbours
        if graph.nodes[nb].get("features", {}).get("community", graph.nodes[nb].get("block", 0)) != node_comm
    )
    cross_ratio = n_cross / max(len(neighbours), 1)

    if cross_ratio > 0.3 and degree > 3:
        return 1  # bridge

    if degree > mean_degree * 1.5:
        return 0  # hub

    return 2  # periphery


def classify_all_nodes(graph: nx.DiGraph) -> dict:
    """Classify all nodes in a graph by topological role.

    Returns: dict mapping node -> role_id (0-3).
    """
    n = graph.number_of_nodes()
    if n == 0:
        return {}
    if graph.is_directed():
        mean_deg = sum(graph.in_degree(node) + graph.out_degree(node) for node in graph.nodes()) / n
    else:
        mean_deg = sum(graph.degree(node) for node in graph.nodes()) / n
    return {node: classify_node_role(graph, node, mean_deg) for node in graph.nodes()}


def compute_role_transition_matrix(
    visit_sequence: list,
    node_to_role: dict,
    n_roles: int = N_ROLES,
) -> np.ndarray:
    """Compute role-to-role transition matrix from a visit sequence.

    Returns (n_roles x n_roles) row-normalised matrix.
    Community-label-invariant: captures HOW the system moves between
    structural roles, not which specific communities it visits.
    """
    T = np.zeros((n_roles, n_roles))
    for t in range(len(visit_sequence) - 1):
        r_from = node_to_role.get(visit_sequence[t], 2)  # default periphery
        r_to = node_to_role.get(visit_sequence[t + 1], 2)
        T[r_from, r_to] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    T = T / row_sums
    return T

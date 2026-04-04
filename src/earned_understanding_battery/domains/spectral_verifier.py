"""Spectral verification for cross-format domain invariance.

Based on spectral verification campaign (spectral.py + similarity.py).
M-04 campaign validated: k=10, full-basin, discrimination margin 0.1037,
noise robust to +/-0.10, eigendecomp <100ms at 500 nodes.

DO NOT mix with engine's spectral analyser — different eigenvalue ordering
and normalisation. This is the M-04 version verbatim.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh

# --- Spectral signature computation (from M-04 spectral.py) ---

def compute_sigma_full(
    G: nx.Graph,
    basin_nodes: list[int],
    k: int = 10,
) -> np.ndarray | None:
    """Compute full-basin spectral signature.

    Returns top-k eigenvalues of the symmetrised weighted Laplacian
    restricted to the basin subgraph, sorted descending.
    Returns None if basin has fewer than 2 nodes.
    """
    if len(basin_nodes) < 2:
        return None

    k_actual = min(k, len(basin_nodes))
    L = _build_laplacian(G, basin_nodes)
    eigenvalues = eigvalsh(L)
    return eigenvalues[-k_actual:][::-1]

def compute_graph_signature(
    G: nx.Graph,
    k: int = 10,
    weight_attr: str = "weight",
) -> np.ndarray | None:
    """Compute spectral signature for an entire graph.

    Convenience wrapper: treats the whole graph as one basin.
    Uses the specified weight attribute for the Laplacian.
    """
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return None

    k_actual = min(k, len(nodes))
    L = _build_laplacian(G, nodes, weight_attr=weight_attr)
    eigenvalues = eigvalsh(L)
    return eigenvalues[-k_actual:][::-1]

def _build_laplacian(
    G: nx.Graph,
    nodes: list[int],
    weight_attr: str = "weight",
) -> np.ndarray:
    """Build the symmetrised weighted Laplacian for a subgraph.

    L = D - A where:
    - A_ij = (w(i,j) + w(j,i)) / 2  (symmetrised)
    - D_ii = sum_j A_ij
    """
    n = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    node_set = set(nodes)

    A = np.zeros((n, n))
    for i, node_i in enumerate(nodes):
        for neighbor in G.neighbors(node_i):
            if neighbor in node_set:
                j = node_to_idx[neighbor]
                w = G[node_i][neighbor].get(weight_attr, 0.0)
                A[i, j] = w

    # Ensure symmetry
    A = (A + A.T) / 2.0

    D = np.diag(A.sum(axis=1))
    L = D - A
    return L

# --- Similarity metric (from M-04 similarity.py) ---

def normalised_l2(sigma_a: np.ndarray, sigma_b: np.ndarray) -> float:
    """L2 distance between two eigenvalue vectors, each normalised by L2 norm.

    Returns 1.0 if either has zero norm (maximally different).
    """
    norm_a = np.linalg.norm(sigma_a)
    norm_b = np.linalg.norm(sigma_b)

    if norm_a < 1e-15 or norm_b < 1e-15:
        return 1.0

    a_normed = sigma_a / norm_a
    b_normed = sigma_b / norm_b

    return float(np.linalg.norm(a_normed - b_normed))

def spectral_similarity(
    sigma_a: np.ndarray | None,
    sigma_b: np.ndarray | None,
) -> float | None:
    """Similarity between two spectral signatures.

    Returns None if either is None.
    Returns value in [0, 1] where 1 = identical, 0 = maximally different.
    """
    if sigma_a is None or sigma_b is None:
        return None

    # Pad shorter vector with zeros if lengths differ
    if len(sigma_a) != len(sigma_b):
        max_len = max(len(sigma_a), len(sigma_b))
        a_padded = np.zeros(max_len)
        b_padded = np.zeros(max_len)
        a_padded[:len(sigma_a)] = sigma_a
        b_padded[:len(sigma_b)] = sigma_b
        sigma_a, sigma_b = a_padded, b_padded

    return 1.0 - normalised_l2(sigma_a, sigma_b)

# --- Cross-format verification ---

def verify_cross_format_invariance(
    graphs: dict[str, nx.Graph],
    k: int = 10,
    tolerance: float = 0.01,
    weight_attr: str = "weight",
) -> tuple[bool, dict[str, float]]:
    """Verify that the same domain encoded in different formats
    preserves spectral invariance.

    Args:
        graphs: dict mapping format name to graph (e.g., {"graph": G1, "gym": G2})
        k: number of eigenvalues
        tolerance: Frobenius norm threshold for equivalence
        weight_attr: edge attribute to use as weight

    Returns:
        (all_pass, pairwise_distances)
    """
    names = list(graphs.keys())
    signatures = {}

    for name, G in graphs.items():
        sig = compute_graph_signature(G, k=k, weight_attr=weight_attr)
        if sig is None:
            return False, {f"{name}": float("nan")}
        signatures[name] = sig

    distances = {}
    all_pass = True
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            dist = normalised_l2(signatures[name_a], signatures[name_b])
            key = f"{name_a}_vs_{name_b}"
            distances[key] = dist
            if dist > tolerance:
                all_pass = False

    return all_pass, distances

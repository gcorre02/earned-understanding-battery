"""Domain construction quality verification (T1-06).

Verifies that domain pairs (A/A', A/B) preserve structural invariants
while destroying surface statistics. Quantitative checks for each
domain relationship.

Literature basis: transfer and generativity validity depend on domain quality.
Naive construction can silently preserve shortcuts (peer review finding).
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def verify_structural_preservation(
    G1: nx.DiGraph, G2: nx.DiGraph, k: int = 10,
) -> dict[str, float]:
    """Verify structural invariants are preserved between two domains.

    Checks: community count, degree distribution shape, spectral similarity.
    Returns dict of metrics — all should be close to 1.0 for preserved structure.
    """
    # Community count
    communities_1 = set()
    communities_2 = set()
    for n in G1.nodes():
        d = G1.nodes[n]
        communities_1.add(d.get("features", {}).get("community", d.get("block", 0)))
    for n in G2.nodes():
        d = G2.nodes[n]
        communities_2.add(d.get("features", {}).get("community", d.get("block", 0)))
    community_match = 1.0 if len(communities_1) == len(communities_2) else 0.0

    # Node count match
    node_count_match = 1.0 if G1.number_of_nodes() == G2.number_of_nodes() else 0.0

    # Degree distribution similarity (Kolmogorov-Smirnov style)
    deg1 = sorted([d for _, d in G1.degree()])
    deg2 = sorted([d for _, d in G2.degree()])
    if deg1 and deg2:
        # Normalise to CDFs and compute max difference
        cdf1 = np.array(deg1) / max(deg1)
        cdf2 = np.array(deg2) / max(deg2)
        # Pad shorter to match
        max_len = max(len(cdf1), len(cdf2))
        cdf1 = np.pad(cdf1, (0, max_len - len(cdf1)))
        cdf2 = np.pad(cdf2, (0, max_len - len(cdf2)))
        degree_similarity = 1.0 - float(np.max(np.abs(cdf1 - cdf2)))
    else:
        degree_similarity = 0.0

    # Spectral similarity (top-k eigenvalues of graph Laplacian)
    try:
        L1 = nx.laplacian_matrix(G1.to_undirected()).toarray().astype(float)
        L2 = nx.laplacian_matrix(G2.to_undirected()).toarray().astype(float)
        eig1 = np.sort(np.linalg.eigvalsh(L1))[:k]
        eig2 = np.sort(np.linalg.eigvalsh(L2))[:k]
        # Normalise
        norm1 = np.linalg.norm(eig1)
        norm2 = np.linalg.norm(eig2)
        if norm1 > 1e-10 and norm2 > 1e-10:
            spectral_similarity = float(np.dot(eig1, eig2) / (norm1 * norm2))
        else:
            spectral_similarity = 0.0
    except Exception:
        spectral_similarity = 0.0

    return {
        "community_match": community_match,
        "node_count_match": node_count_match,
        "degree_similarity": degree_similarity,
        "spectral_similarity": spectral_similarity,
    }


def verify_surface_destruction(
    G1: nx.DiGraph, G2: nx.DiGraph,
) -> dict[str, float]:
    """Verify surface statistics are destroyed between two domains.

    Checks: node label overlap, edge set Jaccard, feature distribution shift.
    Returns dict of metrics — should be near 0.0 for destroyed surface.
    """
    # Node label overlap
    labels_1 = {G1.nodes[n].get("label", str(n)) for n in G1.nodes()}
    labels_2 = {G2.nodes[n].get("label", str(n)) for n in G2.nodes()}
    if labels_1 or labels_2:
        label_overlap = len(labels_1 & labels_2) / max(len(labels_1 | labels_2), 1)
    else:
        label_overlap = 0.0

    # Edge set Jaccard (as node pairs, ignoring direction)
    edges_1 = {(min(u, v), max(u, v)) for u, v in G1.edges()}
    edges_2 = {(min(u, v), max(u, v)) for u, v in G2.edges()}
    if edges_1 or edges_2:
        edge_jaccard = len(edges_1 & edges_2) / max(len(edges_1 | edges_2), 1)
    else:
        edge_jaccard = 0.0

    # Feature distribution shift (mean absolute difference of feature means)
    def _feature_means(G):
        feats = []
        for n in G.nodes():
            f = G.nodes[n].get("features", {})
            vals = [v for k, v in f.items() if k.startswith("feat_") and isinstance(v, (int, float))]
            if vals:
                feats.append(vals)
        if feats:
            return np.mean(feats, axis=0)
        return np.array([])

    fm1 = _feature_means(G1)
    fm2 = _feature_means(G2)
    if len(fm1) > 0 and len(fm2) > 0 and len(fm1) == len(fm2):
        feature_shift = float(np.mean(np.abs(fm1 - fm2)))
    else:
        feature_shift = 0.0

    return {
        "label_overlap": label_overlap,
        "edge_jaccard": edge_jaccard,
        "feature_shift": feature_shift,
    }


def verify_domain_family(
    family: dict[str, nx.DiGraph],
) -> dict[str, dict]:
    """Run full verification on a domain family (A, A', B, C).

    Returns verification results for each domain pair.
    """
    results = {}

    A = family["A"]
    A_prime = family.get("A_prime")
    B = family.get("B")
    C = family.get("C")

    if A_prime is not None:
        results["A_vs_A_prime"] = {
            "structural": verify_structural_preservation(A, A_prime),
            "surface": verify_surface_destruction(A, A_prime),
        }

    if B is not None:
        results["A_vs_B"] = {
            "structural": verify_structural_preservation(A, B),
            "surface": verify_surface_destruction(A, B),
        }

    if C is not None:
        results["A_vs_C"] = {
            "structural": verify_structural_preservation(A, C),
            "surface": verify_surface_destruction(A, C),
        }

    return results


def check_leakage_channels(G: nx.DiGraph) -> list[str]:
    """Document known leakage channels in the domain construction.

    Returns list of potential leakage warnings.
    """
    warnings = []

    # Degree sequence might leak community structure
    communities = {}
    for n in G.nodes():
        d = G.nodes[n]
        c = d.get("features", {}).get("community", d.get("block", 0))
        communities.setdefault(c, []).append(n)

    if len(communities) > 1:
        community_mean_degrees = {}
        for c, nodes in communities.items():
            degs = [G.degree(n) for n in nodes]
            community_mean_degrees[c] = np.mean(degs)

        max_deg = max(community_mean_degrees.values())
        min_deg = min(community_mean_degrees.values())
        if max_deg > 0 and (max_deg - min_deg) / max_deg > 0.1:
            warnings.append(
                f"Degree distribution varies across communities "
                f"(range {min_deg:.1f}-{max_deg:.1f}). "
                f"A system could infer community membership from node degree."
            )

        # Community size distribution
        sizes = [len(nodes) for nodes in communities.values()]
        if max(sizes) - min(sizes) > 1:
            warnings.append(
                f"Community sizes unequal ({sizes}). "
                f"A system could use community size as a structural cue."
            )

    return warnings

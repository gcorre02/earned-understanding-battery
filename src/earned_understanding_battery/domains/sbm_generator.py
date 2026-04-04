"""Stochastic Block Model domain generation for the Earned Understanding Battery."""

from __future__ import annotations

import networkx as nx
import numpy as np

from earned_understanding_battery.core.types import DomainConfig

def generate_domain(config: DomainConfig) -> nx.DiGraph:
    """Generate a directed graph from a Stochastic Block Model.

    Nodes have synthetic features (no real-world content).
    Edges have typed weights.
    """
    rng = np.random.default_rng(config.seed)

    # Community sizes — roughly equal
    base_size = config.n_nodes // config.n_communities
    sizes = [base_size] * config.n_communities
    sizes[-1] += config.n_nodes - sum(sizes)  # absorb remainder

    # Edge probability matrix
    probs = np.full(
        (config.n_communities, config.n_communities),
        config.p_between,
    )
    np.fill_diagonal(probs, config.p_within)

    # Generate undirected SBM then convert to directed
    G_undirected = nx.stochastic_block_model(
        sizes, probs.tolist(),
        seed=int(rng.integers(0, 2**31)),
    )
    G = nx.DiGraph()
    G.add_nodes_from(G_undirected.nodes(data=True))

    # Edge types
    edge_types = [f"type_{i}" for i in range(config.n_edge_types)]

    for u, v in G_undirected.edges():
        weight = rng.uniform(*config.weight_range)
        etype = rng.choice(edge_types)
        G.add_edge(u, v, weight=weight, edge_type=etype)
        # Add reverse edge with some probability for directed structure
        if rng.random() > 0.3:
            G.add_edge(v, u, weight=rng.uniform(*config.weight_range),
                       edge_type=rng.choice(edge_types))

    # Node features — synthetic categorical + numerical
    for node in G.nodes():
        block = G.nodes[node].get("block", 0)
        features = {}
        # Categorical features
        features["community"] = block
        features["node_type"] = chr(65 + (block % 26))
        # Numerical features — drawn from community-specific distributions
        for i in range(config.n_node_features):
            mean = block * 0.5 + i * 0.1
            features[f"feat_{i}"] = float(rng.normal(mean, 0.3))
        G.nodes[node]["features"] = features
        G.nodes[node]["label"] = f"E_{node:03d}"

    return G

def generate_domain_family(config: DomainConfig) -> dict[str, nx.DiGraph]:
    """Generate a full domain family: A, A', B, C.

    - A: Primary domain
    - A': Isomorphic to A with permuted labels and re-sampled features
    - B: Fresh draw from same config (different seed)
    - C: Qualitatively different parameters (negative control)
    """
    rng = np.random.default_rng(config.seed)

    # Domain A
    domain_a = generate_domain(config)

    # Domain A' — same adjacency, permuted labels, re-sampled features
    domain_a_prime = _create_isomorphic(domain_a, config, rng)

    # Domain B — same config family, different seed
    config_b = DomainConfig(
        n_nodes=config.n_nodes,
        n_communities=config.n_communities,
        p_within=config.p_within,
        p_between=config.p_between,
        n_edge_types=config.n_edge_types,
        weight_range=config.weight_range,
        n_node_features=config.n_node_features,
        seed=int(rng.integers(0, 2**31)),
    )
    domain_b = generate_domain(config_b)

    # Domain C — qualitatively different (negative control)
    config_c = DomainConfig(
        n_nodes=config.n_nodes,
        n_communities=max(2, config.n_communities // 3),
        p_within=config.p_between * 3,  # blur community boundaries
        p_between=config.p_within / 2,
        n_edge_types=config.n_edge_types,
        weight_range=config.weight_range,
        n_node_features=config.n_node_features,
        seed=int(rng.integers(0, 2**31)),
    )
    domain_c = generate_domain(config_c)

    return {
        "A": domain_a,
        "A_prime": domain_a_prime,
        "B": domain_b,
        "C": domain_c,
    }

def _create_isomorphic(
    G: nx.DiGraph, config: DomainConfig, rng: np.random.Generator
) -> nx.DiGraph:
    """Create A' — same structure, permuted labels, re-sampled features."""
    nodes = list(G.nodes())
    perm = rng.permutation(len(nodes))
    node_map = {old: int(new) for old, new in zip(nodes, perm)}

    G_prime = nx.relabel_nodes(G, node_map)

    # Re-sample features from different distributions (preserve community)
    edge_types = [f"type_{i}" for i in range(config.n_edge_types)]
    for node in G_prime.nodes():
        block = G_prime.nodes[node].get("block", 0)
        features = {}
        features["community"] = block
        features["node_type"] = chr(65 + ((block + 13) % 26))  # shifted labels
        for i in range(config.n_node_features):
            mean = (block + 3) * 0.5 + i * 0.15  # different distribution
            features[f"feat_{i}"] = float(rng.normal(mean, 0.35))
        G_prime.nodes[node]["features"] = features
        G_prime.nodes[node]["label"] = f"X_{node:03d}"

    # Swap edge type labels
    type_map = {t: edge_types[(i + 1) % len(edge_types)]
                for i, t in enumerate(edge_types)}
    for u, v in G_prime.edges():
        old_type = G_prime.edges[u, v].get("edge_type", edge_types[0])
        G_prime.edges[u, v]["edge_type"] = type_map.get(old_type, old_type)

    return G_prime

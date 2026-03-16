"""Behavioural generativity test — complement to structural-metric instrument.

Asks: does training on domain A change the system's BEHAVIOUR on domain B?
This is what Paper 1 defines as generativity: "coherent responses in unseen
but structurally related domains, consistent with the system's consolidated
organisation."

The structural-metric instrument has a confound for pre-trained components
(DN-18). This behavioural test measures what the system DOES, not what a
number does.

Usage:
    from m8_battery.analysis.behavioural_generativity import (
        generate_paired_domains,
        record_behaviour,
        compute_behavioural_divergence,
        run_behavioural_generativity,
    )
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np
import networkx as nx

from m8_battery.core.test_system import TestSystem
from m8_battery.core.types import DomainConfig
from m8_battery.domains.sbm_generator import generate_domain


@dataclass
class BehaviourTrace:
    """Raw behavioural trace from operating a system on a domain."""
    nodes_visited: list[Any] = field(default_factory=list)
    edges_traversed: list[tuple[Any, Any]] = field(default_factory=list)
    visit_distribution: dict[Any, int] = field(default_factory=dict)
    community_coverage: dict[int, int] = field(default_factory=dict)
    path_diversity: float = 0.0
    unique_nodes: int = 0
    unique_edges: int = 0
    outputs: list[dict] = field(default_factory=list)


@dataclass
class DivergenceResult:
    """Behavioural divergence between fresh and trained systems."""
    visit_js_divergence: float = 0.0
    community_overlap: float = 0.0
    path_diversity_delta: float = 0.0
    edge_jaccard: float = 0.0
    sequence_similarity: float = 0.0


def generate_paired_domains(
    n_nodes: int,
    n_communities: int = 6,
    p_within: float = 0.3,
    p_between: float = 0.05,
) -> tuple[nx.DiGraph, nx.DiGraph]:
    """Generate two structurally similar but distinct SBM domains.

    Same parameters, different seeds. Domain B is "unseen but structurally
    related" — exactly what Paper 1 specifies.
    """
    config_a = DomainConfig(
        n_nodes=n_nodes,
        n_communities=n_communities,
        p_within=p_within,
        p_between=p_between,
        seed=100,
    )
    config_b = DomainConfig(
        n_nodes=n_nodes,
        n_communities=n_communities,
        p_within=p_within,
        p_between=p_between,
        seed=200,
    )
    return generate_domain(config_a), generate_domain(config_b)


def record_behaviour(
    system: TestSystem,
    domain: nx.DiGraph,
    n_steps: int = 50,
) -> BehaviourTrace:
    """Record the system's behavioural trace on a domain.

    Feeds None inputs (free wander) to capture the system's autonomous
    navigation pattern. Records nodes visited, edges traversed, community
    coverage, and path diversity.
    """
    trace = BehaviourTrace()
    nodes = list(domain.nodes())
    prev_node = None

    for step in range(n_steps):
        result = system.step(None)  # Free wander — system chooses
        current = result.get("current_node") if isinstance(result, dict) else None
        if current is None:
            continue

        trace.nodes_visited.append(current)
        trace.outputs.append(result if isinstance(result, dict) else {"raw": result})

        if prev_node is not None and current != prev_node:
            trace.edges_traversed.append((prev_node, current))
        prev_node = current

    # Compute summary metrics
    trace.visit_distribution = dict(Counter(trace.nodes_visited))
    trace.unique_nodes = len(set(trace.nodes_visited))
    trace.unique_edges = len(set(trace.edges_traversed))
    trace.path_diversity = (
        trace.unique_edges / max(len(trace.edges_traversed), 1)
    )

    # Community coverage
    community_visits: dict[int, int] = {}
    for node in trace.nodes_visited:
        if node in domain.nodes:
            comm = domain.nodes[node].get("features", {}).get("community", -1)
            community_visits[comm] = community_visits.get(comm, 0) + 1
    trace.community_coverage = community_visits

    return trace


def _jensen_shannon_divergence(
    dist_a: dict[Any, int],
    dist_b: dict[Any, int],
) -> float:
    """Jensen-Shannon divergence between two visit distributions."""
    all_keys = set(dist_a.keys()) | set(dist_b.keys())
    if not all_keys:
        return 0.0

    total_a = sum(dist_a.values()) or 1
    total_b = sum(dist_b.values()) or 1

    p = np.array([dist_a.get(k, 0) / total_a for k in sorted(all_keys)])
    q = np.array([dist_b.get(k, 0) / total_b for k in sorted(all_keys)])

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(js)


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _normalised_edit_distance(seq_a: list, seq_b: list) -> float:
    """Normalised Levenshtein distance (0 = identical, 1 = completely different)."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 and m == 0:
        return 0.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    max_len = max(n, m)
    return dp[n][m] / max_len if max_len > 0 else 0.0


def compute_behavioural_divergence(
    fresh: BehaviourTrace,
    trained: BehaviourTrace,
) -> DivergenceResult:
    """Measure how differently trained and fresh systems behave on domain B."""
    return DivergenceResult(
        visit_js_divergence=_jensen_shannon_divergence(
            fresh.visit_distribution, trained.visit_distribution
        ),
        community_overlap=_jaccard_similarity(
            set(fresh.community_coverage.keys()),
            set(trained.community_coverage.keys()),
        ),
        path_diversity_delta=trained.path_diversity - fresh.path_diversity,
        edge_jaccard=_jaccard_similarity(
            set(fresh.edges_traversed),
            set(trained.edges_traversed),
        ),
        sequence_similarity=1.0 - _normalised_edit_distance(
            fresh.nodes_visited[:20],
            trained.nodes_visited[:20],
        ),
    )


def classify_divergence(div: DivergenceResult) -> str:
    """Classify divergence as earned, absent, or ambiguous."""
    if div.visit_js_divergence > 0.1 and div.edge_jaccard < 0.5:
        return "earned"
    elif div.visit_js_divergence < 0.01 and div.edge_jaccard > 0.9:
        return "absent"
    else:
        return "ambiguous"


def run_behavioural_generativity(
    system_factory: Callable[[int], TestSystem],
    n_nodes: int,
    n_communities: int = 6,
    seed: int = 42,
    n_warmup: int = 50,
    n_steps: int = 50,
) -> dict[str, Any]:
    """Run a single behavioural generativity test.

    Args:
        system_factory: callable(seed) -> TestSystem
        n_nodes: domain size
        n_communities: number of communities in SBM
        seed: system seed
        n_warmup: training steps on domain A
        n_steps: behaviour recording steps on domain B

    Returns:
        dict with fresh_behaviour, trained_behaviour, divergence, classification
    """
    domain_a, domain_b = generate_paired_domains(n_nodes, n_communities)

    # Fresh run: attach to B, no training
    fresh = system_factory(seed)
    fresh.train_on_domain(domain_b, n_warmup=0)
    fresh_behaviour = record_behaviour(fresh, domain_b, n_steps)

    # Trained run: train on A, switch to B
    trained = system_factory(seed)
    trained.train_on_domain(domain_a, n_warmup=n_warmup)
    trained.set_graph(domain_b)
    trained_behaviour = record_behaviour(trained, domain_b, n_steps)

    # Compare
    divergence = compute_behavioural_divergence(fresh_behaviour, trained_behaviour)
    classification = classify_divergence(divergence)

    return {
        "seed": seed,
        "n_nodes": n_nodes,
        "n_communities": n_communities,
        "n_warmup": n_warmup,
        "n_steps": n_steps,
        "fresh_behaviour": {
            "unique_nodes": fresh_behaviour.unique_nodes,
            "unique_edges": fresh_behaviour.unique_edges,
            "community_coverage": list(fresh_behaviour.community_coverage.keys()),
            "path_diversity": fresh_behaviour.path_diversity,
        },
        "trained_behaviour": {
            "unique_nodes": trained_behaviour.unique_nodes,
            "unique_edges": trained_behaviour.unique_edges,
            "community_coverage": list(trained_behaviour.community_coverage.keys()),
            "path_diversity": trained_behaviour.path_diversity,
        },
        "divergence": asdict(divergence),
        "classification": classification,
    }

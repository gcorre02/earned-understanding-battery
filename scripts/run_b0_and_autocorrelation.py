#!/usr/bin/env python3
"""B₀ isomorphic generativity test + trajectory autocorrelation.

Runs PC1 (role walker) and PC3 (GNN navigator) through the generativity
protocol on A' (isomorphic to A, permuted labels) and B (novel domain).

Also computes trajectory autocorrelation for trained vs fresh on each domain.

Output: JSON results to stdout.
"""

import json
import sys
import numpy as np
import torch

from m8_battery.domains.sbm_generator import generate_domain_family
from m8_battery.domains.presets import MEDIUM
from m8_battery.systems.positive_controls.role_based_walker import RoleBasedWalker
from m8_battery.systems.positive_controls.gnn_navigator import GNNNavigator

def _log(msg):
    print(f"[b0_test] {msg}", file=sys.stderr, flush=True)

def js_divergence(p, q):
    """JSD base e, epsilon-smoothed."""
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return max(0.0, min(jsd, np.log(2)))

def coherence_normalised(trained_h, fresh_h):
    denom = fresh_h + trained_h + 1e-10
    return (fresh_h - trained_h) / denom

def engagement_entropy(dist):
    vals = np.array(list(dist.values()), dtype=np.float64) + 1e-10
    vals = vals / vals.sum()
    return float(-np.sum(vals * np.log(vals)))

def count_visited(dist, min_frac=0.01):
    return sum(1 for v in dist.values() if v > min_frac)

def classify_signal(jsd, coherence, trained_vis, fresh_vis, edge_overlap=None):
    MAX_JSD = np.log(2)
    if jsd < 1e-6:
        return "absent"
    if trained_vis < 3:
        return "degenerate_trained"
    if fresh_vis < 3:
        return "degenerate_fresh"
    if jsd > 0.99 * MAX_JSD:
        return "maximum_divergence"
    if edge_overlap is not None and edge_overlap > 0.05:
        return "potentially_confounded"
    if coherence <= 0:
        return "divergent_incoherent"
    return "candidate_generativity"

def trajectory_autocorrelation(visit_sequence, node_to_community, lag=1):
    """Autocorrelation of community-membership sequence."""
    communities = [node_to_community.get(n, 0) for n in visit_sequence]
    n = len(communities)
    if n <= lag:
        return 0.0
    arr = np.array(communities, dtype=np.float64)
    mean = np.mean(arr)
    var = np.var(arr)
    if var < 1e-10:
        return 1.0
    autocorr = np.sum((arr[:n-lag] - mean) * (arr[lag:] - mean)) / ((n - lag) * var)
    return float(autocorr)

def run_generativity_with_trajectory(system, domain_graph, n_steps, node_to_community):
    """Run generativity protocol and return engagement + trajectory."""
    system.set_domain(domain_graph)
    system.set_training(False)
    system.reset_engagement_tracking()

    trajectory = []
    for _ in range(n_steps):
        node = system.step(None)
        trajectory.append(node)

    engagement = system.get_engagement_distribution()
    entropy = engagement_entropy(engagement)
    visited = count_visited(engagement)
    autocorr = trajectory_autocorrelation(trajectory, node_to_community)

    return engagement, entropy, visited, trajectory, autocorr

def edge_jaccard(g1, g2):
    """Edge set Jaccard similarity."""
    e1 = set(g1.edges())
    e2 = set(g2.edges())
    if not e1 and not e2:
        return 0.0
    return len(e1 & e2) / len(e1 | e2)

def get_community_map(graph):
    """Node to community mapping."""
    mapping = {}
    for node in graph.nodes():
        data = graph.nodes[node]
        features = data.get("features", {})
        mapping[node] = features.get("community", data.get("block", 0))
    return mapping

def run_test(system_name, system_factory, control_factory, domain_a, domain_target,
             target_name, n_steps, seeds, edge_overlap):
    """Run generativity protocol on a target domain for multiple seeds."""
    node_to_community = get_community_map(domain_target)
    results = []

    for seed in seeds:
        _log(f"  {system_name} seed={seed} on {target_name}")

        # Create and train system on A
        system = system_factory(domain_a, seed)
        system.train_on_domain(domain_a, n_steps=n_steps)

        # Run on target domain (frozen)
        trained_eng, trained_h, trained_vis, trained_traj, trained_ac = \
            run_generativity_with_trajectory(system, domain_target, 500, node_to_community)

        # Fresh baseline
        fresh = control_factory(domain_a, seed)
        fresh_eng, fresh_h, fresh_vis, fresh_traj, fresh_ac = \
            run_generativity_with_trajectory(fresh, domain_target, 500, node_to_community)

        # Compute JSD
        regions = sorted(set(list(trained_eng.keys()) + list(fresh_eng.keys())))
        trained_vec = np.array([trained_eng.get(r, 0.0) for r in regions])
        fresh_vec = np.array([fresh_eng.get(r, 0.0) for r in regions])
        jsd = js_divergence(trained_vec, fresh_vec)

        coherence = coherence_normalised(trained_h, fresh_h)
        signal = classify_signal(jsd, coherence, trained_vis, fresh_vis, edge_overlap)

        results.append({
            "seed": seed,
            "jsd": round(jsd, 4),
            "coherence": round(coherence, 4),
            "signal_type": signal,
            "trained_entropy": round(trained_h, 4),
            "fresh_entropy": round(fresh_h, 4),
            "trained_visited": trained_vis,
            "fresh_visited": fresh_vis,
            "trained_autocorrelation": round(trained_ac, 4),
            "fresh_autocorrelation": round(fresh_ac, 4),
        })
        _log(f"    JSD={jsd:.4f} coh={coherence:.4f} sig={signal} "
             f"t_ac={trained_ac:.4f} f_ac={fresh_ac:.4f}")

    return results

def main():
    seeds = [42, 123, 456]
    _log("Generating domain family (MEDIUM)...")
    family = generate_domain_family(MEDIUM)
    domain_a = family["A"]
    domain_a_prime = family["A_prime"]
    domain_b = family["B"]

    # Edge Jaccard for confound reporting
    ej_a_aprime = edge_jaccard(domain_a, domain_a_prime)
    ej_a_b = edge_jaccard(domain_a, domain_b)
    _log(f"Edge Jaccard A vs A': {ej_a_aprime:.4f}")
    _log(f"Edge Jaccard A vs B: {ej_a_b:.4f}")

    all_results = {
        "domain_info": {
            "edge_jaccard_a_aprime": round(ej_a_aprime, 4),
            "edge_jaccard_a_b": round(ej_a_b, 4),
            "a_nodes": len(domain_a.nodes()),
            "aprime_nodes": len(domain_a_prime.nodes()),
            "b_nodes": len(domain_b.nodes()),
        }
    }

    # --- PC1 Role Walker ---
    def pc1_factory(graph, seed):
        return RoleBasedWalker(graph, seed=seed, eta=0.8, temperature=0.01)

    def pc1_fresh_factory(graph, seed):
        return RoleBasedWalker(graph, seed=seed + 1000)

    _log("=== PC1 on B₀ (A') ===")
    all_results["pc1_b0"] = run_test(
        "PC1", pc1_factory, pc1_fresh_factory,
        domain_a, domain_a_prime, "B₀(A')",
        n_steps=10000, seeds=seeds, edge_overlap=ej_a_aprime,
    )

    _log("=== PC1 on B₂ (novel B) ===")
    all_results["pc1_b2"] = run_test(
        "PC1", pc1_factory, pc1_fresh_factory,
        domain_a, domain_b, "B₂(novel)",
        n_steps=10000, seeds=seeds, edge_overlap=ej_a_b,
    )

    # --- PC3 GNN Navigator ---
    def pc3_factory(graph, seed):
        return GNNNavigator(graph, seed=seed, train_epochs=500, temperature=0.05)

    def pc3_fresh_factory(graph, seed):
        return GNNNavigator(graph, seed=seed + 1000)

    _log("=== PC3 on B₀ (A') ===")
    all_results["pc3_b0"] = run_test(
        "PC3", pc3_factory, pc3_fresh_factory,
        domain_a, domain_a_prime, "B₀(A')",
        n_steps=2000, seeds=seeds, edge_overlap=ej_a_aprime,
    )

    _log("=== PC3 on B₂ (novel B) ===")
    all_results["pc3_b2"] = run_test(
        "PC3", pc3_factory, pc3_fresh_factory,
        domain_a, domain_b, "B₂(novel)",
        n_steps=2000, seeds=seeds, edge_overlap=ej_a_b,
    )

    # --- Also run autocorrelation on the 13 calibration systems ---
    # (HEB, 3D, 3E are the graph walkers with set_domain)
    from m8_battery.systems.internal.hebbian_walker import HebbianWalker

    def heb_factory(graph, seed):
        return HebbianWalker(graph, seed=seed, eta=0.1, decay=0.01, temperature=0.5)

    def heb_fresh_factory(graph, seed):
        return HebbianWalker(graph, seed=seed + 1000)

    _log("=== HEB on B₂ (novel B) — autocorrelation ===")
    all_results["heb_b2"] = run_test(
        "HEB", heb_factory, heb_fresh_factory,
        domain_a, domain_b, "B₂(novel)",
        n_steps=2000, seeds=seeds, edge_overlap=ej_a_b,
    )

    print(json.dumps(all_results, indent=2))
    _log("Done.")

if __name__ == "__main__":
    main()

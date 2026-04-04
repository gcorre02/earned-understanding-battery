#!/usr/bin/env python3
"""Reproduce v3 data package headline tables from seed + config.

Runs a minimal subset of the harmonised protocol to verify:
1. Null distribution p95 floors match reported values
2. Positive control signals match reported values
3. ROC AUC matches reported value

Usage:
    pip install -e .
    python scripts/reproduce_v3_tables.py

Expected runtime: ~5-10 minutes (graph walkers only, no LLM systems).
For full 13-system reproduction, use scripts/run_harmonised_generativity.py.
"""

import json
import sys
import time

import numpy as np

from earned_understanding_battery.domains.sbm_generator import generate_domain_family, generate_domain
from earned_understanding_battery.domains.presets import MEDIUM
from earned_understanding_battery.core.types import DomainConfig
from earned_understanding_battery.instruments.generativity import (
    _compute_transition_matrix,
    _js_divergence,
    _transition_jsd,
)
from earned_understanding_battery.instruments.role_utils import classify_all_nodes, compute_role_transition_matrix

import networkx as nx

def _log(msg):
    print(f"[reproduce] {msg}", file=sys.stderr, flush=True)

N_STEPS = 500
SEEDS = [42, 123, 456]

def get_community_map(graph):
    return {
        node: graph.nodes[node].get("features", {}).get(
            "community", graph.nodes[node].get("block", 0)
        )
        for node in graph.nodes()
    }

def navigate_and_record(system, graph, n_steps):
    """Navigate frozen system on graph, return visit sequence."""
    try:
        system.set_domain(graph)
    except (AttributeError, NotImplementedError):
        pass
    system.set_training(False)
    system.reset_engagement_tracking()
    seq = []
    for _ in range(n_steps):
        out = system.step(None)
        if isinstance(out, (int, float, str, np.integer)):
            seq.append(int(out))
    return seq

def compute_transition_jsd(seq_a, seq_b, graph):
    """Compute transition JSD between two visit sequences."""
    n2c = get_community_map(graph)
    nc = len(set(n2c.values()))
    if nc == 0 or len(seq_a) < 2 or len(seq_b) < 2:
        return 0.0
    T_a = _compute_transition_matrix(seq_a, n2c, nc)
    T_b = _compute_transition_matrix(seq_b, n2c, nc)
    return _transition_jsd(T_a, T_b)

def main():
    _log("=== Reproduction Script for v3 Data Package ===")
    _log(f"SBM config: MEDIUM (150 nodes, 6 communities)")
    _log(f"Steps: {N_STEPS}, Seeds: {SEEDS}")
    _log("")

    # --- Domain construction ---
    _log("Generating domains...")
    family = generate_domain_family(MEDIUM)
    domain_a = family["A"]
    domain_b1 = family["B"]

    rng = np.random.default_rng(42)
    config_b2 = DomainConfig(
        n_nodes=MEDIUM.n_nodes,
        n_communities=MEDIUM.n_communities,
        p_within=MEDIUM.p_within,
        p_between=MEDIUM.p_between,
        seed=int(rng.integers(10000, 99999)),
    )
    domain_b2 = generate_domain(config_b2)
    domain_b2 = nx.relabel_nodes(domain_b2, {n: n + 1149 for n in domain_b2.nodes()})

    # --- Import systems ---
    from earned_understanding_battery.systems.positive_controls.role_based_walker import RoleBasedWalker
    from earned_understanding_battery.systems.positive_controls.gnn_navigator import GNNNavigator
    from earned_understanding_battery.systems.internal.hebbian_walker import HebbianWalker
    from earned_understanding_battery.systems.class3.empowerment_agent import EmpowermentAgent
    from earned_understanding_battery.systems.class3.active_inference_agent import ActiveInferenceAgent

    # --- Table 1: Null distribution (50 pairs, PC1 type, B₂) ---
    _log("Computing null distribution (50 pairs, B₂)...")
    null_jsds = []
    for i in range(50):
        sys_a = RoleBasedWalker(domain_a, seed=i)
        sys_b = RoleBasedWalker(domain_a, seed=i + 1000)
        seq_a = navigate_and_record(sys_a, domain_b2, N_STEPS)
        seq_b = navigate_and_record(sys_b, domain_b2, N_STEPS)
        t_jsd = compute_transition_jsd(seq_a, seq_b, domain_b2)
        null_jsds.append(t_jsd)

    null_jsds = np.array(null_jsds)
    p95 = np.percentile(null_jsds, 95)

    # Bootstrap CI
    boot_p95s = [np.percentile(np.random.choice(null_jsds, len(null_jsds), replace=True), 95)
                 for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boot_p95s, [2.5, 97.5])

    _log(f"  Null p95: {p95:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"\n=== TABLE 1: Null Distribution (PC1 type, B₂) ===")
    print(f"  N pairs: 50")
    print(f"  Mean:    {null_jsds.mean():.4f}")
    print(f"  p95:     {p95:.4f}")
    print(f"  95% CI:  [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Max:     {null_jsds.max():.4f}")

    # --- Table 2: Positive controls on B₂ ---
    print(f"\n=== TABLE 2: Positive Controls on B₂ ===")
    print(f"{'System':<8} {'Seed':<6} {'t_jsd':<10} {'Floor':<10} {'Above?':<8}")
    print("-" * 42)

    pos_scores = []
    for seed in SEEDS:
        # PC1
        pc1 = RoleBasedWalker(domain_a, seed=seed, eta=0.8, temperature=0.01)
        pc1.train_on_domain(domain_a, n_steps=10000)
        pc1_seq = navigate_and_record(pc1, domain_b2, N_STEPS)

        fresh = RoleBasedWalker(domain_a, seed=seed + 1000)
        fresh_seq = navigate_and_record(fresh, domain_b2, N_STEPS)

        t_jsd = compute_transition_jsd(pc1_seq, fresh_seq, domain_b2)
        above = "Yes" if t_jsd > p95 else "No"
        print(f"{'PC1':<8} {seed:<6} {t_jsd:<10.4f} {p95:<10.4f} {above:<8}")
        pos_scores.append(t_jsd)

    for seed in SEEDS:
        # PC3
        pc3 = GNNNavigator(domain_a, seed=seed, train_epochs=500, temperature=0.05)
        pc3.train_on_domain(domain_a, n_steps=2000)
        pc3_seq = navigate_and_record(pc3, domain_b2, N_STEPS)

        fresh = GNNNavigator(domain_a, seed=seed + 1000)
        fresh_seq = navigate_and_record(fresh, domain_b2, N_STEPS)

        t_jsd = compute_transition_jsd(pc3_seq, fresh_seq, domain_b2)
        above = "Yes" if t_jsd > p95 else "No"
        print(f"{'PC3':<8} {seed:<6} {t_jsd:<10.4f} {p95:<10.4f} {above:<8}")
        pos_scores.append(t_jsd)

    # --- Table 3: Calibration systems on B₂ ---
    print(f"\n=== TABLE 3: Calibration Systems on B₂ ===")
    print(f"{'System':<8} {'Seed':<6} {'t_jsd':<10} {'Floor':<10} {'Above?':<8}")
    print("-" * 42)

    neg_scores = []
    cal_systems = {
        "HEB": lambda s: HebbianWalker(domain_a, seed=s, eta=0.1, decay=0.01, temperature=0.5),
        "3D": lambda s: EmpowermentAgent(domain_a, seed=s),
        "3E": lambda s: ActiveInferenceAgent(domain_a, seed=s),
    }
    train_steps = {"HEB": 2000, "3D": 2000, "3E": 2000}

    for name, factory in cal_systems.items():
        for seed in SEEDS:
            sys = factory(seed)
            sys.train_on_domain(domain_a, n_steps=train_steps[name])
            sys_seq = navigate_and_record(sys, domain_b2, N_STEPS)

            fresh = factory(seed + 1000)
            fresh_seq = navigate_and_record(fresh, domain_b2, N_STEPS)

            t_jsd = compute_transition_jsd(sys_seq, fresh_seq, domain_b2)
            above = "Yes" if t_jsd > p95 else "No"
            print(f"{name:<8} {seed:<6} {t_jsd:<10.4f} {p95:<10.4f} {above:<8}")
            neg_scores.append(t_jsd)

    # --- Table 4: ROC/AUC ---
    from sklearn.metrics import roc_auc_score
    y_true = np.array([0] * len(neg_scores) + [1] * len(pos_scores))
    y_scores = np.array(neg_scores + pos_scores)
    auc = roc_auc_score(y_true, y_scores)

    # Bootstrap CI
    aucs = []
    for _ in range(10000):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(set(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
    auc_ci = np.percentile(aucs, [2.5, 97.5])

    print(f"\n=== TABLE 4: ROC/AUC ===")
    print(f"  AUC:      {auc:.4f}")
    print(f"  95% CI:   [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
    print(f"  Positive: {len(pos_scores)} runs (PC1 + PC3)")
    print(f"  Negative: {len(neg_scores)} runs (HEB + 3D + 3E)")

    _log("")
    _log("Reproduction complete. Compare tables above to v3 data package.")

if __name__ == "__main__":
    main()

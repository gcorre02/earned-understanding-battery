#!/usr/bin/env python3
"""
Behavioural generativity for graph-native systems (1A-1C, 2A-2B, 3A-3B).

Runs at SMALL and MEDIUM, 3 seeds each.
If 3A or 3B show earned at SMALL, extends to full scale ladder.

Usage: .venv/bin/python scripts/run_graph_native_behavioural.py
"""

import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from math import log2

import numpy as np

from m8_battery.analysis.behavioural_generativity import (
    generate_paired_domains,
    record_behaviour,
    compute_behavioural_divergence,
    classify_divergence,
    BehaviourTrace,
)
from m8_battery.systems.class1.wordnet_graph import WordNetGraph
from m8_battery.systems.class1.rule_navigator import RuleBasedNavigator
from m8_battery.systems.class1.foxworthy_a import FoxworthyA
from m8_battery.systems.class2.frozen_llm import FrozenLLM
from m8_battery.systems.class2.frozen_gnn import FrozenGAT
from m8_battery.systems.class3.dqn_agent import DQNAgent
from m8_battery.systems.class3.curiosity_agent import CuriosityAgent


SCALES_BASE = [
    ("SMALL", 50, 4),
    ("MEDIUM", 150, 6),
]

SCALES_EXTENDED = [
    ("SMALL", 50, 4),
    ("S-M1", 75, 5),
    ("S-M2", 100, 5),
    ("S-M3", 125, 6),
    ("MEDIUM", 150, 6),
]

SEEDS = [42, 123, 456]
N_STEPS = 50
P_INTRA = 0.3
P_INTER = 0.01


def visit_entropy(trace: BehaviourTrace) -> float:
    if not trace.visit_distribution:
        return 0.0
    total = sum(trace.visit_distribution.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for count in trace.visit_distribution.values():
        p = count / total
        if p > 0:
            ent -= p * log2(p)
    return ent


def make_system(system_id, graph, seed):
    """Create a system instance attached to a graph."""
    if system_id == "1A":
        return WordNetGraph(graph, seed=seed)
    elif system_id == "1B":
        return RuleBasedNavigator(graph, seed=seed)
    elif system_id == "1C":
        sys = FoxworthyA(seed=seed)
        sys.set_graph(graph)
        return sys
    elif system_id == "2A":
        sys = FrozenLLM(seed=seed)
        sys.set_graph(graph)
        return sys
    elif system_id == "2B":
        sys = FrozenGAT(seed=seed)
        return sys
    elif system_id == "3A":
        sys = DQNAgent(seed=seed)
        return sys
    elif system_id == "3B":
        sys = CuriosityAgent(seed=seed)
        return sys
    else:
        raise ValueError(f"Unknown system: {system_id}")


def run_single(system_id, scale_name, n_nodes, n_communities, seed):
    """Run a single behavioural generativity test."""
    t0 = time.time()

    domain_a, domain_b = generate_paired_domains(
        n_nodes, n_communities, p_within=P_INTRA, p_between=P_INTER
    )

    # Systems that don't learn: fresh and trained are both just "on the graph"
    # Systems that learn (3A, 3B): train on domain_a first

    if system_id in ("1A", "1B"):
        # Graph in constructor, no learning. Both get domain_b with same seed.
        fresh = make_system(system_id, domain_b, seed)
        trained = make_system(system_id, domain_b, seed)
    elif system_id == "1C":
        fresh = make_system(system_id, domain_b, seed)
        trained = make_system(system_id, domain_a, seed)
        # "Train" on A (just operate)
        nodes_a = list(domain_a.nodes())
        for i in range(50):
            trained.step(nodes_a[i % len(nodes_a)])
        trained.set_graph(domain_b)
    elif system_id == "2A":
        fresh = make_system(system_id, domain_b, seed)
        trained = make_system(system_id, domain_a, seed)
        # Frozen LLM — operate on A (no learning), switch to B
        nodes_a = list(domain_a.nodes())
        for i in range(50):
            trained.step(nodes_a[i % len(nodes_a)])
        trained.set_graph(domain_b)
    elif system_id == "2B":
        # FrozenGAT: train_on_domain trains + freezes. No set_graph method.
        # Fresh: train on B (learns B's structure)
        fresh = FrozenGAT(seed=seed)
        fresh.train_on_domain(domain_b)
        # Trained: train on A (learns A's structure), then swap graph to B
        trained = FrozenGAT(seed=seed)
        trained.train_on_domain(domain_a)
        # Manually switch graph while keeping A-trained frozen weights
        trained._graph = domain_b
        trained._pyg_data = trained._graph_to_pyg(domain_b)
        trained.reset()
    elif system_id in ("3A", "3B"):
        # RL agents: fresh uses random walk (no training), trained uses learned policy
        # Fresh: create on domain_b but don't train — step() falls back to random walk
        if system_id == "3A":
            fresh = DQNAgent(seed=seed)
            fresh._graph = domain_b
            trained = DQNAgent(seed=seed)
            trained.train_on_domain(domain_a)
            trained.set_graph(domain_b)
            # Recreate env for domain_b so observations match
            from m8_battery.environments.graph_navigation import GraphNavigationEnv
            trained._env = GraphNavigationEnv(
                graph=domain_b, n_features=trained._n_features,
                max_degree=trained._max_degree, reward_mode="target",
                max_steps=100,
            )
        else:
            fresh = CuriosityAgent(seed=seed)
            fresh._graph = domain_b
            trained = CuriosityAgent(seed=seed)
            trained.train_on_domain(domain_a)
            trained.set_graph(domain_b)
            # Recreate env for domain_b
            from m8_battery.environments.graph_navigation import GraphNavigationEnv
            trained._env = GraphNavigationEnv(
                graph=domain_b, n_features=8,
                max_degree=20, reward_mode="curiosity",
                max_steps=100,
            )
    else:
        raise ValueError(f"Unknown system: {system_id}")

    fresh_trace = record_behaviour(fresh, domain_b, n_steps=N_STEPS)
    trained_trace = record_behaviour(trained, domain_b, n_steps=N_STEPS)
    div = compute_behavioural_divergence(fresh_trace, trained_trace)
    cls = classify_divergence(div)
    elapsed = time.time() - t0

    fresh_nodes = set(fresh_trace.nodes_visited)
    trained_nodes = set(trained_trace.nodes_visited)

    return {
        "system_id": system_id,
        "scale": scale_name,
        "n_nodes": n_nodes,
        "n_communities": n_communities,
        "seed": seed,
        "divergence": asdict(div),
        "classification": cls,
        "time_s": round(elapsed, 1),
        "diagnostics": {
            "fresh_unique_nodes": fresh_trace.unique_nodes,
            "trained_unique_nodes": trained_trace.unique_nodes,
            "node_overlap": len(fresh_nodes & trained_nodes),
            "fresh_visit_entropy": round(visit_entropy(fresh_trace), 4),
            "trained_visit_entropy": round(visit_entropy(trained_trace), 4),
        },
    }


def main():
    all_results = []
    systems = ["1A", "1B", "1C", "2A", "2B", "3A", "3B"]
    extend_systems = set()  # Systems that need extended scale ladder

    # Phase 1: Base runs (SMALL + MEDIUM)
    total = len(systems) * len(SCALES_BASE) * len(SEEDS)
    print(f"Phase 1: {total} base runs (SMALL + MEDIUM)")
    print("=" * 70)

    run_count = 0
    for system_id in systems:
        for scale_name, n_nodes, n_comm in SCALES_BASE:
            for seed in SEEDS:
                run_count += 1
                try:
                    result = run_single(system_id, scale_name, n_nodes, n_comm, seed)
                    all_results.append(result)
                    d = result["divergence"]
                    diag = result["diagnostics"]

                    status = result["classification"]
                    # Fail-fast: Class 1/2 should NOT be earned
                    if system_id in ("1A", "1B", "1C", "2A", "2B") and status == "earned":
                        print(f"  ⚠ FAIL-FAST: {system_id} classified as earned — INVESTIGATE")

                    # Check if 3A/3B need extension
                    if system_id in ("3A", "3B") and scale_name == "SMALL" and status == "earned":
                        extend_systems.add(system_id)

                    print(
                        f"[{run_count}/{total}] {system_id} {scale_name:6s} "
                        f"n={n_nodes:3d} s{seed:3d}: "
                        f"JS={d['visit_js_divergence']:.4f} "
                        f"EdgeJ={d['edge_jaccard']:.4f} "
                        f"uniq={diag['fresh_unique_nodes']}/{diag['trained_unique_nodes']} "
                        f"→ {status} ({result['time_s']:.1f}s)"
                    )
                except Exception as e:
                    print(f"[{run_count}/{total}] {system_id} {scale_name} s{seed}: ERROR — {e}")
                    import traceback
                    traceback.print_exc()
                sys.stdout.flush()

    # Phase 2: Extended runs for 3A/3B if earned at SMALL
    if extend_systems:
        extra_scales = [s for s in SCALES_EXTENDED if s[0] not in ("SMALL", "MEDIUM")]
        extra_runs = len(extend_systems) * len(extra_scales) * len(SEEDS)
        print(f"\nPhase 2: Extending {extend_systems} — {extra_runs} additional runs")
        print("=" * 70)

        for system_id in sorted(extend_systems):
            for scale_name, n_nodes, n_comm in extra_scales:
                for seed in SEEDS:
                    try:
                        result = run_single(system_id, scale_name, n_nodes, n_comm, seed)
                        all_results.append(result)
                        d = result["divergence"]
                        print(
                            f"  {system_id} {scale_name:6s} n={n_nodes:3d} s{seed:3d}: "
                            f"JS={d['visit_js_divergence']:.4f} → {result['classification']}"
                        )
                    except Exception as e:
                        print(f"  {system_id} {scale_name} s{seed}: ERROR — {e}")
                    sys.stdout.flush()

    # Save
    outpath = "results/graph_native_behavioural_2026-03-16.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'System':6s} {'Scale':8s} {'N':>4s} {'Mean JS':>8s} {'Classification':>15s}")
    print("-" * 50)

    for system_id in systems:
        sys_results = [r for r in all_results if r["system_id"] == system_id]
        scales_seen = sorted(set(r["scale"] for r in sys_results),
                            key=lambda s: next(x[1] for x in SCALES_BASE + SCALES_EXTENDED if x[0] == s))
        for scale_name in scales_seen:
            runs = [r for r in sys_results if r["scale"] == scale_name]
            js_vals = [r["divergence"]["visit_js_divergence"] for r in runs]
            mean_js = np.mean(js_vals)
            classes = [r["classification"] for r in runs]
            majority = max(set(classes), key=classes.count)
            n_nodes = runs[0]["n_nodes"]
            print(f"{system_id:6s} {scale_name:8s} {n_nodes:4d} {mean_js:8.4f} {majority:>15s}")


if __name__ == "__main__":
    main()

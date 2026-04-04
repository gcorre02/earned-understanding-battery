#!/usr/bin/env python3
"""
Multi-scale behavioural generativity sweep.

Runs 3C (FoxworthyF) and 2C (FoxworthyC) across an 8-point scale ladder.
Records supplementary diagnostics: unique nodes, visit entropy, top-5 attractors.

Usage: .venv/bin/python scripts/run_multiscale_behavioural.py
"""

import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from math import log2

import numpy as np

from earned_understanding_battery.analysis.behavioural_generativity import (
    generate_paired_domains,
    record_behaviour,
    compute_behavioural_divergence,
    classify_divergence,
    BehaviourTrace,
)
from earned_understanding_battery.systems.class3.foxworthy_f import FoxworthyF
from earned_understanding_battery.systems.class2.foxworthy_c import FoxworthyC

# Scale ladder
SCALES = [
    ("XS",   30,  3),
    ("SMALL", 50,  4),
    ("S-M1", 75,  5),
    ("S-M2", 100, 5),
    ("S-M3", 125, 6),
    ("MEDIUM", 150, 6),
    ("LARGE", 250, 8),
    ("XL",   500, 10),
]

SEEDS_FULL = [42, 123, 456]
SEEDS_XL = [42]

N_WARMUP = 50
N_STEPS = 50
P_INTRA = 0.3
P_INTER = 0.01

def visit_entropy(trace: BehaviourTrace) -> float:
    """Shannon entropy of visit distribution (bits)."""
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

def top_n_nodes(trace: BehaviourTrace, n: int = 5) -> list[tuple]:
    """Top N most visited nodes."""
    return Counter(trace.nodes_visited).most_common(n)

def run_single(system_id, scale_name, n_nodes, n_communities, seed, p_intra, p_inter):
    """Run a single behavioural generativity test with diagnostics."""
    t0 = time.time()

    domain_a, domain_b = generate_paired_domains(
        n_nodes, n_communities, p_within=p_intra, p_between=p_inter
    )

    if system_id == "3C":
        fresh = FoxworthyF(seed=seed, device="cpu", theta=0.0)
        fresh.train_on_domain(domain_b, n_warmup=0)
        trained = FoxworthyF(seed=seed, device="cpu", theta=0.0)
        trained.train_on_domain(domain_a, n_warmup=N_WARMUP)
        trained.set_graph(domain_b)
    elif system_id == "2C":
        fresh = FoxworthyC(seed=seed)
        fresh.set_graph(domain_b)
        trained = FoxworthyC(seed=seed)
        trained.set_graph(domain_a)
        nodes_a = list(domain_a.nodes())
        for i in range(N_WARMUP):
            trained.step(nodes_a[i % len(nodes_a)])
        trained.set_graph(domain_b)
    else:
        raise ValueError(f"Unknown system: {system_id}")

    fresh_trace = record_behaviour(fresh, domain_b, n_steps=N_STEPS)
    trained_trace = record_behaviour(trained, domain_b, n_steps=N_STEPS)
    div = compute_behavioural_divergence(fresh_trace, trained_trace)
    cls = classify_divergence(div)
    elapsed = time.time() - t0

    # Node overlap
    fresh_nodes = set(fresh_trace.nodes_visited)
    trained_nodes = set(trained_trace.nodes_visited)
    overlap = len(fresh_nodes & trained_nodes)

    result = {
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
            "node_overlap": overlap,
            "fresh_visit_entropy": round(visit_entropy(fresh_trace), 4),
            "trained_visit_entropy": round(visit_entropy(trained_trace), 4),
            "fresh_top5": [(str(n), c) for n, c in top_n_nodes(fresh_trace, 5)],
            "trained_top5": [(str(n), c) for n, c in top_n_nodes(trained_trace, 5)],
            "fresh_communities": sorted(fresh_trace.community_coverage.keys()),
            "trained_communities": sorted(trained_trace.community_coverage.keys()),
        },
    }
    return result

def main():
    all_results = []
    systems = ["3C", "2C"]

    total_runs = 0
    for system_id in systems:
        for scale_name, n_nodes, n_comm in SCALES:
            seeds = SEEDS_XL if scale_name == "XL" else SEEDS_FULL
            total_runs += len(seeds)

    print(f"Multi-scale behavioural generativity sweep: {total_runs} runs")
    print(f"Systems: {systems}")
    print(f"Scales: {[s[0] for s in SCALES]}")
    print(f"p_intra={P_INTRA}, p_inter={P_INTER}")
    print("=" * 70)

    run_count = 0
    for system_id in systems:
        print(f"\n{'='*70}")
        print(f"SYSTEM: {system_id}")
        print(f"{'='*70}")

        for scale_name, n_nodes, n_comm in SCALES:
            seeds = SEEDS_XL if scale_name == "XL" else SEEDS_FULL

            for seed in seeds:
                run_count += 1
                sys.stdout.flush()
                try:
                    result = run_single(
                        system_id, scale_name, n_nodes, n_comm, seed,
                        P_INTRA, P_INTER,
                    )
                    all_results.append(result)
                    d = result["divergence"]
                    diag = result["diagnostics"]
                    print(
                        f"[{run_count}/{total_runs}] {system_id} {scale_name:6s} "
                        f"n={n_nodes:3d} s{seed:3d}: "
                        f"JS={d['visit_js_divergence']:.4f} "
                        f"EdgeJ={d['edge_jaccard']:.4f} "
                        f"fresh_uniq={diag['fresh_unique_nodes']:2d} "
                        f"trained_uniq={diag['trained_unique_nodes']:2d} "
                        f"overlap={diag['node_overlap']:2d} "
                        f"→ {result['classification']} "
                        f"({result['time_s']:.1f}s)"
                    )
                except Exception as e:
                    print(f"[{run_count}/{total_runs}] {system_id} {scale_name} s{seed}: ERROR — {e}")
                    import traceback
                    traceback.print_exc()

                sys.stdout.flush()

    # Save results
    outpath = "results/multiscale_behavioural_3c_2c_2026-03-16.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY — Mean JS divergence by system × scale")
    print("=" * 70)
    print(f"{'System':6s} {'Scale':8s} {'N':>4s} {'Mean JS':>8s} {'Std JS':>8s} {'Classification':>15s}")
    print("-" * 55)

    for system_id in systems:
        for scale_name, n_nodes, n_comm in SCALES:
            runs = [r for r in all_results
                    if r["system_id"] == system_id and r["scale"] == scale_name]
            if not runs:
                continue
            js_vals = [r["divergence"]["visit_js_divergence"] for r in runs]
            mean_js = np.mean(js_vals)
            std_js = np.std(js_vals) if len(js_vals) > 1 else 0.0
            # Majority classification
            classes = [r["classification"] for r in runs]
            majority = max(set(classes), key=classes.count)
            print(
                f"{system_id:6s} {scale_name:8s} {n_nodes:4d} "
                f"{mean_js:8.4f} {std_js:8.4f} {majority:>15s}"
            )

    print(f"\nTotal time: {sum(r['time_s'] for r in all_results):.0f}s")

if __name__ == "__main__":
    main()

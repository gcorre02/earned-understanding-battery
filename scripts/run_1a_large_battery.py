#!/usr/bin/env python3
"""
1A LARGE full battery — Phase B blocker.

Runs WordNetGraph at 500 nodes (LARGE preset) with full structural battery
(5 instruments) + behavioural generativity, for 3 seeds.

Usage: .venv/bin/python scripts/run_1a_large_battery.py
"""

import json
import sys
import time

from m8_battery.core.types import SystemClass
from m8_battery.domains.sbm_generator import generate_domain_family
from m8_battery.domains.presets import LARGE
from m8_battery.instruments.battery_runner import run_battery, BatteryConfig
from m8_battery.systems.class1.wordnet_graph import WordNetGraph
from m8_battery.analysis.behavioural_generativity import (
    generate_paired_domains,
    record_behaviour,
    compute_behavioural_divergence,
    classify_divergence,
)

SEEDS = [42, 123, 456]


def run_structural_battery(seed):
    """Run full 5-instrument structural battery on 1A LARGE."""
    t0 = time.time()
    family = generate_domain_family(LARGE._replace(seed=seed) if hasattr(LARGE, '_replace') else LARGE)

    # Generate with the specific seed
    from m8_battery.core.types import DomainConfig
    config = DomainConfig(
        n_nodes=500, n_communities=12,
        p_within=0.25, p_between=0.015,
        seed=seed,
    )
    from m8_battery.domains.sbm_generator import generate_domain_family as gen_family
    family = gen_family(config)

    G = family["A"]
    system = WordNetGraph(G, seed=seed)
    nodes_a = list(G.nodes())

    battery_config = BatteryConfig(
        domain_a_inputs=nodes_a[:50],
        domain_a_prime_inputs=list(family["A_prime"].nodes())[:25],
        domain_b_inputs=list(family["B"].nodes())[:25],
        measurement_interval=5,
        wander_steps=15,
        recovery_window=15,
    )

    result = run_battery(
        system=system,
        system_name="WordNet Static Graph",
        system_class=SystemClass.CLASS_1,
        config=battery_config,
        control_factory=lambda: WordNetGraph(G, seed=seed + 1000),
    )

    elapsed = time.time() - t0

    # Extract key results
    instruments = {}
    for name, r in result.instrument_results.items():
        instruments[name] = {
            "passed": r.passed,
            "effect_size": round(r.effect_size, 4) if r.effect_size else 0.0,
        }

    baseline = result.metadata.get("baseline", {})
    classifications = baseline.get("instrument_classifications", {})

    return {
        "system_id": "1A",
        "scale": "LARGE",
        "seed": seed,
        "n_nodes": 500,
        "overall_passed": result.overall_passed,
        "provenance_passed": result.provenance_passed,
        "instruments": instruments,
        "classifications": classifications,
        "battery_time_s": round(elapsed, 1),
        "trajectory_training": baseline.get("trajectory_training", "unknown"),
        "trajectory_battery": baseline.get("trajectory_battery", "unknown"),
    }


def run_behavioural(seed):
    """Run behavioural generativity on 1A LARGE."""
    t0 = time.time()
    domain_a, domain_b = generate_paired_domains(500, n_communities=12, p_within=0.25, p_between=0.015)

    fresh = WordNetGraph(domain_b, seed=seed)
    trained = WordNetGraph(domain_b, seed=seed)  # Same — no learning

    fresh_trace = record_behaviour(fresh, domain_b, n_steps=50)
    trained_trace = record_behaviour(trained, domain_b, n_steps=50)
    div = compute_behavioural_divergence(fresh_trace, trained_trace)
    cls = classify_divergence(div)

    return {
        "system_id": "1A",
        "scale": "LARGE",
        "seed": seed,
        "behavioural_js": round(div.visit_js_divergence, 6),
        "behavioural_edge_jaccard": round(div.edge_jaccard, 6),
        "behavioural_classification": cls,
        "time_s": round(time.time() - t0, 1),
    }


def main():
    all_results = []

    print("1A LARGE Full Battery — 3 seeds")
    print("=" * 70)

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # Structural battery
        print(f"  Running structural battery (500 nodes, 12 communities)...")
        sys.stdout.flush()
        try:
            struct = run_structural_battery(seed)
            all_results.append({"type": "structural", **struct})
            print(f"  Structural: overall={struct['overall_passed']}, "
                  f"provenance={struct['provenance_passed']}, "
                  f"time={struct['battery_time_s']}s")
            for name, r in struct["instruments"].items():
                cls = struct["classifications"].get(name, "?")
                print(f"    {name}: passed={r['passed']}, ES={r['effect_size']}, cls={cls}")
        except Exception as e:
            print(f"  Structural ERROR: {e}")
            import traceback
            traceback.print_exc()

        # Behavioural generativity
        print(f"  Running behavioural generativity...")
        sys.stdout.flush()
        try:
            behav = run_behavioural(seed)
            all_results.append({"type": "behavioural", **behav})
            print(f"  Behavioural: JS={behav['behavioural_js']}, "
                  f"EdgeJ={behav['behavioural_edge_jaccard']}, "
                  f"cls={behav['behavioural_classification']}, "
                  f"time={behav['time_s']}s")
        except Exception as e:
            print(f"  Behavioural ERROR: {e}")
            import traceback
            traceback.print_exc()

        sys.stdout.flush()

    outpath = "results/1a_large_battery_2026-03-16.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()

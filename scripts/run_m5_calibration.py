"""M5 Max calibration runner — runs 2A (TinyLlama) and 3C (Foxworthy F) only.

These two systems require >16GB RAM due to model copies during battery
instruments (ablation, perturbation, transfer control). The M5 Max with
36GB unified memory handles this comfortably.

Usage:
    .venv/bin/python scripts/run_m5_calibration.py

Outputs:
    results/calibration/*.json — individual BatteryResult files
    results/calibration/summary_m5.json — summary with all results
    results/calibration/m5_complete.flag — signals completion to Razer

After completion, copy results back to Razer via:
    - The knowledge_base_rag comms folder, OR
    - Direct file copy / git push
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earned_understanding_battery.core.types import DomainConfig, SystemClass
from earned_understanding_battery.domains.sbm_generator import generate_domain_family
from earned_understanding_battery.domains.presets import SMALL, MEDIUM, LARGE
from earned_understanding_battery.instruments.battery_runner import run_battery, BatteryConfig

PRESETS = {"small": SMALL, "medium": MEDIUM, "large": LARGE}

RESULTS_DIR = Path(__file__).parent.parent / "results" / "calibration"

SYSTEM_NAMES = {
    "2A": "Frozen TinyLlama 1.1B",
    "3C": "Foxworthy Variant F",
}

SYSTEM_CLASSES = {
    "2A": SystemClass.CLASS_2,
    "3C": SystemClass.CLASS_3,
}

SEEDS = [42, 123, 456]
SYSTEMS = ["2A", "3C"]

def _count_communities(graph) -> int:
    return len({graph.nodes[n].get("features", {}).get("community", 0)
                for n in graph.nodes()})

def _get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def make_system(system_id: str, graph, seed: int, n_features: int):
    """Instantiate system. Uses MPS on Apple Silicon, CUDA if available, else CPU."""
    device = _get_device()
    print(f"  Using device: {device}")

    if system_id == "2A":
        from earned_understanding_battery.systems.class2.frozen_llm import FrozenLLM
        system = FrozenLLM(seed=seed, device=device)
        system.set_graph(graph)
        return system, lambda: _make_llm_control(seed + 1000, graph)

    elif system_id == "3C":
        from earned_understanding_battery.systems.class3.foxworthy_f import FoxworthyF
        system = FoxworthyF(seed=seed, device=device, theta=0.0)
        system.train_on_domain(graph, n_warmup=50)
        return system, lambda: _make_foxworthy_control(seed + 1000, graph)

    else:
        raise ValueError(f"This script only runs 2A and 3C, got: {system_id}")

def _make_llm_control(seed, graph):
    from earned_understanding_battery.systems.class2.frozen_llm import FrozenLLM
    s = FrozenLLM(seed=seed, device=_get_device())
    s.set_graph(graph)
    return s

def _make_foxworthy_control(seed, graph):
    from earned_understanding_battery.systems.class3.foxworthy_f import FoxworthyF
    s = FoxworthyF(seed=seed, device=_get_device(), theta=0.0)
    s.set_graph(graph)
    return s

def run_single(system_id: str, seed: int, preset=None, scale_name: str = "medium") -> dict:
    """Run full battery on one system with one seed."""
    if preset is None:
        preset = MEDIUM

    config = DomainConfig(
        n_nodes=preset.n_nodes,
        n_communities=preset.n_communities,
        p_within=preset.p_within,
        p_between=preset.p_between,
        n_edge_types=preset.n_edge_types,
        weight_range=preset.weight_range,
        n_node_features=preset.n_node_features,
        seed=seed,
    )
    family = generate_domain_family(config)
    graph_a = family["A"]
    nodes_a = list(graph_a.nodes())
    nodes_a_prime = list(family["A_prime"].nodes())
    nodes_b = list(family["B"].nodes())

    n_features = config.n_node_features

    print(f"  Instantiating {system_id} ({SYSTEM_NAMES[system_id]})...")
    t0 = time.time()
    system, control_factory = make_system(system_id, graph_a, seed, n_features)
    t_setup = time.time() - t0
    print(f"  Setup: {t_setup:.1f}s")

    # Reduced inputs for LLM systems (CPU inference is slow)
    n_inputs = 20
    battery_config = BatteryConfig(
        domain_a_inputs=nodes_a[:n_inputs],
        domain_a_prime_inputs=nodes_a_prime[:n_inputs],
        domain_b_inputs=nodes_b[:n_inputs],
        probe_inputs=nodes_a[:10],
        measurement_interval=5,
        wander_steps=15,
        recovery_window=15,
    )

    print(f"  Running battery ({n_inputs} inputs per instrument)...")
    t0 = time.time()
    result = run_battery(
        system=system,
        system_name=SYSTEM_NAMES[system_id],
        system_class=SYSTEM_CLASSES[system_id],
        config=battery_config,
        control_factory=control_factory,
    )
    t_battery = time.time() - t0
    print(f"  Battery: {t_battery:.1f}s — overall={result.overall_passed}")

    # Unload to free memory before next run
    if hasattr(system, 'unload_model'):
        system.unload_model()

    # Serialise
    result_dict = {
        "system_id": system_id,
        "system_name": SYSTEM_NAMES[system_id],
        "system_class": SYSTEM_CLASSES[system_id].value,
        "scale": scale_name,
        "seed": seed,
        "n_nodes": preset.n_nodes,
        "overall_passed": result.overall_passed,
        "provenance_passed": result.provenance_passed,
        "setup_time_s": t_setup,
        "battery_time_s": t_battery,
        "run_machine": "m5_max",
        "instruments": {},
        "metadata": {},
    }

    for name, ir in result.instrument_results.items():
        result_dict["instruments"][name] = {
            "passed": ir.passed,
            "effect_size": ir.effect_size,
            "notes": ir.notes,
        }

    if result.metadata:
        for k, v in result.metadata.items():
            try:
                json.dumps(v)
                result_dict["metadata"][k] = v
            except (TypeError, ValueError):
                result_dict["metadata"][k] = str(v)

    return result_dict

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="medium", choices=["small", "medium", "large"])
    args = parser.parse_args()

    scale_name = args.scale
    preset = PRESETS[scale_name]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Calibration — M5 Max (2A + 3C only) ===")
    print(f"Systems: {SYSTEMS}")
    print(f"Seeds: {SEEDS}")
    print(f"Scale: {scale_name.upper()} ({preset.n_nodes} nodes)")
    print()

    all_results = []
    total_t0 = time.time()

    for system_id in SYSTEMS:
        for seed in SEEDS:
            print(f"[{system_id}] seed={seed}, scale={scale_name}")
            try:
                result = run_single(system_id, seed, preset=preset, scale_name=scale_name)
                all_results.append(result)

                fname = f"{system_id}_{scale_name}_{seed}.json"
                (RESULTS_DIR / fname).write_text(
                    json.dumps(result, indent=2, default=str)
                )
                print(f"  Saved: {fname}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results.append({
                    "system_id": system_id, "scale": scale_name, "seed": seed,
                    "error": str(e), "run_machine": "m5_max",
                })
            print()

    total_time = time.time() - total_t0

    # Summary
    summary = {
        "run_machine": "m5_max",
        "scale": scale_name,
        "n_nodes": preset.n_nodes,
        "seeds": SEEDS,
        "systems": SYSTEMS,
        "total_time_s": total_time,
        "results": all_results,
    }
    summary_path = RESULTS_DIR / f"summary_m5_{scale_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    # Completion flag
    flag_path = RESULTS_DIR / "m5_complete.flag"
    flag_path.write_text(json.dumps({
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_s": total_time,
        "n_results": len([r for r in all_results if "error" not in r]),
        "n_errors": len([r for r in all_results if "error" in r]),
    }, indent=2))

    # Print summary
    print(f"=== SUMMARY ({total_time:.0f}s total) ===")
    print(f"{'System':<6} {'Seed':<6} {'Overall':<10} {'Time':<10}")
    print("-" * 40)
    for r in all_results:
        if "error" in r:
            print(f"{r['system_id']:<6} {r['seed']:<6} {'ERROR':<10} {r['error'][:30]}")
        else:
            passed = "PASS" if r["overall_passed"] else "FAIL"
            t = f"{r['battery_time_s']:.1f}s"
            print(f"{r['system_id']:<6} {r['seed']:<6} {passed:<10} {t:<10}")

    print(f"\nResults in: {RESULTS_DIR}")
    print(f"Summary: {summary_path}")
    print(f"Flag: {flag_path}")

if __name__ == "__main__":
    main()

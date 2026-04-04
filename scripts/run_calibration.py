"""Calibration runner — run full battery across all 9 Class 1-3 systems.

Usage: python scripts/run_calibration.py [--scale small|medium|large] [--seeds 42,123,456]

Outputs BatteryResult JSON files to results/calibration/.
"""

from __future__ import annotations

# Force CPU — CUDA segfaults on RTX 3060 Laptop with torch 2.6.0+cu124.
# SB3 also creates cuda tensors that fail numpy conversion.
# TODO: re-enable CUDA when driver resolved.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_default_device("cpu")
# Also monkey-patch cuda availability for SB3
torch.cuda.is_available = lambda: False

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earned_understanding_battery.core.types import DomainConfig, SystemClass
from earned_understanding_battery.domains.sbm_generator import generate_domain_family
from earned_understanding_battery.domains.presets import SMALL, MEDIUM, LARGE
from earned_understanding_battery.instruments.battery_runner import run_battery, BatteryConfig

PRESETS = {"small": SMALL, "medium": MEDIUM, "large": LARGE}
RESULTS_DIR = Path(__file__).parent.parent / "results" / "calibration"

def _get_device() -> str:
    """Auto-detect CUDA. Use GPU for calibration, CPU for tests.

    Note: CUDA 12.4 + RTX 3060 Laptop segfaults during TinyLlama
    inference on this machine. Forced to CPU until driver resolved.
    """
    # TODO: re-enable CUDA when driver issue resolved
    # import torch
    # return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"

def _count_communities(graph) -> int:
    """Count communities in graph."""
    return len({graph.nodes[n].get("features", {}).get("community", 0)
                for n in graph.nodes()})

def make_system(system_id: str, graph, seed: int, n_features: int):
    """Instantiate a system by ID. Returns (system, control_factory)."""
    device = _get_device()
    n_classes = _count_communities(graph)

    if system_id == "1A":
        from earned_understanding_battery.systems.class1.wordnet_graph import WordNetGraph
        system = WordNetGraph(graph, seed=seed)
        return system, lambda: WordNetGraph(graph, seed=seed + 1000)

    elif system_id == "1B":
        from earned_understanding_battery.systems.class1.rule_navigator import RuleBasedNavigator
        system = RuleBasedNavigator(graph, seed=seed)
        return system, lambda: RuleBasedNavigator(graph, seed=seed + 1000)

    elif system_id == "1C":
        from earned_understanding_battery.systems.class1.foxworthy_a import FoxworthyA
        system = FoxworthyA(n_features=n_features, seed=seed)
        system.set_graph(graph)
        return system, lambda: _clone_with_graph(FoxworthyA(n_features=n_features, seed=seed + 1000), graph)

    elif system_id == "2A":
        from earned_understanding_battery.systems.class2.frozen_llm import FrozenLLM
        system = FrozenLLM(seed=seed, device=device)
        system.set_graph(graph)
        return system, lambda: _clone_with_graph(FrozenLLM(seed=seed + 1000, device=device), graph)

    elif system_id == "2B":
        from earned_understanding_battery.systems.class2.frozen_gnn import FrozenGAT
        system = FrozenGAT(n_features=n_features, n_classes=n_classes, seed=seed)
        system.train_on_domain(graph)
        return system, lambda: _train_gat(FrozenGAT(n_features=n_features, n_classes=n_classes, seed=seed + 1000), graph)

    elif system_id == "2C":
        from earned_understanding_battery.systems.class2.foxworthy_c import FoxworthyC
        system = FoxworthyC(n_features=n_features, hidden_dim=32, seed=seed)
        system.set_graph(graph)
        return system, lambda: _clone_with_graph(FoxworthyC(n_features=n_features, hidden_dim=32, seed=seed + 1000), graph)

    elif system_id == "3A":
        from earned_understanding_battery.systems.class3.dqn_agent import DQNAgent
        system = DQNAgent(n_features=n_features, seed=seed, total_timesteps=2000)
        system.train_on_domain(graph)
        return system, lambda: _train_dqn(DQNAgent(n_features=n_features, seed=seed + 1000, total_timesteps=2000), graph)

    elif system_id == "3B":
        from earned_understanding_battery.systems.class3.curiosity_agent import CuriosityAgent
        system = CuriosityAgent(n_features=n_features, seed=seed, total_timesteps=2000)
        system.train_on_domain(graph)
        return system, lambda: _train_curiosity(CuriosityAgent(n_features=n_features, seed=seed + 1000, total_timesteps=2000), graph)

    elif system_id == "3C":
        from earned_understanding_battery.systems.class3.foxworthy_f import FoxworthyF
        system = FoxworthyF(seed=seed, device=device, theta=0.0)
        system.train_on_domain(graph, n_warmup=50)
        return system, lambda: _clone_with_graph(FoxworthyF(seed=seed + 1000, device=device, theta=0.0), graph)

    else:
        raise ValueError(f"Unknown system: {system_id}")

def _clone_with_graph(system, graph):
    system.set_graph(graph)
    return system

def _train_gat(system, graph):
    system.train_on_domain(graph)
    return system

def _train_dqn(system, graph):
    system.train_on_domain(graph)
    return system

def _train_curiosity(system, graph):
    system.train_on_domain(graph)
    return system

SYSTEM_CLASSES = {
    "1A": SystemClass.CLASS_1, "1B": SystemClass.CLASS_1, "1C": SystemClass.CLASS_1,
    "2A": SystemClass.CLASS_2, "2B": SystemClass.CLASS_2, "2C": SystemClass.CLASS_2,
    "3A": SystemClass.CLASS_3, "3B": SystemClass.CLASS_3, "3C": SystemClass.CLASS_3,
}

SYSTEM_NAMES = {
    "1A": "WordNet Static Graph",
    "1B": "Rule-Based Navigator",
    "1C": "Foxworthy Variant A",
    "2A": "Frozen TinyLlama 1.1B",
    "2B": "Frozen GAT",
    "2C": "Foxworthy Variant C",
    "3A": "DQN (MaskablePPO)",
    "3B": "Curiosity (RND)",
    "3C": "Foxworthy Variant F",
}

ALL_SYSTEMS = ["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C"]

def run_single(
    system_id: str, scale: str, seed: int, preset: DomainConfig,
) -> dict:
    """Run full battery on one system at one scale with one seed."""

    # Generate domain family with this seed
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

    # Configure battery — reduce inputs for LLM systems (slow on CPU)
    LLM_SYSTEMS = {"2A", "3C"}
    n_inputs = min(20 if system_id in LLM_SYSTEMS else 50, len(nodes_a))
    battery_config = BatteryConfig(
        domain_a_inputs=nodes_a[:n_inputs],
        domain_a_prime_inputs=nodes_a_prime[:n_inputs],
        domain_b_inputs=nodes_b[:n_inputs],
        probe_inputs=nodes_a[:10],
        measurement_interval=5,
        wander_steps=15,
        recovery_window=15,
    )

    print(f"  Running battery...")
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

    # Unload heavy models to free memory
    if hasattr(system, 'unload_model'):
        system.unload_model()

    # Serialise result
    result_dict = {
        "system_id": system_id,
        "system_name": SYSTEM_NAMES[system_id],
        "system_class": SYSTEM_CLASSES[system_id].value,
        "scale": scale,
        "seed": seed,
        "n_nodes": preset.n_nodes,
        "overall_passed": result.overall_passed,
        "provenance_passed": result.provenance_passed,
        "setup_time_s": t_setup,
        "battery_time_s": t_battery,
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
    parser = argparse.ArgumentParser(description="Earned Understanding Battery Calibration Runner")
    parser.add_argument("--scale", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    args = parser.parse_args()

    scale = args.scale
    seeds = [int(s) for s in args.seeds.split(",")]
    systems = [s.strip() for s in args.systems.split(",")]
    preset = PRESETS[scale]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Calibration: {scale.upper()} ({preset.n_nodes} nodes) ===")
    print(f"Systems: {systems}")
    print(f"Seeds: {seeds}")
    print()

    all_results = []
    total_t0 = time.time()

    for system_id in systems:
        for seed in seeds:
            print(f"[{system_id}] seed={seed}, scale={scale}")
            try:
                result = run_single(system_id, scale, seed, preset)
                all_results.append(result)

                # Save individual result
                fname = f"{system_id}_{scale}_{seed}.json"
                (RESULTS_DIR / fname).write_text(
                    json.dumps(result, indent=2, default=str)
                )
                print(f"  Saved: {fname}")
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({
                    "system_id": system_id, "scale": scale, "seed": seed,
                    "error": str(e),
                })
            print()

    total_time = time.time() - total_t0

    # Save summary
    summary = {
        "scale": scale,
        "n_nodes": preset.n_nodes,
        "seeds": seeds,
        "systems": systems,
        "total_time_s": total_time,
        "results": all_results,
    }
    summary_path = RESULTS_DIR / f"summary_{scale}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    # Print summary table
    print(f"=== SUMMARY ({total_time:.0f}s total) ===")
    print(f"{'System':<6} {'Seed':<6} {'Overall':<10} {'Time':<8}")
    print("-" * 35)
    for r in all_results:
        if "error" in r:
            print(f"{r['system_id']:<6} {r['seed']:<6} {'ERROR':<10}")
        else:
            passed = "PASS" if r["overall_passed"] else "FAIL"
            t = f"{r['battery_time_s']:.1f}s"
            print(f"{r['system_id']:<6} {r['seed']:<6} {passed:<10} {t:<8}")

if __name__ == "__main__":
    main()

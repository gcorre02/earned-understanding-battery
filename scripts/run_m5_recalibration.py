#!/usr/bin/env python3
"""
M5 Max recalibration — all 9 systems × 3 seeds × MEDIUM.

Uses MPS for LLM systems (2A, 3C). CPU for everything else.
SB3 doesn't support MPS — RL systems stay on CPU.

Usage: .venv/bin/python scripts/run_m5_recalibration.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from m8_battery.core.types import DomainConfig, SystemClass
from m8_battery.domains.sbm_generator import generate_domain_family
from m8_battery.instruments.battery_runner import run_battery, BatteryConfig


MEDIUM = DomainConfig(
    n_nodes=150, n_communities=6,
    p_within=0.3, p_between=0.02,
    seed=42,
)

SEEDS = [42, 123, 456]
RESULTS_DIR = Path(__file__).parent.parent / "results" / "recalibration_m5"

SYSTEM_CLASSES = {
    "1A": SystemClass.CLASS_1, "1B": SystemClass.CLASS_1, "1C": SystemClass.CLASS_1,
    "2A": SystemClass.CLASS_2, "2B": SystemClass.CLASS_2, "2C": SystemClass.CLASS_2,
    "3A": SystemClass.CLASS_3, "3B": SystemClass.CLASS_3, "3C": SystemClass.CLASS_3,
    "HEB": SystemClass.CLASS_3,   # Publishable per DN-28
    "STDP": SystemClass.CLASS_3,  # Paper 2 system 4A-anchor
    "3D": SystemClass.CLASS_3,    # Empowerment (Klyubin 2005)
    "3E": SystemClass.CLASS_3,    # Active inference (Friston 2010)
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
    "HEB": "Hebbian Walker (System HEB)",
    "STDP": "Brian2 STDP (4A-anchor)",
    "3D": "Empowerment (Klyubin)",
    "3E": "Active Inference (Friston)",
}

# LLM systems get MPS on M5, everything else CPU
LLM_SYSTEMS = {"2A", "3C"}


def get_device(system_id: str) -> str:
    if system_id in LLM_SYSTEMS and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _count_communities(graph) -> int:
    return len({graph.nodes[n].get("features", {}).get("community", 0)
                for n in graph.nodes()})


def _clone_with_graph(system, graph):
    system.set_graph(graph)
    return system


def make_system(system_id, graph, seed, n_features):
    device = get_device(system_id)
    n_classes = _count_communities(graph)

    if system_id == "1A":
        from m8_battery.systems.class1.wordnet_graph import WordNetGraph
        s = WordNetGraph(graph, seed=seed)
        return s, lambda: WordNetGraph(graph, seed=seed + 1000)

    elif system_id == "1B":
        from m8_battery.systems.class1.rule_navigator import RuleBasedNavigator
        s = RuleBasedNavigator(graph, seed=seed)
        return s, lambda: RuleBasedNavigator(graph, seed=seed + 1000)

    elif system_id == "1C":
        from m8_battery.systems.class1.foxworthy_a import FoxworthyA
        s = FoxworthyA(n_features=n_features, seed=seed)
        s.set_graph(graph)
        return s, lambda: _clone_with_graph(FoxworthyA(n_features=n_features, seed=seed + 1000), graph)

    elif system_id == "2A":
        from m8_battery.systems.class2.frozen_llm import FrozenLLM
        s = FrozenLLM(seed=seed, device=device)
        s.set_graph(graph)
        return s, lambda: _clone_with_graph(FrozenLLM(seed=seed + 1000, device=device), graph)

    elif system_id == "2B":
        from m8_battery.systems.class2.frozen_gnn import FrozenGAT
        s = FrozenGAT(n_features=n_features, n_classes=n_classes, seed=seed)
        s.train_on_domain(graph)
        return s, lambda: (lambda g: (g.train_on_domain(graph), g)[-1])(FrozenGAT(n_features=n_features, n_classes=n_classes, seed=seed + 1000))

    elif system_id == "2C":
        from m8_battery.systems.class2.foxworthy_c import FoxworthyC
        s = FoxworthyC(n_features=n_features, hidden_dim=32, seed=seed)
        s.set_graph(graph)
        return s, lambda: _clone_with_graph(FoxworthyC(n_features=n_features, hidden_dim=32, seed=seed + 1000), graph)

    elif system_id == "3A":
        from m8_battery.systems.class3.dqn_agent import DQNAgent
        s = DQNAgent(n_features=n_features, seed=seed, total_timesteps=2000)
        s.train_on_domain(graph)
        return s, lambda: (lambda g: (g.train_on_domain(graph), g)[-1])(DQNAgent(n_features=n_features, seed=seed + 1000, total_timesteps=2000))

    elif system_id == "3B":
        from m8_battery.systems.class3.curiosity_agent import CuriosityAgent
        s = CuriosityAgent(n_features=n_features, seed=seed, total_timesteps=2000)
        s.train_on_domain(graph)
        return s, lambda: (lambda g: (g.train_on_domain(graph), g)[-1])(CuriosityAgent(n_features=n_features, seed=seed + 1000, total_timesteps=2000))

    elif system_id == "3C":
        from m8_battery.systems.class3.foxworthy_f import FoxworthyF
        s = FoxworthyF(seed=seed, device=device, theta=0.0)
        s.train_on_domain(graph, n_warmup=50)
        def _make_3c_control(g=graph, d=device, sd=seed):
            """Control factory: graph-attached, model-loaded fresh FoxworthyF.
            train_on_domain(g, n_warmup=0) loads model without training.
            See F-022 lazy loading investigation."""
            f = FoxworthyF(seed=sd + 1000, device=d, theta=0.0)
            f.train_on_domain(g, n_warmup=0)
            return f
        return s, _make_3c_control

    elif system_id == "HEB":
        from m8_battery.systems.internal.hebbian_walker import HebbianWalker
        s = HebbianWalker(graph, seed=seed, eta=0.1, decay=0.01, temperature=0.5)
        s.train_on_domain(graph, n_steps=300)
        return s, lambda sd=seed: HebbianWalker(graph, seed=sd + 1000, eta=0.1, decay=0.01, temperature=0.5)

    elif system_id == "STDP":
        from m8_battery.systems.anchor.stdp_network import STDPNetwork
        s = STDPNetwork(n_neurons=1000, connection_prob=0.1, seed=seed, n_groups=4)
        s.train_on_domain(None, duration_s=10.0)
        def _make_stdp_fresh(sd=seed):
            f = STDPNetwork(n_neurons=1000, connection_prob=0.1, seed=sd + 1000, n_groups=4)
            for _ in range(100):
                f.step(0)
            return f
        return s, _make_stdp_fresh

    elif system_id == "3D":
        from m8_battery.systems.class3.empowerment_agent import EmpowermentAgent
        s = EmpowermentAgent(graph, seed=seed, recompute_interval=100)
        s.train_on_domain(graph, n_steps=2000)
        return s, lambda g=graph, sd=seed: EmpowermentAgent(g, seed=sd + 1000, recompute_interval=100)

    elif system_id == "3E":
        from m8_battery.systems.class3.active_inference_agent import ActiveInferenceAgent
        s = ActiveInferenceAgent(graph, seed=seed)
        s.train_on_domain(graph, n_steps=1000)
        return s, lambda g=graph, sd=seed: ActiveInferenceAgent(g, seed=sd + 1000)

    raise ValueError(f"Unknown system: {system_id}")


def run_single(system_id, seed):
    config = DomainConfig(
        n_nodes=MEDIUM.n_nodes, n_communities=MEDIUM.n_communities,
        p_within=MEDIUM.p_within, p_between=MEDIUM.p_between,
        n_edge_types=MEDIUM.n_edge_types, weight_range=MEDIUM.weight_range,
        n_node_features=MEDIUM.n_node_features, seed=seed,
    )
    family = generate_domain_family(config)
    graph_a = family["A"]
    nodes_a = list(graph_a.nodes())

    t0 = time.time()
    system, control_factory = make_system(system_id, graph_a, seed, config.n_node_features)
    t_setup = time.time() - t0

    # STDP needs more inputs for trajectory detection (R² requires enough measurement points)
    if system_id == "STDP":
        n_inputs = 160
    elif system_id in LLM_SYSTEMS:
        n_inputs = 20
    else:
        n_inputs = 50
    battery_config = BatteryConfig(
        domain_a_inputs=nodes_a[:n_inputs],
        domain_a_prime_inputs=list(family["A_prime"].nodes())[:n_inputs],
        domain_b_inputs=list(family["B"].nodes())[:n_inputs],
        probe_inputs=nodes_a[:10],
        measurement_interval=5,
        wander_steps=15,
        recovery_window=15,
    )

    t0 = time.time()
    result = run_battery(
        system=system, system_name=SYSTEM_NAMES[system_id],
        system_class=SYSTEM_CLASSES[system_id],
        config=battery_config, control_factory=control_factory,
    )
    t_battery = time.time() - t0

    if hasattr(system, 'unload_model'):
        system.unload_model()

    instruments = {}
    for name, ir in result.instrument_results.items():
        instruments[name] = {
            "passed": ir.passed, "effect_size": ir.effect_size, "notes": ir.notes,
        }

    baseline = result.metadata.get("baseline", {})

    out = {
        "system_id": system_id,
        "system_name": SYSTEM_NAMES[system_id],
        "system_class": SYSTEM_CLASSES[system_id].value,
        "scale": "medium",
        "seed": seed,
        "n_nodes": MEDIUM.n_nodes,
        "overall_passed": result.overall_passed,
        "provenance_passed": result.provenance_passed,
        "setup_time_s": round(t_setup, 1),
        "battery_time_s": round(t_battery, 1),
        "run_machine": "m5_max",
        "device": get_device(system_id),
        "instruments": instruments,
    }

    # Add baseline classifications
    cls = baseline.get("instrument_classifications", {})
    if cls:
        out["classifications"] = cls

    # Add trajectory
    out["trajectory_training"] = baseline.get("trajectory_training", "unknown")
    out["trajectory_battery"] = baseline.get("trajectory_battery", "unknown")

    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_systems = ["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C", "HEB", "STDP", "3D", "3E"]
    total = len(all_systems) * len(SEEDS)

    print(f"M5 Max Full Recalibration (T1-09) — {total} runs")
    print(f"Scale: MEDIUM (150 nodes, 6 communities)")
    print(f"LLM systems ({LLM_SYSTEMS}) on MPS, rest on CPU")
    print(f"Anchor systems: HEB (internal), STDP (1000 neurons, ~18min/run)")
    print("=" * 70)

    count = 0
    for system_id in all_systems:
        for seed in SEEDS:
            count += 1
            print(f"\n[{count}/{total}] {system_id} seed={seed} device={get_device(system_id)}")
            sys.stdout.flush()

            try:
                result = run_single(system_id, seed)

                # Save individual result
                fname = f"{system_id}_medium_{seed}.json"
                with open(RESULTS_DIR / fname, "w") as f:
                    json.dump(result, f, indent=2)

                # Print summary
                inst = result["instruments"]
                cls = result.get("classifications", {})
                print(f"  overall={result['overall_passed']} "
                      f"setup={result['setup_time_s']}s battery={result['battery_time_s']}s")
                for name, r in inst.items():
                    c = cls.get(name, "?")
                    es = r['effect_size'] if r['effect_size'] is not None else 0.0
                    print(f"    {name}: passed={r['passed']} ES={es:.4f} cls={c}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

            sys.stdout.flush()

    # Summary
    print("\n" + "=" * 70)
    print("RECALIBRATION COMPLETE")
    print(f"Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

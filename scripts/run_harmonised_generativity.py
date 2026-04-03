#!/usr/bin/env python3
"""Harmonised generativity runner — single session, all domains, both metrics.

Generates B₀/B₁/B₂ from a single domain A instance. Runs all 13 calibration
systems + positive controls with fixed 500 steps, full logging, visit sequence
capture, and both marginal + transition JSD.

Also computes null distributions (50 pairs per system type) and exports raw
null samples to CSV.

Output: JSON results to stdout, null CSV to results/ directory.
"""

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from m8_battery.domains.sbm_generator import generate_domain_family
from m8_battery.domains.presets import MEDIUM
from m8_battery.instruments.generativity import (
    MAX_JSD,
    _compute_transition_matrix,
    _coherence_normalised,
    _count_visited,
    _engagement_entropy,
    _js_divergence,
    _self_transition_rate,
    _structural_consistency,
    _transition_entropy,
    _transition_jsd,
)
from m8_battery.instruments.role_utils import classify_all_nodes, compute_role_transition_matrix


def _log(msg: str) -> None:
    print(f"[harmonised] {msg}", file=sys.stderr, flush=True)


N_STEPS = 500  # Fixed step count for all generativity measurements
SEEDS = [42, 123, 456]


@dataclass
class GenerativityResult:
    system_name: str
    seed: int
    domain_variant: str
    marginal_jsd: float
    transition_jsd: float
    marginal_coherence: float
    transition_coherence: float
    trained_self_transition: float
    fresh_self_transition: float
    trained_entropy: float
    fresh_entropy: float
    trained_visited: int
    fresh_visited: int
    structural_consistency: float
    edge_jaccard: float
    signal_type: str
    commit: str


def edge_jaccard(g1, g2) -> float:
    e1 = set(g1.edges())
    e2 = set(g2.edges())
    if not e1 and not e2:
        return 0.0
    return len(e1 & e2) / len(e1 | e2)


def get_community_map(graph) -> dict:
    mapping = {}
    for node in graph.nodes():
        data = graph.nodes[node]
        features = data.get("features", {})
        mapping[node] = features.get("community", data.get("block", 0))
    return mapping


def run_generativity_measurement(
    system,
    fresh_system,
    domain_graph,
    system_name: str,
    seed: int,
    domain_variant: str,
    ej: float,
    commit: str,
    training_role_T: np.ndarray | None = None,
) -> GenerativityResult:
    """Run generativity protocol on a single system/domain/seed."""
    node_to_community = get_community_map(domain_graph)
    n_communities = len(set(node_to_community.values()))

    # Trained system on domain
    try:
        system.set_domain(domain_graph)
    except (AttributeError, NotImplementedError):
        pass  # Systems without set_domain navigate their init graph
    system.set_training(False)
    system.reset_engagement_tracking()
    trained_seq = []
    for _ in range(N_STEPS):
        output = system.step(None)
        # Some systems return dicts or tuples — extract node if hashable
        if isinstance(output, (int, float, str, np.integer)):
            trained_seq.append(int(output))
    trained_eng = system.get_engagement_distribution()
    trained_entropy = _engagement_entropy(trained_eng)
    trained_visited = _count_visited(trained_eng)

    # Fresh system on domain
    try:
        fresh_system.set_domain(domain_graph)
    except (AttributeError, NotImplementedError):
        pass
    fresh_system.set_training(False)
    fresh_system.reset_engagement_tracking()
    fresh_seq = []
    for _ in range(N_STEPS):
        output = fresh_system.step(None)
        if isinstance(output, (int, float, str, np.integer)):
            fresh_seq.append(int(output))
    fresh_eng = fresh_system.get_engagement_distribution()
    fresh_entropy = _engagement_entropy(fresh_eng)
    fresh_visited = _count_visited(fresh_eng)

    # Marginal JSD
    regions = sorted(set(list(trained_eng.keys()) + list(fresh_eng.keys())))
    t_vec = np.array([trained_eng.get(r, 0.0) for r in regions])
    f_vec = np.array([fresh_eng.get(r, 0.0) for r in regions])
    m_jsd = _js_divergence(t_vec, f_vec)

    # Marginal coherence
    m_coh = _coherence_normalised(trained_entropy, fresh_entropy)

    # Transition metrics
    t_jsd = 0.0
    t_coh = 0.0
    trained_st = 0.0
    fresh_st = 0.0
    if n_communities > 0 and len(trained_seq) > 1 and len(fresh_seq) > 1:
        T_trained = _compute_transition_matrix(trained_seq, node_to_community, n_communities)
        T_fresh = _compute_transition_matrix(fresh_seq, node_to_community, n_communities)
        t_jsd = _transition_jsd(T_trained, T_fresh)
        t_coh = _coherence_normalised(_transition_entropy(T_trained), _transition_entropy(T_fresh))
        trained_st = _self_transition_rate(T_trained)
        fresh_st = _self_transition_rate(T_fresh)

    # DN-30a: Role-aggregated structural consistency
    sc = 0.0
    if len(trained_seq) > 1 and len(fresh_seq) > 1:
        node_to_role = classify_all_nodes(domain_graph)
        T_role_trained_B = compute_role_transition_matrix(trained_seq, node_to_role)
        T_role_fresh_B = compute_role_transition_matrix(fresh_seq, node_to_role)
        if training_role_T is not None:
            sc = _structural_consistency(training_role_T, T_role_trained_B, T_role_fresh_B)

    # Signal classification (simplified — confound based on edge Jaccard)
    signal = "candidate"
    if m_jsd < 1e-6 and t_jsd < 1e-6:
        signal = "absent"
    elif trained_visited < 3 or fresh_visited < 3:
        signal = "degenerate"
    elif ej > 0.05:
        signal = "potentially_confounded"
    elif m_coh <= 0 and t_coh <= 0:
        signal = "divergent_incoherent"

    return GenerativityResult(
        system_name=system_name,
        seed=seed,
        domain_variant=domain_variant,
        marginal_jsd=round(m_jsd, 6),
        transition_jsd=round(t_jsd, 6),
        marginal_coherence=round(m_coh, 4),
        transition_coherence=round(t_coh, 4),
        trained_self_transition=round(trained_st, 4),
        fresh_self_transition=round(fresh_st, 4),
        trained_entropy=round(trained_entropy, 4),
        fresh_entropy=round(fresh_entropy, 4),
        trained_visited=trained_visited,
        fresh_visited=fresh_visited,
        structural_consistency=round(sc, 4),
        edge_jaccard=round(ej, 4),
        signal_type=signal,
        commit=commit,
    )


def compute_null_pair(
    system_factory: Callable,
    domain_a,
    domain_b,
    seed_a: int,
    seed_b: int,
    system_type: str,
    domain_variant: str,
) -> dict:
    """Compute one null pair: two untrained systems on domain B."""
    node_to_community = get_community_map(domain_b)
    n_communities = len(set(node_to_community.values()))

    sys_a = system_factory(domain_a, seed_a)
    sys_a.set_domain(domain_b)
    sys_a.set_training(False)
    sys_a.reset_engagement_tracking()
    seq_a = []
    for _ in range(N_STEPS):
        seq_a.append(sys_a.step(None))
    eng_a = sys_a.get_engagement_distribution()

    sys_b = system_factory(domain_a, seed_b)
    sys_b.set_domain(domain_b)
    sys_b.set_training(False)
    sys_b.reset_engagement_tracking()
    seq_b = []
    for _ in range(N_STEPS):
        seq_b.append(sys_b.step(None))
    eng_b = sys_b.get_engagement_distribution()

    # Marginal JSD
    regions = sorted(set(list(eng_a.keys()) + list(eng_b.keys())))
    vec_a = np.array([eng_a.get(r, 0.0) for r in regions])
    vec_b = np.array([eng_b.get(r, 0.0) for r in regions])
    m_jsd = _js_divergence(vec_a, vec_b)

    # Transition JSD
    t_jsd = 0.0
    if n_communities > 0 and len(seq_a) > 1 and len(seq_b) > 1:
        T_a = _compute_transition_matrix(seq_a, node_to_community, n_communities)
        T_b = _compute_transition_matrix(seq_b, node_to_community, n_communities)
        t_jsd = _transition_jsd(T_a, T_b)

    # Structural consistency null: both systems are untrained.
    # _structural_consistency(T_trained_A, T_trained_B, T_fresh_B) expects:
    #   arg1 = reference transition pattern from training domain
    #   arg2 = trained system's pattern on novel domain
    #   arg3 = fresh system's pattern on novel domain
    # For null pairs: sys_a is arbitrarily designated as "reference/trained"
    # and sys_b as "fresh". Since neither has trained, SC should be ≈ 0.
    # Using T_role_a for both arg1 and arg2 means we're asking: does sys_a's
    # B-pattern resemble its own B-pattern more than sys_b's does? This
    # measures how much two independent untrained walkers diverge in their
    # relationship to an arbitrary reference — i.e., seed noise in SC space.
    sc = 0.0
    if len(seq_a) > 1 and len(seq_b) > 1:
        node_to_role = classify_all_nodes(domain_b)
        T_role_a = compute_role_transition_matrix(seq_a, node_to_role)
        T_role_b = compute_role_transition_matrix(seq_b, node_to_role)
        sc = _structural_consistency(T_role_a, T_role_a, T_role_b)

    return {
        "system_type": system_type,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "marginal_jsd": round(m_jsd, 6),
        "transition_jsd": round(t_jsd, 6),
        "structural_consistency": round(sc, 6),
        "domain_variant": domain_variant,
    }


def main():
    import subprocess
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
    ).decode().strip()

    _log(f"Harmonised generativity runner — commit {commit}")
    _log(f"Fixed step count: {N_STEPS}")

    # --- Generate all domains from single seed ---
    _log("Generating domain family (MEDIUM, seed=42)...")
    family = generate_domain_family(MEDIUM)
    domain_a = family["A"]
    domain_b0 = family["A_prime"]  # B₀: isomorphic
    domain_b1 = family["B"]       # B₁: same params, different seed
    # B₂: zero-overlap — create with shifted node IDs
    import networkx as nx
    from m8_battery.core.types import DomainConfig
    from m8_battery.domains.sbm_generator import generate_domain

    rng = np.random.default_rng(42)
    config_b2 = DomainConfig(
        n_nodes=MEDIUM.n_nodes,
        n_communities=MEDIUM.n_communities,
        p_within=MEDIUM.p_within,
        p_between=MEDIUM.p_between,
        seed=int(rng.integers(10000, 99999)),
    )
    domain_b2 = generate_domain(config_b2)
    # Shift node IDs to ensure zero overlap
    shift = max(domain_a.nodes()) + 1000
    domain_b2 = nx.relabel_nodes(domain_b2, {n: n + shift for n in domain_b2.nodes()})

    ej_b0 = edge_jaccard(domain_a, domain_b0)
    ej_b1 = edge_jaccard(domain_a, domain_b1)
    ej_b2 = edge_jaccard(domain_a, domain_b2)
    _log(f"Edge Jaccard: A-B₀={ej_b0:.4f}, A-B₁={ej_b1:.4f}, A-B₂={ej_b2:.4f}")

    # --- System factories ---
    from m8_battery.systems.class1.wordnet_graph import WordNetGraph
    from m8_battery.systems.class1.rule_navigator import RuleBasedNavigator
    from m8_battery.systems.class1.foxworthy_a import FoxworthyA
    from m8_battery.systems.internal.hebbian_walker import HebbianWalker
    from m8_battery.systems.class3.empowerment_agent import EmpowermentAgent
    from m8_battery.systems.class3.active_inference_agent import ActiveInferenceAgent
    from m8_battery.systems.positive_controls.role_based_walker import RoleBasedWalker
    from m8_battery.systems.positive_controls.gnn_navigator import GNNNavigator

    # GPU device for LLM-based systems
    _llm_device = "mps" if torch.backends.mps.is_available() else "cpu"
    _log(f"LLM device: {_llm_device}")

    # Systems that support set_domain() — can test on all domain variants
    graph_walker_configs = {
        "HEB": {
            "factory": lambda g, s: HebbianWalker(g, seed=s, eta=0.1, decay=0.01, temperature=0.5),
            "fresh": lambda g, s: HebbianWalker(g, seed=s + 1000),
            "train_steps": 2000,
        },
        "3D": {
            "factory": lambda g, s: EmpowermentAgent(g, seed=s),
            "fresh": lambda g, s: EmpowermentAgent(g, seed=s + 1000),
            "train_steps": 2000,
        },
        "3E": {
            "factory": lambda g, s: ActiveInferenceAgent(g, seed=s),
            "fresh": lambda g, s: ActiveInferenceAgent(g, seed=s + 1000),
            "train_steps": 2000,
        },
        "PC1": {
            "factory": lambda g, s: RoleBasedWalker(g, seed=s, eta=0.8, temperature=0.01),
            "fresh": lambda g, s: RoleBasedWalker(g, seed=s + 1000),
            "train_steps": 10000,
        },
        "PC3": {
            "factory": lambda g, s: GNNNavigator(g, seed=s, train_epochs=500, temperature=0.05),
            "fresh": lambda g, s: GNNNavigator(g, seed=s + 1000),
            "train_steps": 2000,
        },
    }

    # Systems without set_domain() — B₁ only (standard domain B via battery runner)
    # Class 1/2 systems produce JSD=0 by design (no learning, trained=fresh).
    # They navigate their init graph, so set_domain is a no-op.
    non_graph_configs = {
        "1A": {
            "factory": lambda g, s: WordNetGraph(g, seed=s),
            "fresh": lambda g, s: WordNetGraph(g, seed=s + 1000),
            "train_steps": 0,
        },
        "1B": {
            "factory": lambda g, s: RuleBasedNavigator(g, seed=s),
            "fresh": lambda g, s: RuleBasedNavigator(g, seed=s + 1000),
            "train_steps": 0,
        },
    }
    # All remaining systems now have set_domain(). Add what we can import.
    try:
        from m8_battery.systems.class1.foxworthy_a import FoxworthyA
        non_graph_configs["1C"] = {
            "factory": lambda g, s: (fa := FoxworthyA(seed=s), fa.set_graph(g), fa)[-1],
            "fresh": lambda g, s: (fa := FoxworthyA(seed=s + 1000), fa.set_graph(g), fa)[-1],
            "train_steps": 0,
        }
    except ImportError:
        _log("SKIP 1C: FoxworthyA import failed")

    try:
        from m8_battery.systems.class2.frozen_gnn import FrozenGAT
        def _make_gat(g, s):
            n_feat = MEDIUM.n_node_features if hasattr(MEDIUM, 'n_node_features') else 8
            n_comm = MEDIUM.n_communities
            sys = FrozenGAT(n_features=n_feat, n_classes=n_comm, seed=s)
            sys.set_domain(g)
            return sys
        non_graph_configs["2B"] = {
            "factory": _make_gat,
            "fresh": lambda g, s: _make_gat(g, s + 1000),
            "train_steps": 0,
        }
    except ImportError:
        _log("SKIP 2B: FrozenGAT import failed (torch_geometric)")

    try:
        from m8_battery.systems.class2.foxworthy_c import FoxworthyC
        def _make_fc(g, s):
            n_feat = MEDIUM.n_node_features if hasattr(MEDIUM, 'n_node_features') else 8
            sys = FoxworthyC(n_features=n_feat, seed=s)
            sys.set_domain(g)
            return sys
        non_graph_configs["2C"] = {
            "factory": _make_fc,
            "fresh": lambda g, s: _make_fc(g, s + 1000),
            "train_steps": 0,
        }
    except ImportError:
        _log("SKIP 2C: FoxworthyC import failed")

    # 2A FrozenLLM: constructor takes (seed, device), graph via set_graph()
    try:
        from m8_battery.systems.class2.frozen_llm import FrozenLLM
        pass  # device set globally below
        def _make_llm(g, s):
            sys = FrozenLLM(seed=s, device=_llm_device)
            sys.set_graph(g)
            return sys
        non_graph_configs["2A"] = {
            "factory": _make_llm,
            "fresh": lambda g, s: _make_llm(g, s + 1000),
            "train_steps": 0,
        }
    except ImportError:
        _log("SKIP 2A: FrozenLLM import failed")

    # 3A DQN: constructor takes (n_features, max_degree, seed), graph via set_graph()
    try:
        from m8_battery.systems.class3.dqn_agent import DQNAgent
        def _make_dqn(g, s):
            sys = DQNAgent(seed=s)
            sys.set_graph(g)
            return sys
        non_graph_configs["3A"] = {
            "factory": _make_dqn,
            "fresh": lambda g, s: _make_dqn(g, s + 1000),
            "train_steps": 0,  # Don't train — just test frozen (untrained) on B
        }
    except ImportError:
        _log("SKIP 3A: DQNAgent import failed (sb3)")

    # 3B Curiosity: same pattern as DQN
    try:
        from m8_battery.systems.class3.curiosity_agent import CuriosityAgent
        def _make_curiosity(g, s):
            sys = CuriosityAgent(seed=s)
            sys.set_graph(g)
            return sys
        non_graph_configs["3B"] = {
            "factory": _make_curiosity,
            "fresh": lambda g, s: _make_curiosity(g, s + 1000),
            "train_steps": 0,
        }
    except ImportError:
        _log("SKIP 3B: CuriosityAgent import failed (sb3)")

    # 3C FoxworthyF: requires peft (DistilGPT-2 + LoRA)
    try:
        from m8_battery.systems.class3.foxworthy_f import FoxworthyF
        def _make_ff(g, s):
            sys = FoxworthyF(seed=s, device=_llm_device)
            sys.set_graph(g)
            return sys
        non_graph_configs["3C"] = {
            "factory": _make_ff,
            "fresh": lambda g, s: _make_ff(g, s + 1000),
            "train_steps": 0,
        }
    except ImportError:
        _log("SKIP 3C: FoxworthyF import failed (peft)")

    all_results = []

    # --- Phase A: Graph walker systems on B₁ + B₂ (+ B₀ for PC1/PC3) ---
    for name, cfg in graph_walker_configs.items():
        domains = [("B1", domain_b1, ej_b1), ("B2", domain_b2, ej_b2)]
        if name in ("PC1", "PC3"):
            domains.insert(0, ("B0", domain_b0, ej_b0))

        for domain_label, domain_graph, ej in domains:
            for seed in SEEDS:
                _log(f"  {name} seed={seed} on {domain_label}")
                t0 = time.monotonic()

                trained = cfg["factory"](domain_a, seed)
                if cfg["train_steps"] > 0:
                    trained.train_on_domain(domain_a, n_steps=cfg["train_steps"])

                # DN-30a: Capture training transition matrix (role-aggregated)
                # Post-training recording phase on domain A (frozen)
                trained.set_training(False)
                trained.reset_engagement_tracking()
                training_seq = []
                for _ in range(N_STEPS):
                    out = trained.step(None)
                    if isinstance(out, (int, float, str, np.integer)):
                        training_seq.append(int(out))
                node_to_role_a = classify_all_nodes(domain_a)
                training_role_T = compute_role_transition_matrix(training_seq, node_to_role_a)

                # DN-37: sync fresh baseline starting position with trained
                init_pos = trained.get_initial_position()
                fresh = cfg["fresh"](domain_a, seed)
                if init_pos is not None and hasattr(fresh, '_current_node'):
                    fresh._current_node = init_pos
                    fresh._initial_position = init_pos

                result = run_generativity_measurement(
                    trained, fresh, domain_graph,
                    name, seed, domain_label, ej, commit,
                    training_role_T=training_role_T,
                )
                all_results.append(result)
                _log(f"    m_jsd={result.marginal_jsd:.4f} t_jsd={result.transition_jsd:.4f} "
                     f"sc={result.structural_consistency:.4f} "
                     f"({time.monotonic()-t0:.1f}s)")

    # --- Phase B: Non-graph-walker systems on B₁ ---
    # These now have set_domain(). Run on actual B₁ domain.
    for name, cfg in non_graph_configs.items():
        for seed in SEEDS:
            _log(f"  {name} seed={seed} on B1")
            t0 = time.monotonic()

            trained = cfg["factory"](domain_a, seed)
            if cfg["train_steps"] > 0:
                trained.train_on_domain(domain_a, n_steps=cfg["train_steps"])

            # Post-training recording on A (same as Phase A)
            trained.set_training(False)
            trained.reset_engagement_tracking()
            training_seq = []
            for _ in range(N_STEPS):
                out = trained.step(None)
                if isinstance(out, (int, float, str, np.integer)):
                    training_seq.append(int(out))
            node_to_role_a = classify_all_nodes(domain_a)
            training_role_T = compute_role_transition_matrix(training_seq, node_to_role_a)

            # DN-37: sync fresh baseline starting position
            init_pos = trained.get_initial_position()
            fresh = cfg["fresh"](domain_a, seed)
            if init_pos is not None and hasattr(fresh, '_current_node'):
                fresh._current_node = init_pos
                fresh._initial_position = init_pos

            result = run_generativity_measurement(
                trained, fresh, domain_b1,
                name, seed, "B1", ej_b1, commit,
                training_role_T=training_role_T,
            )
            all_results.append(result)
            _log(f"    m_jsd={result.marginal_jsd:.4f} t_jsd={result.transition_jsd:.4f} "
                 f"sc={result.structural_consistency:.4f} "
                 f"({time.monotonic()-t0:.1f}s)")

    # --- Phase C: Null distributions (50 pairs per type, B₁ + B₂) ---
    _log("=== Null distributions ===")
    null_samples = []
    null_types = {
        "PC1": graph_walker_configs["PC1"],
        "PC3": graph_walker_configs["PC3"],
        "HEB": graph_walker_configs["HEB"],
        "3D": graph_walker_configs["3D"],
        "3E": graph_walker_configs["3E"],
    }
    for sys_type, cfg in null_types.items():
        for domain_label, domain_graph in [("B1", domain_b1), ("B2", domain_b2)]:
            _log(f"  Null: {sys_type} on {domain_label} (50 pairs)")
            t0 = time.monotonic()
            for i in range(50):
                seed_a = i
                seed_b = i + 1000
                pair = compute_null_pair(
                    cfg["fresh"], domain_a, domain_graph,
                    seed_a, seed_b, sys_type, domain_label,
                )
                null_samples.append(pair)
            _log(f"    {time.monotonic()-t0:.1f}s")

    # --- Export null samples to CSV ---
    os.makedirs("results", exist_ok=True)
    csv_path = "results/null_distribution_raw_samples.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "system_type", "seed_a", "seed_b",
            "marginal_jsd", "transition_jsd", "structural_consistency", "domain_variant",
        ])
        writer.writeheader()
        writer.writerows(null_samples)
    _log(f"Null samples exported: {csv_path} ({len(null_samples)} rows)")

    # --- Compute null summary statistics ---
    null_summary = {}
    for sys_type in null_types:
        for dv in ["B1", "B2"]:
            key = f"{sys_type}_{dv}"
            samples = [s for s in null_samples if s["system_type"] == sys_type and s["domain_variant"] == dv]
            m_vals = [s["marginal_jsd"] for s in samples]
            t_vals = [s["transition_jsd"] for s in samples]
            sc_vals = [s["structural_consistency"] for s in samples]
            null_summary[key] = {
                "n": len(samples),
                "marginal_mean": round(np.mean(m_vals), 4),
                "marginal_p95": round(np.percentile(m_vals, 95), 4),
                "marginal_max": round(np.max(m_vals), 4),
                "transition_mean": round(np.mean(t_vals), 4),
                "transition_p95": round(np.percentile(t_vals, 95), 4),
                "transition_max": round(np.max(t_vals), 4),
                "sc_mean": round(np.mean(sc_vals), 4),
                "sc_p95": round(np.percentile(sc_vals, 95), 4),
                "sc_max": round(np.max(sc_vals), 4),
            }

    # --- Task 4: Bootstrap CIs on noise floors ---
    _log("=== Bootstrap CIs ===")
    bootstrap_cis = {}
    for key, summary in null_summary.items():
        samples = [s for s in null_samples
                   if f"{s['system_type']}_{s['domain_variant']}" == key]
        t_vals = np.array([s["transition_jsd"] for s in samples])
        if len(t_vals) >= 10:
            boot_p95s = []
            for _ in range(10000):
                boot = np.random.choice(t_vals, size=len(t_vals), replace=True)
                boot_p95s.append(np.percentile(boot, 95))
            boot_p95s = np.array(boot_p95s)
            ci_lo, ci_hi = np.percentile(boot_p95s, [2.5, 97.5])
            bootstrap_cis[key] = {
                "p95": round(np.percentile(t_vals, 95), 6),
                "ci_lower": round(ci_lo, 6),
                "ci_upper": round(ci_hi, 6),
            }
            _log(f"  {key}: p95={np.percentile(t_vals, 95):.4f} "
                 f"CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    # --- Task 5: ROC/AUC ---
    _log("=== ROC/AUC (B₂ transition JSD) ===")
    # Collect B₂ scores
    pos_scores = [r.transition_jsd for r in all_results
                  if r.system_name in ("PC1", "PC3") and r.domain_variant == "B2"]
    neg_scores = [r.transition_jsd for r in all_results
                  if r.system_name not in ("PC1", "PC3") and r.domain_variant == "B2"]

    roc_result = None
    if pos_scores and neg_scores:
        y_true = np.array([0]*len(neg_scores) + [1]*len(pos_scores))
        y_scores = np.array(neg_scores + pos_scores)
        # AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_scores)
        # Bootstrap CI
        aucs = []
        for _ in range(10000):
            idx = np.random.choice(len(y_true), len(y_true), replace=True)
            if len(set(y_true[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
        aucs = np.array(aucs)
        auc_ci_lo, auc_ci_hi = np.percentile(aucs, [2.5, 97.5])
        roc_result = {
            "auc": round(auc, 4),
            "ci_lower": round(auc_ci_lo, 4),
            "ci_upper": round(auc_ci_hi, 4),
            "n_positive": len(pos_scores),
            "n_negative": len(neg_scores),
        }
        _log(f"  AUC={auc:.4f} CI=[{auc_ci_lo:.4f}, {auc_ci_hi:.4f}] "
             f"(n_pos={len(pos_scores)}, n_neg={len(neg_scores)})")

    # --- Output ---
    output = {
        "commit": commit,
        "n_steps": N_STEPS,
        "domains": {
            "edge_jaccard_b0": round(ej_b0, 4),
            "edge_jaccard_b1": round(ej_b1, 4),
            "edge_jaccard_b2": round(ej_b2, 4),
        },
        "results": [r.__dict__ for r in all_results],
        "null_summary": null_summary,
        "bootstrap_cis": bootstrap_cis,
        "roc_auc": roc_result,
    }
    print(json.dumps(output, indent=2))
    _log("Done.")


if __name__ == "__main__":
    main()

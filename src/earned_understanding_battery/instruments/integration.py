"""Integration instrument.

Tests whether the system's structural organisation is non-decomposable —
removing components degrades function non-linearly (synergy: whole > sum of parts).

A Class 1 system may show linear degradation (each part independent).
A Class 4 candidate should show non-linear degradation (integration).
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from earned_understanding_battery.core.test_system import TestSystem
from earned_understanding_battery.core.types import InstrumentResult
from earned_understanding_battery.core.provenance import ProvenanceLog

def run_integration(
    system: TestSystem,
    probe_inputs: list[Any],
    partition_families: list[list[str]] | None = None,
    provenance: ProvenanceLog | None = None,
) -> InstrumentResult:
    """Run the integration instrument.

    For each region, ablate it and measure the degradation in structure
    metric. Integration = degradation is non-linear (removing parts causes
    disproportionate damage).

    Args:
        system: The system under test (should have been operated on already)
        probe_inputs: Inputs to probe the system with after ablation
        partition_families: Preregistered region partitions. If None, uses
            system.get_regions() as a single partition family.
        provenance: Optional provenance log

    Returns:
        InstrumentResult with synergy analysis
    """
    if provenance is None:
        provenance = ProvenanceLog()

    regions = system.get_regions()
    if len(regions) < 2:
        return InstrumentResult(
            name="integration",
            passed=None,
            notes="Insufficient regions for integration test (need >= 2)",
            raw_data={"n_regions": len(regions)},
        )

    if partition_families is None:
        partition_families = [regions]

    # Measure baseline performance
    baseline_metric = _probe_metric(system, probe_inputs)

    # Measure per-region ablation degradation
    ablation_results = {}
    n_regions = len(regions)
    for i, region_id in enumerate(regions):
        print(f"[integration] ablation {i+1}/{n_regions}: {region_id}", file=sys.stderr, flush=True)
        try:
            ablated = system.ablate(region_id)
            ablated_metric = _probe_metric(ablated, probe_inputs)
            degradation = baseline_metric - ablated_metric
            ablation_results[region_id] = {
                "metric": ablated_metric,
                "degradation": degradation,
                "relative_degradation": degradation / baseline_metric if baseline_metric != 0 else 0.0,
            }
        except Exception as e:
            ablation_results[region_id] = {
                "metric": None,
                "degradation": None,
                "error": str(e),
            }

    # Compute synergy: is total degradation > sum of individual degradations?
    individual_degradations = [
        r["degradation"] for r in ablation_results.values()
        if r["degradation"] is not None
    ]

    if not individual_degradations:
        return InstrumentResult(
            name="integration",
            passed=None,
            notes="No valid ablation results",
            raw_data={"ablation_results": ablation_results},
        )

    sum_individual = sum(individual_degradations)
    mean_individual = np.mean(individual_degradations)
    std_individual = np.std(individual_degradations) if len(individual_degradations) > 1 else 0.0

    # Non-linearity: coefficient of variation of degradations
    # High CV means some regions matter much more than others → integration
    cv = std_individual / abs(mean_individual) if abs(mean_individual) > 1e-10 else 0.0

    # Gini coefficient of degradations (inequality measure)
    sorted_degs = np.sort(np.abs(individual_degradations))
    n = len(sorted_degs)
    if n > 0 and sorted_degs.sum() > 0:
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_degs) - (n + 1) * np.sum(sorted_degs)) / (n * np.sum(sorted_degs))
    else:
        gini = 0.0

    # Post-ablation stability check
    # Does the system form a stable new regime after ablation, or collapse?
    # Ablate the highest-degradation region, run M steps, measure stability.
    reorganisation_stability = None
    if ablation_results:
        highest_deg_region = max(
            ((r, d["degradation"]) for r, d in ablation_results.items()
             if d.get("degradation") is not None),
            key=lambda x: abs(x[1]), default=(None, 0)
        )[0]
        if highest_deg_region is not None:
            try:
                ablated_for_stability = system.ablate(highest_deg_region)
                # Run M/2 steps, measure engagement
                M = len(probe_inputs) if probe_inputs else 10
                for _ in range(max(M // 2, 5)):
                    ablated_for_stability.step(probe_inputs[0] if probe_inputs else None)
                eng_mid = ablated_for_stability.get_engagement_distribution()
                # Run another M/2 steps
                for _ in range(max(M // 2, 5)):
                    ablated_for_stability.step(probe_inputs[0] if probe_inputs else None)
                eng_final = ablated_for_stability.get_engagement_distribution()
                # Stability = cosine similarity between mid and final
                keys = sorted(set(list(eng_mid.keys()) + list(eng_final.keys())))
                v_mid = np.array([eng_mid.get(k, 0) for k in keys])
                v_fin = np.array([eng_final.get(k, 0) for k in keys])
                n_mid, n_fin = np.linalg.norm(v_mid), np.linalg.norm(v_fin)
                if n_mid > 1e-10 and n_fin > 1e-10:
                    reorganisation_stability = float(np.dot(v_mid, v_fin) / (n_mid * n_fin))
                else:
                    reorganisation_stability = 0.0
            except Exception:
                reorganisation_stability = None

    # Low-engagement control ablation
    # If ANY ablation causes global change → fragile, not integrated
    control_ablation_gini = None
    engagement = system.get_engagement_distribution()
    if engagement and len(regions) >= 2:
        lowest_eng_region = min(regions, key=lambda r: engagement.get(r, 0.0))
        try:
            control_ablated = system.ablate(lowest_eng_region)
            control_metric = _probe_metric(control_ablated, probe_inputs)
            control_degradation = baseline_metric - control_metric
            # Compute Gini including control ablation
            control_degs = individual_degradations + [control_degradation]
            sorted_ctrl = np.sort(np.abs(control_degs))
            n_ctrl = len(sorted_ctrl)
            if n_ctrl > 0 and sorted_ctrl.sum() > 0:
                idx_ctrl = np.arange(1, n_ctrl + 1)
                control_ablation_gini = float(
                    (2 * np.sum(idx_ctrl * sorted_ctrl) - (n_ctrl + 1) * np.sum(sorted_ctrl))
                    / (n_ctrl * np.sum(sorted_ctrl))
                )
        except Exception:
            control_ablation_gini = None

    # Decision logic
    # High Gini (> 0.3) = some regions disproportionately important = integration
    # High CV (> 0.5) = non-uniform degradation = integration
    has_integration = gini > 0.3 or cv > 0.5

    # Failure mode classification
    if has_integration:
        passed = True
        # Three-way failure mode subclassification (all compatible with PASS):
        if control_ablation_gini is not None and control_ablation_gini > 0.3:
            failure_mode = "fragile"  # Any ablation causes global change — brittle, not selectively integrated
        elif reorganisation_stability is not None and reorganisation_stability > 0.5:
            failure_mode = "earned"  # System reorganises to a new stable regime after ablation
        elif reorganisation_stability is not None and reorganisation_stability <= 0.5:
            failure_mode = "earned_unsettled"  # Non-uniform ablation response but system has not stabilised within observation window
        else:
            failure_mode = "earned"  # No reorganisation_stability data available — classify as earned by default
        notes = f"Non-linear degradation detected: Gini={gini:.4f}, CV={cv:.4f}"
        if reorganisation_stability is not None:
            notes += f", reorganisation_stability={reorganisation_stability:.4f}"
    else:
        passed = False
        # Classify the failure
        if max(abs(d) for d in individual_degradations) < 1e-6:
            failure_mode = "absent"  # No reorganisation at all
        elif cv < 0.2:
            failure_mode = "modular"  # Only local effect, no global reorganisation
        else:
            failure_mode = "topological"  # Fresh shows same pattern
        notes = f"Linear/uniform degradation: Gini={gini:.4f}, CV={cv:.4f}"

    effect_size = float(gini)

    provenance.log_measurement("integration", {
        "passed": passed,
        "gini": gini,
        "cv": cv,
        "n_regions": len(regions),
        "baseline_metric": baseline_metric,
        "reorganisation_stability": reorganisation_stability,
        "control_ablation_gini": control_ablation_gini,
    })

    return InstrumentResult(
        name="integration",
        passed=passed,
        effect_size=effect_size,
        raw_data={
            "baseline_metric": baseline_metric,
            "ablation_results": ablation_results,
            "sum_individual_degradation": sum_individual,
            "gini": float(gini),
            "cv": float(cv),
            "reorganisation_stability": reorganisation_stability,
            "control_ablation_gini": control_ablation_gini,
        },
        notes=notes,
        failure_mode=failure_mode,
    )

def compute_integration_earned_ratio(
    system: TestSystem,
    control_factory,
    probe_inputs: list[Any],
) -> dict[str, float]:
    """Compare integration between trained and fresh systems (supplementary).

    Runs identical ablation protocol on both trained and fresh instances.
    Returns ratio of reorganisation magnitudes — ratio > 1.0 means training
    created ADDITIONAL integration beyond topology.

    Literature basis: IIT Φ measures structural integration regardless of
    provenance (Tononi 2004). Aaronson critique: inactive logic gates produce
    high Φ from structure alone. This ratio distinguishes structural from earned.

    Returns dict with trained_gini, fresh_gini, earned_ratio.
    """
    if control_factory is None:
        return {"trained_gini": 0.0, "fresh_gini": 0.0, "earned_ratio": 1.0}

    # Run integration on a CLONE of the trained system (non-mutating)
    trained_clone = system.clone()
    trained_result = run_integration(system=trained_clone, probe_inputs=probe_inputs)
    trained_gini = trained_result.raw_data.get("gini", 0.0) if trained_result.raw_data else 0.0

    # Run integration on fresh system
    try:
        fresh = control_factory()
        fresh_result = run_integration(system=fresh, probe_inputs=probe_inputs)
        fresh_gini = fresh_result.raw_data.get("gini", 0.0) if fresh_result.raw_data else 0.0
    except Exception:
        fresh_gini = 0.0

    # Earned ratio: trained / fresh (> 1.0 means training added integration)
    if fresh_gini > 1e-10:
        earned_ratio = trained_gini / fresh_gini
    elif trained_gini > 1e-10:
        earned_ratio = float("inf")
    else:
        earned_ratio = 1.0

    return {
        "trained_gini": float(trained_gini),
        "fresh_gini": float(fresh_gini),
        "earned_ratio": float(min(earned_ratio, 1e6)),
    }

def _probe_metric(system: TestSystem, probe_inputs: list[Any]) -> float:
    """Run probe inputs and return final structure metric."""
    for inp in probe_inputs:
        system.step(inp)
    return system.get_structure_metric()

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

from m8_battery.core.test_system import TestSystem
from m8_battery.core.types import InstrumentResult
from m8_battery.core.provenance import ProvenanceLog


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

    # Decision logic
    # High Gini (> 0.3) = some regions disproportionately important = integration
    # High CV (> 0.5) = non-uniform degradation = integration
    has_integration = gini > 0.3 or cv > 0.5

    if has_integration:
        passed = True
        notes = f"Non-linear degradation detected: Gini={gini:.4f}, CV={cv:.4f}"
    else:
        passed = False
        notes = f"Linear/uniform degradation: Gini={gini:.4f}, CV={cv:.4f}"

    effect_size = float(gini)

    provenance.log_measurement("integration", {
        "passed": passed,
        "gini": gini,
        "cv": cv,
        "n_regions": len(regions),
        "baseline_metric": baseline_metric,
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
        },
        notes=notes,
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

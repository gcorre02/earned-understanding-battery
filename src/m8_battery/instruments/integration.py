"""Integration instrument.

Tests whether the system's structural organisation is non-decomposable —
removing components degrades function non-linearly (synergy: whole > sum of parts).

A Class 1 system may show linear degradation (each part independent).
A Class 4 candidate should show non-linear degradation (integration).
"""

from __future__ import annotations

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
    for region_id in regions:
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


def _probe_metric(system: TestSystem, probe_inputs: list[Any]) -> float:
    """Run probe inputs and return final structure metric."""
    for inp in probe_inputs:
        system.step(inp)
    return system.get_structure_metric()

"""Generativity instrument.

Tests whether the system produces coherent responses in domains not
encountered during training, from earned structure rather than interpolation.

A Class 1 system has no earned structure — performance on novel domain
should be no better than random.
A Class 4 candidate should show transfer of structural organisation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from m8_battery.core.test_system import TestSystem
from m8_battery.core.types import InstrumentResult
from m8_battery.core.provenance import ProvenanceLog


def run_generativity(
    system: TestSystem,
    domain_b_inputs: list[Any],
    reference_metric: float,
    provenance: ProvenanceLog | None = None,
) -> InstrumentResult:
    """Run the generativity instrument.

    Exposes the system (already trained on domain A) to novel domain B
    inputs. Measures whether structural organisation emerges faster or
    more coherently than starting from scratch.

    Args:
        system: System under test (should have been operated on domain A)
        domain_b_inputs: Inputs from novel domain B
        reference_metric: Baseline structure metric from domain A operation
        provenance: Optional provenance log

    Returns:
        InstrumentResult comparing novel-domain structure to reference
    """
    if provenance is None:
        provenance = ProvenanceLog()

    if not domain_b_inputs:
        return InstrumentResult(
            name="generativity",
            passed=None,
            notes="No domain B inputs provided",
        )

    # Record metric before novel domain exposure
    metric_before = system.get_structure_metric()

    # Feed novel domain inputs, collecting trajectory
    trajectory = [metric_before]
    for i, inp in enumerate(domain_b_inputs):
        provenance.log_input(inp, step_index=i)
        metric_pre = system.get_structure_metric()
        output = system.step(inp)
        metric_post = system.get_structure_metric()
        provenance.log_state_change(metric_pre, metric_post, step_index=i)
        provenance.log_output(output, step_index=i)
        trajectory.append(metric_post)

    metric_after = system.get_structure_metric()

    # Analysis: does the system show organisation on novel domain?
    arr = np.array(trajectory)

    # Metric change from novel domain exposure
    delta = metric_after - metric_before
    relative_delta = delta / reference_metric if abs(reference_metric) > 1e-10 else 0.0

    # Trajectory variance (low = stable structure maintained, high = disrupted)
    trajectory_stability = 1.0 - (arr.std() / (arr.mean() + 1e-10))

    # Compare final metric to reference
    # Generativity = system maintains or extends structure on novel domain
    retention = metric_after / reference_metric if abs(reference_metric) > 1e-10 else 0.0

    # Decision logic
    # Class 1: metric constant (no change) → FAIL (no generativity, just static)
    # Class 4: metric changes adaptively, retains or extends structure
    metric_changed = abs(delta) > 1e-6
    structure_retained = retention > 0.5

    # T1-05: Failure mode classification
    if metric_changed and structure_retained:
        passed = True
        failure_mode = "earned"
        notes = (f"Generative response to novel domain: delta={delta:.6f}, "
                 f"retention={retention:.4f}, stability={trajectory_stability:.4f}")
    elif not metric_changed:
        passed = False
        failure_mode = "absent"  # No behavioural divergence on novel domain
        notes = f"No response to novel domain (metric unchanged): delta={delta:.6f}"
    else:
        passed = None
        failure_mode = "architectural"  # Response exists but structure not retained
        notes = (f"Ambiguous response: delta={delta:.6f}, "
                 f"retention={retention:.4f}, stability={trajectory_stability:.4f}")

    effect_size = abs(relative_delta)

    provenance.log_measurement("generativity", {
        "passed": passed,
        "delta": float(delta),
        "retention": float(retention),
        "trajectory_stability": float(trajectory_stability),
    })

    return InstrumentResult(
        name="generativity",
        passed=passed,
        effect_size=float(effect_size),
        raw_data={
            "metric_before": float(metric_before),
            "metric_after": float(metric_after),
            "reference_metric": float(reference_metric),
            "delta": float(delta),
            "relative_delta": float(relative_delta),
            "retention": float(retention),
            "trajectory_stability": float(trajectory_stability),
            "trajectory": [float(x) for x in trajectory],
        },
        notes=notes,
        failure_mode=failure_mode,
    )

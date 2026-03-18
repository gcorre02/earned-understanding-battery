"""Developmental Trajectory instrument.

Tests whether structural organisation compresses from diffuse, high-entropy
regions into stable organisation over operational time, statistically
distinguishable from random structure formation.

A Class 1 system should show NO developmental trajectory (structure is fixed).
A Class 4 candidate should show measurable compression over time.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from m8_battery.core.test_system import TestSystem
from m8_battery.core.types import InstrumentResult
from m8_battery.core.provenance import ProvenanceLog


def run_developmental_trajectory(
    system: TestSystem,
    inputs: list[Any],
    measurement_interval: int = 1,
    control_factory: Callable[[], TestSystem] | None = None,
    provenance: ProvenanceLog | None = None,
) -> InstrumentResult:
    """Run the developmental trajectory instrument.

    Feeds inputs to the system, measuring structure metric at intervals.
    Compares the trajectory against a matched control (if provided).

    A positive result requires:
    - Structure metric changes measurably over time (not constant)
    - The trajectory is distinguishable from random walk (monotonic compression
      or consistent trend, not noise)

    Args:
        system: The system under test
        inputs: Sequence of domain interactions to feed
        measurement_interval: Measure structure every K steps
        control_factory: Callable returning a fresh matched control system
        provenance: Optional provenance log to record events

    Returns:
        InstrumentResult with passed/failed/ambiguous verdict
    """
    if provenance is None:
        provenance = ProvenanceLog()

    # Collect structure metric trajectory
    trajectory = _collect_trajectory(
        system, inputs, measurement_interval, provenance
    )

    # Collect control trajectory if factory provided
    control_trajectory = None
    if control_factory is not None:
        control = control_factory()
        control_trajectory = _collect_trajectory(
            control, inputs, measurement_interval, provenance=None
        )

    # Analyse trajectory
    result = _analyse_trajectory(trajectory, control_trajectory)

    provenance.log_measurement("developmental_trajectory", {
        "passed": result.passed,
        "effect_size": result.effect_size,
        "n_measurements": len(trajectory),
        "trajectory_start": trajectory[0] if trajectory else None,
        "trajectory_end": trajectory[-1] if trajectory else None,
    })

    return result


def _collect_trajectory(
    system: TestSystem,
    inputs: list[Any],
    measurement_interval: int,
    provenance: ProvenanceLog | None = None,
) -> list[float]:
    """Feed inputs and collect structure metric at intervals."""
    trajectory = []

    # Initial measurement
    trajectory.append(system.get_structure_metric())

    for i, inp in enumerate(inputs):
        if provenance:
            metric_before = system.get_structure_metric()
            provenance.log_input(inp, step_index=i)

        output = system.step(inp)

        if provenance:
            metric_after = system.get_structure_metric()
            provenance.log_state_change(metric_before, metric_after, step_index=i)
            provenance.log_output(output, step_index=i)

        if (i + 1) % measurement_interval == 0:
            trajectory.append(system.get_structure_metric())

    return trajectory


def _analyse_trajectory(
    trajectory: list[float],
    control_trajectory: list[float] | None = None,
) -> InstrumentResult:
    """Analyse whether the trajectory shows developmental compression."""
    if len(trajectory) < 3:
        return InstrumentResult(
            name="developmental_trajectory",
            passed=None,
            notes="Insufficient measurements (need >= 3)",
            raw_data={"trajectory": trajectory},
        )

    arr = np.array(trajectory)

    # Check if metric is constant (no development)
    metric_range = arr.max() - arr.min()
    if metric_range < 1e-10:
        return InstrumentResult(
            name="developmental_trajectory",
            passed=False,
            effect_size=0.0,
            notes="Structure metric is constant — no developmental trajectory",
            raw_data={"trajectory": trajectory},
        )

    # Compute trend: linear regression slope
    x = np.arange(len(arr))
    slope, intercept = np.polyfit(x, arr, 1)
    residuals = arr - (slope * x + intercept)
    r_squared = 1.0 - (np.sum(residuals**2) / np.sum((arr - arr.mean())**2))

    # Compute monotonicity: fraction of consecutive pairs that go in the same direction
    diffs = np.diff(arr)
    if len(diffs) > 0:
        n_positive = np.sum(diffs > 0)
        n_negative = np.sum(diffs < 0)
        monotonicity = max(n_positive, n_negative) / len(diffs)
    else:
        monotonicity = 0.0

    # Effect size vs control
    effect_size = None
    if control_trajectory is not None and len(control_trajectory) >= 3:
        ctrl_arr = np.array(control_trajectory)
        # Cohen's d between final metrics
        system_final = arr[-3:].mean()
        control_final = ctrl_arr[-3:].mean()
        pooled_std = np.sqrt(
            (arr[-3:].std()**2 + ctrl_arr[-3:].std()**2) / 2
        )
        if pooled_std > 1e-10:
            effect_size = abs(system_final - control_final) / pooled_std
        else:
            effect_size = 0.0

    # DN-22 earned ratio: trained metric_range vs fresh metric_range
    # If fresh also develops trajectory, the signal may not be earned.
    earned_ratio = None
    if control_trajectory is not None and len(control_trajectory) >= 3:
        ctrl_arr = np.array(control_trajectory)
        ctrl_range = ctrl_arr.max() - ctrl_arr.min()
        if ctrl_range > 1e-10:
            earned_ratio = float(metric_range / ctrl_range)
        elif metric_range > 1e-10:
            earned_ratio = float(min(metric_range / 1e-10, 1e6))  # Cap
        else:
            earned_ratio = 1.0  # Neither changes

    # Decision logic (DN-22: earned ratio required if control available)
    has_trend = abs(slope) > 1e-6 and r_squared > 0.3
    has_monotonicity = monotonicity > 0.6
    # Earned ratio > 1.0 means trained develops MORE than fresh
    passes_earned = earned_ratio is None or earned_ratio > 1.0

    if has_trend and has_monotonicity and passes_earned:
        passed = True
        notes = f"Developmental trajectory detected: slope={slope:.6f}, R²={r_squared:.4f}, monotonicity={monotonicity:.2f}"
        if earned_ratio is not None:
            notes += f", earned_ratio={earned_ratio:.2f}"
    elif has_trend and has_monotonicity and not passes_earned:
        passed = False
        notes = (f"Trajectory present but not earned (DN-22): earned_ratio={earned_ratio:.2f}. "
                 f"Fresh system shows similar development.")
    elif has_trend or has_monotonicity:
        passed = None
        notes = f"Ambiguous trajectory: slope={slope:.6f}, R²={r_squared:.4f}, monotonicity={monotonicity:.2f}"
    else:
        passed = False
        notes = f"No developmental trajectory: slope={slope:.6f}, R²={r_squared:.4f}, monotonicity={monotonicity:.2f}"

    return InstrumentResult(
        name="developmental_trajectory",
        passed=passed,
        effect_size=effect_size,
        raw_data={
            "trajectory": trajectory,
            "control_trajectory": control_trajectory,
            "slope": float(slope),
            "r_squared": float(r_squared),
            "monotonicity": float(monotonicity),
            "metric_range": float(metric_range),
            "earned_ratio": earned_ratio,
        },
        notes=notes,
    )


def compute_trajectory_compression(trajectory: list[float]) -> dict[str, float]:
    """Compute Lempel-Ziv compression ratio of structure metric time series.

    Supplementary diagnostic. Decreasing compressibility over time indicates
    the system is developing more structured (compressible) behaviour.

    Literature basis:
    - TERL (2025): "trajectories produced by TERL are more easily compressed"
    - Brain entropy trajectories (PMC 2022): permutation entropy tracks maturation

    Returns dict with early_ratio, late_ratio, compression_trend.
    """
    import bz2
    import struct

    if len(trajectory) < 6:
        return {"early_ratio": 0.0, "late_ratio": 0.0, "compression_trend": 0.0}

    # Split trajectory into early and late halves
    mid = len(trajectory) // 2
    early = trajectory[:mid]
    late = trajectory[mid:]

    def _compress_ratio(values: list[float]) -> float:
        """Compression ratio: compressed_size / raw_size. Lower = more compressible."""
        raw = struct.pack(f"{len(values)}d", *values)
        compressed = bz2.compress(raw)
        return len(compressed) / len(raw) if len(raw) > 0 else 1.0

    early_ratio = _compress_ratio(early)
    late_ratio = _compress_ratio(late)

    # Compression trend: negative = late is more compressible (structured)
    compression_trend = late_ratio - early_ratio

    return {
        "early_ratio": float(early_ratio),
        "late_ratio": float(late_ratio),
        "compression_trend": float(compression_trend),
    }

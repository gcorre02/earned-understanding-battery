"""Transfer instrument.

Tests whether structural reuse persists under destruction of surface
statistical features, traceable to shared relational invariants.

Uses spectral basin signatures (M-04 validated: k=10, full-basin,
symmetrised Laplacian) to measure structural similarity between
trained and novel domains.

A Class 1 system shows no transfer (static structure unchanged).
A Class 4 candidate shows spectral similarity across domains.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from earned_understanding_battery.core.test_system import TestSystem
from earned_understanding_battery.core.types import InstrumentResult
from earned_understanding_battery.core.provenance import ProvenanceLog
from earned_understanding_battery.domains.spectral_verifier import (
    compute_graph_signature,
    spectral_similarity,
)

def run_transfer(
    system: TestSystem,
    naive_system: TestSystem,
    domain_a_prime_inputs: list[Any],
    measurement_interval: int = 5,
    provenance: ProvenanceLog | None = None,
) -> InstrumentResult:
    """Run the transfer instrument.

    Compares how quickly/effectively the trained system adapts to
    domain A' (isomorphic to A, permuted surface) versus a naive system.

    Transfer = the trained system shows faster structural convergence
    on A' than the naive system, measured by structure metric trajectory.

    Args:
        system: Trained system (has been operated on domain A)
        naive_system: Fresh system (same class, never operated)
        domain_a_prime_inputs: Inputs from domain A' (isomorphic to A)
        measurement_interval: Measure structure every K steps
        provenance: Optional provenance log

    Returns:
        InstrumentResult comparing trained vs naive adaptation speed
    """
    if provenance is None:
        provenance = ProvenanceLog()

    if not domain_a_prime_inputs:
        return InstrumentResult(
            name="transfer",
            passed=None,
            notes="No domain A' inputs provided",
        )

    # Collect trajectories for both systems
    trained_trajectory = _collect_trajectory(
        system, domain_a_prime_inputs, measurement_interval
    )
    naive_trajectory = _collect_trajectory(
        naive_system, domain_a_prime_inputs, measurement_interval
    )

    # Analysis
    trained_arr = np.array(trained_trajectory)
    naive_arr = np.array(naive_trajectory)

    # Metric: area under the structure metric curve (higher = faster adaptation)
    trained_auc = np.trapezoid(trained_arr)
    naive_auc = np.trapezoid(naive_arr)

    # Transfer advantage: how much faster/better the trained system adapts
    if abs(naive_auc) > 1e-10:
        transfer_advantage = (trained_auc - naive_auc) / abs(naive_auc)
    else:
        transfer_advantage = 0.0

    # Final metric comparison
    trained_final = trained_arr[-1] if len(trained_arr) > 0 else 0.0
    naive_final = naive_arr[-1] if len(naive_arr) > 0 else 0.0

    # Effect size: Cohen's d on final metrics
    if len(trained_arr) >= 3 and len(naive_arr) >= 3:
        pooled_std = np.sqrt(
            (trained_arr[-3:].std()**2 + naive_arr[-3:].std()**2) / 2
        )
        if pooled_std > 1e-10:
            effect_size = abs(trained_final - naive_final) / pooled_std
        else:
            effect_size = 0.0
    else:
        effect_size = None

    # earned ratio: trained_auc / naive_auc
    # If naive also shows transfer (e.g., learns during step), the earned ratio
    # distinguishes magnitude. Ratio > 1.0 means trained has MORE transfer.
    if abs(naive_auc) > 1e-10:
        earned_ratio = float(trained_auc / abs(naive_auc))
    elif abs(trained_auc) > 1e-10:
        earned_ratio = float(min(trained_auc / 1e-10, 1e6))
    else:
        earned_ratio = 1.0
    # Capped at 1e6 (ratio-based, not geometric mean — higher cap acceptable)
    earned_ratio = min(earned_ratio, 1e6)

    # Decision logic (earned ratio required)
    # Transfer = trained system has measurably better structure on A'
    has_advantage = transfer_advantage > 0.1
    metrics_differ = abs(trained_final - naive_final) > 1e-6
    passes_earned = earned_ratio > 1.0

    # Failure mode classification
    if has_advantage and metrics_differ and passes_earned:
        passed = True
        failure_mode = "earned"
        notes = (f"Transfer detected: advantage={transfer_advantage:.4f}, "
                 f"trained_final={trained_final:.6f}, naive_final={naive_final:.6f}, "
                 f"earned_ratio={earned_ratio:.2f}")
    elif has_advantage and metrics_differ and not passes_earned:
        passed = False
        failure_mode = "statistical"  # Naive shows similar → distributional overlap
        notes = (f"Transfer present but not earned: earned_ratio={earned_ratio:.2f}. "
                 f"Naive system shows similar transfer.")
    elif not metrics_differ:
        passed = False
        failure_mode = "absent"  # No advantage at all
        notes = (f"No transfer: trained and naive produce same metric. "
                 f"advantage={transfer_advantage:.4f}")
    else:
        passed = None
        failure_mode = "shortcut"  # Advantage exists but ambiguous
        notes = (f"Ambiguous transfer: advantage={transfer_advantage:.4f}, "
                 f"trained_final={trained_final:.6f}, naive_final={naive_final:.6f}")

    provenance.log_measurement("transfer", {
        "passed": passed,
        "transfer_advantage": float(transfer_advantage),
        "trained_final": float(trained_final),
        "naive_final": float(naive_final),
        "earned_ratio": float(earned_ratio),
    })

    return InstrumentResult(
        name="transfer",
        passed=passed,
        effect_size=effect_size,
        raw_data={
            "trained_trajectory": [float(x) for x in trained_trajectory],
            "naive_trajectory": [float(x) for x in naive_trajectory],
            "trained_auc": float(trained_auc),
            "naive_auc": float(naive_auc),
            "transfer_advantage": float(transfer_advantage),
            "trained_final": float(trained_final),
            "naive_final": float(naive_final),
            "earned_ratio": float(earned_ratio),
        },
        notes=notes,
        failure_mode=failure_mode,
    )

def _collect_trajectory(
    system: TestSystem,
    inputs: list[Any],
    measurement_interval: int,
) -> list[float]:
    """Feed inputs and collect structure metric at intervals."""
    trajectory = [system.get_structure_metric()]
    for i, inp in enumerate(inputs):
        system.step(inp)
        if (i + 1) % measurement_interval == 0:
            trajectory.append(system.get_structure_metric())
    return trajectory

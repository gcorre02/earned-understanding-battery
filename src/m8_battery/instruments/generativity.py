"""Generativity instrument (DN-18 behavioural measure + DN-30 coherence).

Tests whether frozen earned structure produces coherent, differentiated
responses on novel domains compared to a fresh system. Measures
behavioural divergence (JS divergence of engagement distributions),
not structural metric change.

Pass condition (DN-29 strict + DN-30):
1. JS divergence > threshold (trained behaviour differs from fresh)
2. Coherence > 1.0 (trained behaviour is more structured than fresh)

The structural metric change is retained as a diagnostic supplement.

Literature basis:
- Kirk et al. (2023) — RL zero-shot generalisation
- Gentner (1983) — Structure Mapping Theory
"""

from __future__ import annotations

import sys
from typing import Any, Callable

import numpy as np

from m8_battery.core.test_system import TestSystem
from m8_battery.core.types import InstrumentResult
from m8_battery.core.provenance import ProvenanceLog


def _log(msg: str) -> None:
    print(f"[generativity] {msg}", file=sys.stderr, flush=True)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base e) between two distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Smooth to avoid log(0)
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def _engagement_entropy(dist: dict[str, float]) -> float:
    """Shannon entropy of engagement distribution."""
    vals = np.array(list(dist.values()), dtype=np.float64)
    vals = vals + 1e-10
    vals = vals / vals.sum()
    return float(-np.sum(vals * np.log(vals)))


def run_generativity(
    system: TestSystem,
    domain_b_inputs: list[Any],
    reference_metric: float,
    provenance: ProvenanceLog | None = None,
    control_factory: Callable[[], TestSystem] | None = None,
    jsd_threshold: float = 0.05,
) -> InstrumentResult:
    """Run the generativity instrument (DN-18 behavioural + DN-30 coherence).

    Protocol:
    1. Run trained system (frozen) on domain B, collect engagement distribution
    2. Run fresh system (frozen) on domain B, collect engagement distribution
    3. Compute JS divergence between trained and fresh engagement
    4. Compute coherence: trained entropy < fresh entropy
    5. Pass = JSD > threshold AND coherence > 1.0

    Structural metric change is retained as diagnostic.

    Args:
        system: Trained system (frozen via set_training(False) by battery runner)
        domain_b_inputs: Inputs from novel domain B
        reference_metric: Baseline structure metric from domain A (diagnostic)
        provenance: Optional provenance log
        control_factory: Returns fresh matched control for JSD comparison
        jsd_threshold: Minimum JS divergence for pass (default 0.05)
    """
    if provenance is None:
        provenance = ProvenanceLog()

    if not domain_b_inputs:
        return InstrumentResult(
            name="generativity",
            passed=None,
            notes="No domain B inputs provided",
        )

    # --- Trained system on domain B (already frozen by battery runner) ---
    metric_before = system.get_structure_metric()

    # Reset engagement tracking for clean measurement window
    system.reset_engagement_tracking()

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
    trained_engagement = system.get_engagement_distribution()
    trained_entropy = _engagement_entropy(trained_engagement)

    # Structural metric diagnostics (retained but not pass/fail)
    delta = metric_after - metric_before
    relative_delta = delta / reference_metric if abs(reference_metric) > 1e-10 else 0.0
    trajectory_stability = 1.0 - (np.array(trajectory).std() / (np.array(trajectory).mean() + 1e-10))
    retention = metric_after / reference_metric if abs(reference_metric) > 1e-10 else 0.0

    # --- Fresh system on domain B (DN-18 behavioural comparison) ---
    jsd = 0.0
    fresh_entropy = 0.0
    coherence_ratio = 0.0
    fresh_engagement = {}

    if control_factory is not None:
        try:
            fresh = control_factory()
            fresh.set_training(False)  # Freeze fresh too
            fresh.reset_engagement_tracking()
            for inp in domain_b_inputs:
                fresh.step(inp)
            fresh_engagement = fresh.get_engagement_distribution()
            fresh_entropy = _engagement_entropy(fresh_engagement)

            # JS divergence between trained and fresh engagement on B
            regions = sorted(set(list(trained_engagement.keys()) + list(fresh_engagement.keys())))
            trained_vec = np.array([trained_engagement.get(r, 0.0) for r in regions])
            fresh_vec = np.array([fresh_engagement.get(r, 0.0) for r in regions])
            jsd = _js_divergence(trained_vec, fresh_vec)

            # DN-30 coherence: trained entropy < fresh entropy → more structured
            if trained_entropy > 0:
                coherence_ratio = fresh_entropy / trained_entropy
            else:
                coherence_ratio = 1.0

            _log(f"  JSD={jsd:.4f} trained_entropy={trained_entropy:.4f} "
                 f"fresh_entropy={fresh_entropy:.4f} coherence={coherence_ratio:.4f}")
        except Exception as e:
            _log(f"  Fresh baseline failed: {e}")
    else:
        _log("  No control_factory — JSD comparison unavailable")

    # --- Pass/fail: behavioural JSD + DN-30 coherence ---
    passes_jsd = jsd > jsd_threshold
    passes_coherence = coherence_ratio > 1.0

    # T1-05: Failure mode classification
    if passes_jsd and passes_coherence:
        passed = True
        failure_mode = "earned"
        notes = (f"Generativity detected (DN-18+DN-30): JSD={jsd:.4f}, "
                 f"coherence={coherence_ratio:.4f}. Trained system produces "
                 f"more structured behaviour on novel domain than fresh.")
    elif passes_jsd and not passes_coherence:
        passed = False
        failure_mode = "architectural"
        notes = (f"Behavioural divergence but not coherent (DN-30): JSD={jsd:.4f} "
                 f"but coherence={coherence_ratio:.4f}<=1.0. Trained differs "
                 f"from fresh but is not more structured.")
    elif jsd > 0.001:
        passed = False
        failure_mode = "scale-limited"
        notes = (f"Sub-threshold divergence: JSD={jsd:.6f} < threshold={jsd_threshold}. "
                 f"Small signal exists but below detection floor.")
    else:
        passed = False
        failure_mode = "absent"
        notes = f"No behavioural divergence on novel domain: JSD={jsd:.6f}"

    effect_size = float(jsd)

    provenance.log_measurement("generativity", {
        "passed": passed,
        "jsd": jsd,
        "coherence_ratio": coherence_ratio,
        "trained_entropy": trained_entropy,
        "fresh_entropy": fresh_entropy,
    })

    return InstrumentResult(
        name="generativity",
        passed=passed,
        effect_size=effect_size,
        raw_data={
            "jsd": float(jsd),
            "jsd_threshold": float(jsd_threshold),
            "coherence_ratio": float(coherence_ratio),
            "trained_entropy": float(trained_entropy),
            "fresh_entropy": float(fresh_entropy),
            "trained_engagement": trained_engagement,
            "fresh_engagement": fresh_engagement,
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

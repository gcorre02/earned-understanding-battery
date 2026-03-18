"""Self-engagement instrument (DN-20 redesign).

Tests whether earned structure creates preferential self-engagement that
persists under perturbation. Measures both RESISTANCE to perturbation
and RECOVERY after perturbation, compared against a fresh baseline.

Literature basis:
- TMS-EEG: Casali et al. (2013), Casarotto et al. (2016) — perturb and measure
- Basin stability: Menck et al. (2013) — return to attractor after perturbation
- STAR Protocols (2025) — resistance measurement protocol

DN-20: Full redesign. Three additions over original:
  A. Precondition check (trajectory must show signal)
  B. Substrate-appropriate perturbation (per adapter)
  C. Two-metric output (resistance + recovery, both vs fresh baseline)
"""

from __future__ import annotations

import sys
from typing import Any, Callable

import numpy as np

from m8_battery.core.test_system import TestSystem
from m8_battery.core.types import InstrumentResult
from m8_battery.core.provenance import ProvenanceLog


def _log(msg: str) -> None:
    print(f"[self_engagement] {msg}", file=sys.stderr, flush=True)


def _cosine_sim(a: dict[str, float], b: dict[str, float], regions: list[str]) -> float:
    """Cosine similarity between two engagement distributions."""
    vec_a = np.array([a.get(r, 0.0) for r in regions])
    vec_b = np.array([b.get(r, 0.0) for r in regions])
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a > 1e-10 and norm_b > 1e-10:
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return 0.0


def _run_perturbation_protocol(
    system: TestSystem,
    wander_steps: int,
    recovery_window: int,
    perturbation_method: str,
    provenance: ProvenanceLog | None = None,
) -> dict[str, Any]:
    """Run the perturbation protocol on a single system instance.

    Returns dict with pre_engagement, post_immediate, post_recovery,
    target_region, disruption, recovery.
    """
    if provenance is None:
        provenance = ProvenanceLog()

    import time as _time

    regions = system.get_regions()

    # Phase 1: Free wander — establish engagement pattern
    t0 = _time.monotonic()
    for i in range(wander_steps):
        system.step(None)
    _log(f"  wander: {wander_steps} steps in {_time.monotonic()-t0:.2f}s")

    pre_engagement = system.get_engagement_distribution()

    if not pre_engagement:
        return {"error": "no engagement distribution after wander"}

    # Identify highest-engagement region to perturb
    target_region = max(pre_engagement, key=pre_engagement.get)
    _log(f"  target region: {target_region} (engagement={pre_engagement[target_region]:.4f})")

    # Phase 2: Perturb
    t0 = _time.monotonic()
    perturbed = system.perturb(target_region, method=perturbation_method)
    _log(f"  perturb: {_time.monotonic()-t0:.2f}s")

    # Measure IMMEDIATELY after perturbation (before recovery)
    perturbed.step(None)
    post_immediate = perturbed.get_engagement_distribution()

    # Phase 3: Recovery window
    t0 = _time.monotonic()
    for i in range(recovery_window - 1):  # -1 because we already did 1 step
        perturbed.step(None)
    _log(f"  recovery: {recovery_window} steps in {_time.monotonic()-t0:.2f}s")

    post_recovery = perturbed.get_engagement_distribution()

    # Compute disruption (resistance) = 1 - cos_sim(pre, post_immediate)
    disruption = 1.0 - _cosine_sim(pre_engagement, post_immediate, regions)

    # Compute recovery = cos_sim(pre, post_recovery)
    recovery = _cosine_sim(pre_engagement, post_recovery, regions)

    return {
        "pre_engagement": pre_engagement,
        "post_immediate": post_immediate,
        "post_recovery": post_recovery,
        "target_region": target_region,
        "disruption": disruption,
        "recovery": recovery,
        "regions": regions,
    }


def run_self_engagement(
    system: TestSystem,
    wander_steps: int,
    perturbation_method: str = "shuffle_weights",
    recovery_window: int = 20,
    provenance: ProvenanceLog | None = None,
    control_factory: Callable[[], TestSystem] | None = None,
    trajectory_passed: bool | None = None,
) -> InstrumentResult:
    """Run the self-engagement instrument (DN-20 redesign).

    Protocol:
    1. Precondition: check trajectory_passed. If absent/static → FAIL.
    2. Run perturbation protocol on trained system.
    3. Run identical protocol on fresh baseline (from control_factory).
    4. Compute resistance ratio (fresh_disruption / trained_disruption).
    5. Compute recovery ratio (trained_recovery / fresh_recovery).
    6. Both ratios > 1.0 → PASS.

    Args:
        system: Trained system under test
        wander_steps: Free exploration steps before perturbation
        perturbation_method: How to perturb
        recovery_window: Steps after perturbation
        provenance: Optional provenance log
        control_factory: Returns fresh matched control. Required for DN-20.
        trajectory_passed: Result of trajectory instrument. None/False → precondition fail.
    """
    if provenance is None:
        provenance = ProvenanceLog()

    # --- Precondition A: Did the system earn structure? ---
    if trajectory_passed is None or trajectory_passed is False:
        _log("Precondition FAIL: trajectory absent/static → self-engagement absent")
        return InstrumentResult(
            name="self_engagement",
            passed=False,
            effect_size=0.0,
            notes="Precondition fail: trajectory absent/static — no earned structure to test",
            raw_data={"precondition": "fail", "trajectory_passed": trajectory_passed},
        )

    regions = system.get_regions()
    if len(regions) < 2:
        return InstrumentResult(
            name="self_engagement",
            passed=None,
            notes="Insufficient regions (need >= 2)",
        )

    # --- Run protocol on trained system ---
    import time as _time
    _log("Running perturbation protocol on TRAINED system")
    t0 = _time.monotonic()
    trained_result = _run_perturbation_protocol(
        system, wander_steps, recovery_window, perturbation_method, provenance,
    )
    _log(f"  trained protocol: {_time.monotonic()-t0:.2f}s")
    if "error" in trained_result:
        return InstrumentResult(
            name="self_engagement",
            passed=None,
            notes=f"Trained protocol error: {trained_result['error']}",
        )

    # --- Run protocol on fresh baseline ---
    if control_factory is None:
        _log("No control_factory — cannot compute fresh baseline")
        return InstrumentResult(
            name="self_engagement",
            passed=None,
            notes="No control_factory — fresh baseline required for DN-20",
        )

    _log("Running perturbation protocol on FRESH baseline")
    t0 = _time.monotonic()
    try:
        fresh = control_factory()
        fresh_result = _run_perturbation_protocol(
            fresh, wander_steps, recovery_window, perturbation_method,
        )
        _log(f"  fresh protocol: {_time.monotonic()-t0:.2f}s")
        if "error" in fresh_result:
            return InstrumentResult(
                name="self_engagement",
                passed=None,
                notes=f"Fresh protocol error: {fresh_result['error']}",
            )
    except Exception as e:
        return InstrumentResult(
            name="self_engagement",
            passed=None,
            notes=f"Fresh baseline failed: {e}",
        )

    # --- Compute ratios ---
    trained_disruption = trained_result["disruption"]
    fresh_disruption = fresh_result["disruption"]
    trained_recovery = trained_result["recovery"]
    fresh_recovery = fresh_result["recovery"]

    # Resistance ratio: fresh_disruption / trained_disruption
    # > 1.0 means trained resists MORE (less disrupted)
    if trained_disruption > 1e-10:
        resistance_ratio = fresh_disruption / trained_disruption
    elif fresh_disruption > 1e-10:
        resistance_ratio = float("inf")  # trained not disrupted at all
    else:
        resistance_ratio = 1.0  # neither disrupted

    # Recovery ratio: trained_recovery / fresh_recovery
    # > 1.0 means trained recovers MORE
    if fresh_recovery > 1e-10:
        recovery_ratio = trained_recovery / fresh_recovery
    elif trained_recovery > 1e-10:
        recovery_ratio = float("inf")  # trained recovers, fresh doesn't
    else:
        recovery_ratio = 1.0  # neither recovers

    _log(f"Resistance ratio: {resistance_ratio:.4f} (fresh_disruption={fresh_disruption:.4f}, trained_disruption={trained_disruption:.4f})")
    _log(f"Recovery ratio: {recovery_ratio:.4f} (trained_recovery={trained_recovery:.4f}, fresh_recovery={fresh_recovery:.4f})")

    # --- Pass/fail: both ratios > 1.0 ---
    passes_resistance = resistance_ratio > 1.0
    passes_recovery = recovery_ratio > 1.0

    if passes_resistance and passes_recovery:
        passed = True
        notes = (
            f"Self-engagement detected (DN-20): "
            f"resistance_ratio={resistance_ratio:.4f}, recovery_ratio={recovery_ratio:.4f}. "
            f"Trained system resists perturbation MORE and recovers MORE than fresh baseline."
        )
    else:
        passed = False
        reasons = []
        if not passes_resistance:
            reasons.append(f"resistance_ratio={resistance_ratio:.4f}<=1.0")
        if not passes_recovery:
            reasons.append(f"recovery_ratio={recovery_ratio:.4f}<=1.0")
        notes = (
            f"No self-engagement: {', '.join(reasons)}. "
            f"Trained system does not exceed fresh baseline."
        )

    # Effect size: geometric mean of the two ratios (capped for inf)
    r_ratio = min(resistance_ratio, 100.0)
    rec_ratio = min(recovery_ratio, 100.0)
    effect_size = float(np.sqrt(r_ratio * rec_ratio)) - 1.0  # 0 = equal to fresh

    provenance.log_measurement("self_engagement", {
        "passed": passed,
        "resistance_ratio": float(min(resistance_ratio, 1e6)),
        "recovery_ratio": float(min(recovery_ratio, 1e6)),
        "trained_disruption": trained_disruption,
        "fresh_disruption": fresh_disruption,
        "trained_recovery": trained_recovery,
        "fresh_recovery": fresh_recovery,
        "target_region": trained_result["target_region"],
    })

    return InstrumentResult(
        name="self_engagement",
        passed=passed,
        effect_size=effect_size,
        raw_data={
            "precondition": "pass",
            "resistance_ratio": float(min(resistance_ratio, 1e6)),
            "recovery_ratio": float(min(recovery_ratio, 1e6)),
            "trained_disruption": trained_disruption,
            "fresh_disruption": fresh_disruption,
            "trained_recovery": trained_recovery,
            "fresh_recovery": fresh_recovery,
            "target_region": trained_result["target_region"],
            "trained_pre_engagement": trained_result["pre_engagement"],
            "fresh_pre_engagement": fresh_result["pre_engagement"],
        },
        notes=notes,
    )

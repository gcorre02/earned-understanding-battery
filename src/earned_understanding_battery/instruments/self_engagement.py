"""Self-engagement instrument (redesign).

Tests whether earned structure creates preferential self-engagement that
persists under perturbation. Measures both RESISTANCE to perturbation
and RECOVERY after perturbation, compared against a fresh baseline.

Literature basis:
- TMS-EEG: Casali et al. (2013), Casarotto et al. (2016) — perturb and measure
- Basin stability: Menck et al. (2013) — return to attractor after perturbation
- STAR Protocols (2025) — resistance measurement protocol

Full redesign. Three additions over original:
  A. Precondition check (trajectory must show signal)
  B. Substrate-appropriate perturbation (per adapter)
  C. Two-metric output (resistance + recovery, both vs fresh baseline)
"""

from __future__ import annotations

import sys
from typing import Any, Callable

import numpy as np

from earned_understanding_battery.core.test_system import TestSystem
from earned_understanding_battery.core.types import InstrumentResult
from earned_understanding_battery.core.provenance import ProvenanceLog

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
    # Reset engagement tracking so we measure THIS window, not cumulative history
    system.reset_engagement_tracking()
    t0 = _time.monotonic()
    for i in range(wander_steps):
        system.step(None)
    _log(f"  wander: {wander_steps} steps in {_time.monotonic()-t0:.2f}s")

    pre_engagement = system.get_engagement_distribution()

    if not pre_engagement or all(v < 1e-10 for v in pre_engagement.values()):
        return {"error": "degenerate engagement distribution after wander"}

    # Target highest-STRUCTURE region, not highest-engagement.
    # SBM generates homogeneous communities, so the highest-engagement
    # region rarely has elevated structure. Targeting by structure directly
    # asks: "does the system's EARNED structure resist perturbation?"
    pre_structure = system.get_structure_distribution()

    if not pre_structure or all(v < 1e-10 for v in pre_structure.values()):
        return {"error": "degenerate structure distribution after wander"}

    target_region = max(pre_structure, key=pre_structure.get)
    _log(f"  target region: {target_region} (structure={pre_structure[target_region]:.4f}, "
         f"engagement={pre_engagement.get(target_region, 0.0):.4f})")

    # Perturbation validation gate
    target_structure_pre = pre_structure.get(target_region, 0.0)
    non_target_structures = [v for k, v in pre_structure.items() if k != target_region]
    mean_non_target = sum(non_target_structures) / max(len(non_target_structures), 1)
    perturbation_validated = True
    perturbation_caveat = None

    # Phase 2: Perturb
    t0 = _time.monotonic()
    perturbed = system.perturb(target_region, method=perturbation_method)
    _log(f"  perturb: {_time.monotonic()-t0:.2f}s")

    # Verify perturbation reduced target structure
    post_structure = perturbed.get_structure_distribution()
    target_structure_post = post_structure.get(target_region, 0.0)

    if target_structure_pre <= mean_non_target:
        perturbation_caveat = (f"Perturbation validation: target region {target_region} structure "
                               f"({target_structure_pre:.4f}) not elevated vs "
                               f"non-target mean ({mean_non_target:.4f})")
        perturbation_validated = False
        _log(f"  PRECONDITION FAILED: {perturbation_caveat}")
    elif target_structure_post >= target_structure_pre:
        perturbation_caveat = (f"Perturbation validation: perturbation did not reduce target structure "
                               f"(pre={target_structure_pre:.4f}, post={target_structure_post:.4f})")
        perturbation_validated = False
        _log(f"  PRECONDITION FAILED: {perturbation_caveat}")
    else:
        _log(f"  validated: target structure {target_structure_pre:.4f} → "
             f"{target_structure_post:.4f} (non-target mean={mean_non_target:.4f})")

    # Option C: return indeterminate when preconditions fail
    if not perturbation_validated:
        return {
            "indeterminate": True,
            "perturbation_caveat": perturbation_caveat,
            "target_region": target_region,
            "target_structure_pre": target_structure_pre,
            "target_structure_post": target_structure_post,
            "mean_non_target": mean_non_target,
        }

    # Measure IMMEDIATELY after perturbation — windowed
    perturbed.reset_engagement_tracking()
    perturbed.step(None)
    post_immediate = perturbed.get_engagement_distribution()

    # False-attractor control
    # Boost a non-target region to create a decoy attractor. If the system
    # reconstructs the ORIGINAL pattern instead of drifting to the decoy,
    # that's stronger evidence of earned self-engagement.
    decoy_region = None
    original_recovery_ratio = None
    decoy_drift_ratio = None
    non_target_regions = [r for r in regions if r != target_region]
    if non_target_regions:
        # Pick lowest-engagement region as decoy
        decoy_region = min(non_target_regions, key=lambda r: pre_engagement.get(r, 0.0))
        boosted = perturbed.boost(decoy_region)

        # Run recovery on the boosted system
        boosted.reset_engagement_tracking()
        for _ in range(recovery_window):
            boosted.step(None)
        boosted_recovery = boosted.get_engagement_distribution()

        # Compare: did the system return to original target or drift to decoy?
        pre_target = pre_engagement.get(target_region, 0.0)
        rec_target = boosted_recovery.get(target_region, 0.0)
        pre_decoy = pre_engagement.get(decoy_region, 0.0)
        rec_decoy = boosted_recovery.get(decoy_region, 0.0)

        original_recovery_ratio = rec_target / max(pre_target, 1e-10)
        decoy_drift_ratio = rec_decoy / max(pre_decoy, 1e-10)

        prefers_original = original_recovery_ratio > decoy_drift_ratio
        _log(f"  decoy={decoy_region} original_recovery={original_recovery_ratio:.4f} "
             f"decoy_drift={decoy_drift_ratio:.4f} prefers_original={prefers_original}")

    # Phase 3: Recovery horizon family
    # Measure recovery at multiple windows: W/2, W, 2W, 4W
    # The curve shape is diagnostic — instant=topology, gradual=genuine, none=destroyed
    t0 = _time.monotonic()
    recovery_horizons = [max(1, recovery_window // 2), recovery_window,
                         recovery_window * 2, recovery_window * 4]
    recovery_curve = []
    steps_run = 1  # Already ran 1 step for post_immediate

    for horizon in recovery_horizons:
        steps_needed = horizon - steps_run
        if steps_needed > 0:
            perturbed.reset_engagement_tracking()
            for _ in range(steps_needed):
                perturbed.step(None)
            steps_run = horizon
        eng = perturbed.get_engagement_distribution()
        cos_sim = _cosine_sim(pre_engagement, eng, regions)
        recovery_curve.append((horizon, cos_sim))

    _log(f"  recovery curve: {[(h, f'{c:.4f}') for h, c in recovery_curve]} "
         f"in {_time.monotonic()-t0:.2f}s")

    # Primary recovery uses W (second point in curve)
    post_recovery = perturbed.get_engagement_distribution()
    recovery_at_W = next((c for h, c in recovery_curve if h == recovery_window), 0.0)

    # Compute disruption (resistance) = 1 - cos_sim(pre, post_immediate)
    disruption = 1.0 - _cosine_sim(pre_engagement, post_immediate, regions)

    # Compute recovery = cos_sim at primary window W
    recovery = recovery_at_W

    return {
        "pre_engagement": pre_engagement,
        "post_immediate": post_immediate,
        "post_recovery": post_recovery,
        "target_region": target_region,
        "disruption": disruption,
        "recovery": recovery,
        "recovery_curve": recovery_curve,
        "regions": regions,
        "perturbation_validated": perturbation_validated,
        "perturbation_caveat": perturbation_caveat,
        "decoy_region": decoy_region,
        "original_recovery_ratio": original_recovery_ratio,
        "decoy_drift_ratio": decoy_drift_ratio,
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
    """Run the self-engagement instrument (redesign).

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
        control_factory: Returns fresh matched control. Required for self-engagement measurement.
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
            failure_mode="precondition-fail",
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
    # Option C: perturbation precondition failed → indeterminate
    if trained_result.get("indeterminate"):
        _log(f"  perturbation precondition failed — returning indeterminate")
        return InstrumentResult(
            name="self_engagement",
            passed=None,
            notes=f"Perturbation precondition failed: {trained_result['perturbation_caveat']}",
            raw_data=trained_result,
            failure_mode="perturbation-precondition-failed",
        )

    # --- Run protocol on fresh baseline ---
    if control_factory is None:
        _log("No control_factory — cannot compute fresh baseline")
        return InstrumentResult(
            name="self_engagement",
            passed=None,
            notes="No control_factory — fresh baseline required for ",
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

    # Check perturbation validation from both trained and fresh protocols
    trained_validated = trained_result.get("perturbation_validated", True)
    trained_caveat = trained_result.get("perturbation_caveat")
    fresh_validated = fresh_result.get("perturbation_validated", True)

    # Failure mode classification
    if passes_resistance and passes_recovery:
        passed = True
        failure_mode = "earned"
        notes = (
            f"Self-engagement detected: "
            f"resistance_ratio={resistance_ratio:.4f}, recovery_ratio={recovery_ratio:.4f}. "
            f"Trained system resists perturbation MORE and recovers MORE than fresh baseline."
        )
    else:
        passed = False
        reasons = []
        if not passes_resistance:
            reasons.append(f"resistance_ratio={resistance_ratio:.4f}<=1.0")
            failure_mode = "no-resistance"
        if not passes_recovery:
            reasons.append(f"recovery_ratio={recovery_ratio:.4f}<=1.0")
            failure_mode = "no-recovery"
        if not passes_resistance and not passes_recovery:
            # Check if fresh also recovers (topology-driven)
            if fresh_result.get("recovery", 0) > 0.8:
                failure_mode = "topology-driven"
            else:
                failure_mode = "no-resistance"
        notes = (
            f"No self-engagement: {', '.join(reasons)}. "
            f"Trained system does not exceed fresh baseline."
        )

    # Check decoy drift
    trained_decoy_drift = trained_result.get("decoy_drift_ratio")
    trained_original_recovery = trained_result.get("original_recovery_ratio")
    if (passed and trained_decoy_drift is not None and trained_original_recovery is not None
            and trained_decoy_drift > trained_original_recovery):
        failure_mode = "decoy-drift"

    # Append perturbation validation caveat if present
    if trained_caveat:
        notes += f" [CAVEAT: {trained_caveat}]"

    # Effect size: geometric mean of the two ratios (capped for inf)
    # Capped at 100.0 (self-engagement uses geometric mean of ratios; lower cap prevents one extreme ratio from dominating)
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
            "perturbation_validated": trained_validated,
            "perturbation_caveat": trained_caveat,
            "decoy_region": trained_result.get("decoy_region"),
            "original_recovery_ratio": trained_result.get("original_recovery_ratio"),
            "decoy_drift_ratio": trained_result.get("decoy_drift_ratio"),
        },
        notes=notes,
        failure_mode=failure_mode,
    )

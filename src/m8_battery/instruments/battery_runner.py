"""Battery runner — orchestrates all five instruments + provenance.

Runs the full Emergent Understanding Battery against a single system.
All five instruments must pass + provenance constraint for overall PASS.

Includes reset discrimination diagnostic (§6.7) — supplementary test
that distinguishes earned, received, and transient structure by measuring
persistence and regrowth after system reset.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable


def _log(msg: str) -> None:
    """Log battery progress to stderr (visible during long runs)."""
    print(f"[battery] {msg}", file=sys.stderr, flush=True)

import networkx as nx

from m8_battery.core.test_system import TestSystem
from m8_battery.core.types import BatteryResult, InstrumentResult, SystemClass
from m8_battery.core.provenance import ProvenanceLog
from m8_battery.instruments.developmental_trajectory import run_developmental_trajectory, compute_trajectory_compression
from m8_battery.instruments.integration import run_integration, compute_integration_earned_ratio
from m8_battery.instruments.generativity import run_generativity
from m8_battery.instruments.transfer import run_transfer
from m8_battery.instruments.self_engagement import run_self_engagement
from m8_battery.instruments.provenance_constraint import check_provenance


@dataclass
class BatteryConfig:
    """Configuration for a full battery run."""
    # Domain inputs
    domain_a_inputs: list[Any] = field(default_factory=list)
    domain_a_prime_inputs: list[Any] = field(default_factory=list)
    domain_b_inputs: list[Any] = field(default_factory=list)
    probe_inputs: list[Any] = field(default_factory=list)

    # Instrument parameters
    measurement_interval: int = 5
    wander_steps: int = 20
    perturbation_method: str = "shuffle_weights"
    recovery_window: int = 20

    # Region partitions for integration (None = use system default)
    partition_families: list[list[str]] | None = None


def run_reset_discrimination(
    system: TestSystem,
    domain_inputs: list[Any],
    n_steps: int = 20,
) -> dict[str, float]:
    """Supplementary diagnostic — not a pass/fail instrument.

    Distinguishes earned, received, and transient structure by measuring
    what happens after system reset:
    - Received (Class 1/2): metric persists (topology/weights unchanged)
    - Transient (e.g. Foxworthy C): metric drops (hidden state resets)
    - Earned (Class 3/4): metric drops then regrows (learning capability)

    Args:
        system: System under test (should have been operated already)
        domain_inputs: Inputs for post-reset interaction
        n_steps: Number of post-reset interaction steps

    Returns:
        Dict with pre_reset, post_reset, post_rerun metrics plus
        reset_persistence and regrowth_rate diagnostics.
    """
    pre_reset = system.get_structure_metric()
    system.reset()
    post_reset = system.get_structure_metric()

    for inp in domain_inputs[:n_steps]:
        system.step(inp)
    post_rerun = system.get_structure_metric()

    reset_persistence = post_reset / pre_reset if pre_reset > 1e-10 else 0.0
    if abs(pre_reset - post_reset) > 1e-10:
        regrowth_rate = (post_rerun - post_reset) / n_steps
    else:
        regrowth_rate = 0.0

    return {
        "pre_reset": pre_reset,
        "post_reset": post_reset,
        "post_rerun": post_rerun,
        "reset_persistence": reset_persistence,
        "regrowth_rate": regrowth_rate,
    }


def _collect_baseline(
    system: TestSystem,
    control_factory: Callable[[], TestSystem] | None = None,
) -> dict[str, float]:
    """Collect three-phase baseline metrics (supplementary diagnostic).

    Measures structure metric at three points:
    1. fresh: from a clone (untrained baseline)
    2. post_training: the system as received (already trained)
    3. during_battery: measured after instruments run (added later)

    Also measures per-instrument baselines on a fresh system:
    integration, generativity, self-engagement on a system that
    has never been operated. Distinguishes "received" properties
    (present at start) from "earned" (developed during operation).
    """
    # Post-training metric (system as received)
    post_training_metric = system.get_structure_metric()
    post_training_distribution = system.get_structure_distribution()

    # Fresh baseline from a clone
    fresh_metric = 0.0
    fresh_distribution: dict[str, float] = {}
    if control_factory is not None:
        try:
            fresh = control_factory()
            fresh_metric = fresh.get_structure_metric()
            fresh_distribution = fresh.get_structure_distribution()
        except Exception:
            pass  # Some systems can't produce metrics without training

    return {
        "fresh_metric": fresh_metric,
        "post_training_metric": post_training_metric,
        "metric_change_during_training": post_training_metric - fresh_metric,
        "fresh_distribution": fresh_distribution,
        "post_training_distribution": post_training_distribution,
    }


def _run_baseline_instruments(
    control_factory: Callable[[], TestSystem] | None,
    config: BatteryConfig,
) -> dict[str, dict]:
    """Run all 5 instruments on a fresh untrained system.

    Returns per-instrument baseline: passed, effect_size, classification.
    The classification is determined later by comparing with trained results.
    """
    if control_factory is None:
        return {}

    try:
        fresh = control_factory()
    except Exception:
        return {}

    baseline_results: dict[str, dict] = {}

    # Trajectory: feed domain A inputs to fresh system
    try:
        dt = run_developmental_trajectory(
            system=fresh, inputs=config.domain_a_inputs,
            measurement_interval=config.measurement_interval,
        )
        baseline_results["developmental_trajectory"] = {
            "passed": dt.passed, "effect_size": dt.effect_size,
        }
    except Exception:
        baseline_results["developmental_trajectory"] = {"passed": None, "effect_size": None}

    fresh_ref = fresh.get_structure_metric()

    # Integration
    try:
        integ = run_integration(
            system=fresh,
            probe_inputs=config.probe_inputs or config.domain_a_inputs[:10],
        )
        baseline_results["integration"] = {
            "passed": integ.passed, "effect_size": integ.effect_size,
        }
    except Exception:
        baseline_results["integration"] = {"passed": None, "effect_size": None}

    # Generativity
    try:
        gen = run_generativity(
            system=fresh, domain_b_inputs=config.domain_b_inputs,
            reference_metric=fresh_ref,
        )
        baseline_results["generativity"] = {
            "passed": gen.passed, "effect_size": gen.effect_size,
        }
    except Exception:
        baseline_results["generativity"] = {"passed": None, "effect_size": None}

    # Transfer (fresh vs another fresh)
    try:
        fresh2 = control_factory()
        trans = run_transfer(
            system=fresh, naive_system=fresh2,
            domain_a_prime_inputs=config.domain_a_prime_inputs,
            measurement_interval=config.measurement_interval,
        )
        baseline_results["transfer"] = {
            "passed": trans.passed, "effect_size": trans.effect_size,
        }
    except Exception:
        baseline_results["transfer"] = {"passed": None, "effect_size": None}

    # Self-engagement (DN-20: fresh system has no trajectory → precondition fail)
    try:
        se = run_self_engagement(
            system=fresh, wander_steps=config.wander_steps,
            perturbation_method=config.perturbation_method,
            recovery_window=config.recovery_window,
            trajectory_passed=None,  # Fresh system — no trajectory
        )
        baseline_results["self_engagement"] = {
            "passed": se.passed, "effect_size": se.effect_size,
        }
    except Exception:
        baseline_results["self_engagement"] = {"passed": None, "effect_size": None}

    return baseline_results


def _classify_instruments(
    trained_results: dict[str, InstrumentResult],
    baseline_results: dict[str, dict],
) -> dict[str, str]:
    """Classify each instrument: earned / received / absent / anomalous."""
    classifications = {}
    for name, trained in trained_results.items():
        bl = baseline_results.get(name, {})
        bl_passed = bl.get("passed")
        if trained.passed and bl_passed:
            classifications[name] = "received"
        elif trained.passed and not bl_passed:
            classifications[name] = "earned"
        elif not trained.passed and not bl_passed:
            classifications[name] = "absent"
        elif not trained.passed and bl_passed:
            classifications[name] = "anomalous"
        else:
            classifications[name] = "unknown"
    return classifications


def run_battery(
    system: TestSystem,
    system_name: str,
    system_class: SystemClass,
    config: BatteryConfig,
    control_factory: Callable[[], TestSystem] | None = None,
) -> BatteryResult:
    """Run the full Emergent Understanding Battery.

    Sequence:
    1. Operate system on domain A (training phase)
    2. Run developmental trajectory (measured during training)
    3. Run integration (on trained system)
    4. Run generativity (expose to domain B)
    5. Run transfer (compare to naive system on domain A')
    6. Run self-engagement (wander + perturb + recover)
    7. Check provenance

    All five instruments + provenance must pass for overall PASS.

    Args:
        system: The system under test
        system_name: Human-readable name
        system_class: Expected Antikythera class
        config: Battery configuration
        control_factory: Returns fresh matched control systems

    Returns:
        BatteryResult with per-instrument results and overall verdict
    """
    import time as _time

    provenance = ProvenanceLog()
    results: dict[str, InstrumentResult] = {}
    timings: dict[str, float] = {}
    _battery_start = _time.monotonic()

    # --- Phase 0: Baseline measurement (supplementary diagnostic) ---
    _log(f"Phase 0: baseline measurement ({system_name})")
    t0 = _time.monotonic()
    baseline = _collect_baseline(system, control_factory)
    timings["baseline"] = _time.monotonic() - t0
    _log(f"  baseline: {timings['baseline']:.1f}s")

    # --- Phase 1: Train on domain A + measure developmental trajectory ---
    _log("Phase 1: developmental trajectory")
    t0 = _time.monotonic()
    results["developmental_trajectory"] = run_developmental_trajectory(
        system=system,
        inputs=config.domain_a_inputs,
        measurement_interval=config.measurement_interval,
        control_factory=control_factory,
        provenance=provenance,
    )
    timings["developmental_trajectory"] = _time.monotonic() - t0
    _log(f"  dev_trajectory: {timings['developmental_trajectory']:.1f}s passed={results['developmental_trajectory'].passed}")

    # Trajectory compression supplement (LZ compressibility, TERL 2025)
    traj_data = results["developmental_trajectory"].raw_data or {}
    traj_values = traj_data.get("trajectory", [])
    if traj_values:
        baseline["trajectory_compression"] = compute_trajectory_compression(traj_values)

    # Record reference metric after domain A training
    reference_metric = system.get_structure_metric()

    # --- Phase 2: Integration ---
    _log(f"Phase 2: integration ({len(system.get_regions())} regions)")
    t0 = _time.monotonic()
    results["integration"] = run_integration(
        system=system,
        probe_inputs=config.probe_inputs or config.domain_a_inputs[:10],
        partition_families=config.partition_families,
        provenance=provenance,
    )
    timings["integration"] = _time.monotonic() - t0
    _log(f"  integration: {timings['integration']:.1f}s passed={results['integration'].passed}")

    # Integration earned ratio (supplementary — IIT/Aaronson critique)
    # Uses trained Gini from Phase 2 result, runs fresh integration separately
    if control_factory is not None:
        trained_gini = results["integration"].raw_data.get("gini", 0.0) if results["integration"].raw_data else 0.0
        try:
            fresh_for_integ = control_factory()
            fresh_integ = run_integration(system=fresh_for_integ, probe_inputs=config.probe_inputs or config.domain_a_inputs[:10])
            fresh_gini = fresh_integ.raw_data.get("gini", 0.0) if fresh_integ.raw_data else 0.0
        except Exception:
            fresh_gini = 0.0
        earned_ratio = trained_gini / fresh_gini if fresh_gini > 1e-10 else (float("inf") if trained_gini > 1e-10 else 1.0)
        earned_ratio = float(min(earned_ratio, 1e6))
        baseline["integration_earned_ratio"] = {
            "trained_gini": trained_gini, "fresh_gini": fresh_gini,
            "earned_ratio": earned_ratio,
        }
        _log(f"  integration earned ratio: {earned_ratio:.4f}")

        # DN-22: Wire earned ratio into pass condition. If integration passed
        # but earned ratio <= 1.0, the signal is received (topology), not earned.
        integ_result = results["integration"]
        if integ_result.passed and earned_ratio <= 1.0:
            results["integration"] = InstrumentResult(
                name="integration",
                passed=False,
                effect_size=integ_result.effect_size,
                raw_data={**(integ_result.raw_data or {}), "earned_ratio": earned_ratio},
                notes=(f"Integration present but not earned (DN-22): earned_ratio={earned_ratio:.2f}. "
                       f"Fresh system shows similar reorganisation. {integ_result.notes}"),
            )
            _log(f"  integration DOWNGRADED: earned_ratio={earned_ratio:.2f} <= 1.0")
        elif integ_result.raw_data is not None:
            integ_result.raw_data["earned_ratio"] = earned_ratio

    # --- Phase 3: Generativity (novel domain B) ---
    # T1-03: Freeze learning during measurement. Generativity measures structural
    # influence from domain A, not online adaptation on domain B.
    system.set_training(False)
    _log("Phase 3: generativity (learning FROZEN)")
    t0 = _time.monotonic()
    results["generativity"] = run_generativity(
        system=system,
        domain_b_inputs=config.domain_b_inputs,
        reference_metric=reference_metric,
        provenance=provenance,
    )
    timings["generativity"] = _time.monotonic() - t0
    _log(f"  generativity: {timings['generativity']:.1f}s passed={results['generativity'].passed}")
    system.set_training(True)  # Restore for subsequent phases

    # --- Phase 4: Transfer (domain A' vs naive) ---
    _log("Phase 4: transfer")
    t0 = _time.monotonic()
    if control_factory is not None:
        naive = control_factory()
        results["transfer"] = run_transfer(
            system=system,
            naive_system=naive,
            domain_a_prime_inputs=config.domain_a_prime_inputs,
            measurement_interval=config.measurement_interval,
            provenance=provenance,
        )
    else:
        results["transfer"] = InstrumentResult(
            name="transfer",
            passed=None,
            notes="No control factory provided — cannot run transfer comparison",
        )
    timings["transfer"] = _time.monotonic() - t0
    _log(f"  transfer: {timings['transfer']:.1f}s passed={results['transfer'].passed}")

    # --- Phase 5: Self-engagement (wander + perturb + recover) ---
    trajectory_passed = results["developmental_trajectory"].passed
    _log(f"Phase 5: self-engagement (trajectory_passed={trajectory_passed})")
    t0 = _time.monotonic()
    results["self_engagement"] = run_self_engagement(
        system=system,
        wander_steps=config.wander_steps,
        perturbation_method=config.perturbation_method,
        recovery_window=config.recovery_window,
        provenance=provenance,
        control_factory=control_factory,
        trajectory_passed=trajectory_passed,
    )
    timings["self_engagement"] = _time.monotonic() - t0
    _log(f"  self_engagement: {timings['self_engagement']:.1f}s passed={results['self_engagement'].passed}")

    # --- Phase 5b: Post-battery metric for two-window trajectory ---
    post_battery_metric = system.get_structure_metric()
    baseline["post_battery_metric"] = post_battery_metric
    baseline["metric_change_during_battery"] = (
        post_battery_metric - baseline["post_training_metric"]
    )

    # Two-window trajectory summary:
    # Window 1 (training): fresh → post-training
    # Window 2 (battery): post-training → post-battery
    training_change = baseline["metric_change_during_training"]
    battery_change = baseline["metric_change_during_battery"]
    baseline["trajectory_training"] = (
        "earned" if abs(training_change) > 0.1
        else "static"
    )
    baseline["trajectory_battery"] = (
        "earned" if abs(battery_change) > 0.1
        else "static"
    )

    # --- Phase 6: Provenance constraint ---
    _log("Phase 6: provenance")
    t0 = _time.monotonic()
    prov_result = check_provenance(system, provenance)
    results["provenance_constraint"] = prov_result
    timings["provenance"] = _time.monotonic() - t0
    _log(f"  provenance: {timings['provenance']:.1f}s passed={prov_result.passed}")

    # --- Phase 7: Reset discrimination diagnostic (§6.7) ---
    _log("Phase 7: reset discrimination")
    t0 = _time.monotonic()
    reset_inputs = config.domain_a_inputs or config.probe_inputs
    reset_diag = run_reset_discrimination(
        system=system,
        domain_inputs=reset_inputs,
        n_steps=min(20, len(reset_inputs)),
    )
    timings["reset_discrimination"] = _time.monotonic() - t0
    _log(f"  reset_discrimination: {timings['reset_discrimination']:.1f}s")

    # --- Phase 8: Baseline instruments (all 5 on fresh system) ---
    skip_label = "SKIP (Class 1)" if system_class == SystemClass.CLASS_1 else "running"
    _log(f"Phase 8: baseline instruments ({skip_label})")
    # Option C (F-020): For Class 1 static systems, the baseline IS the trained
    # system — no training occurs, fresh = trained. Skip redundant instrument runs
    # and use trained results directly. Baseline result = trained result, explicitly
    # recorded (not omitted). See F-020 governance.
    t0 = _time.monotonic()
    instrument_results_for_classify = {
        k: v for k, v in results.items() if k != "provenance_constraint"
    }
    if system_class == SystemClass.CLASS_1:
        # Baseline skip for static systems — see F-020 governance.
        # Fresh = trained for Class 1 (no learning). Record trained results
        # as baseline to maintain identical output format.
        baseline_instrument_results = {
            name: {"passed": r.passed, "effect_size": r.effect_size}
            for name, r in instrument_results_for_classify.items()
        }
    else:
        baseline_instrument_results = _run_baseline_instruments(control_factory, config)
    instrument_classifications = _classify_instruments(
        instrument_results_for_classify, baseline_instrument_results,
    )
    baseline["instrument_baselines"] = baseline_instrument_results
    baseline["instrument_classifications"] = instrument_classifications
    timings["baseline_instruments"] = _time.monotonic() - t0
    _log(f"  baseline_instruments: {timings['baseline_instruments']:.1f}s")
    total_elapsed = _time.monotonic() - _battery_start
    timings["total"] = total_elapsed
    _log(f"Battery complete: {total_elapsed:.1f}s total")

    # --- Assemble result ---
    battery_result = BatteryResult(
        system_name=system_name,
        system_class=system_class,
        instrument_results={
            k: v for k, v in results.items()
            if k != "provenance_constraint"
        },
        provenance_passed=prov_result.passed,
        metadata={
            "reference_metric": reference_metric,
            "baseline": baseline,
            "reset_discrimination": reset_diag,
            "instrument_timings_s": timings,
            "config": {
                "measurement_interval": config.measurement_interval,
                "wander_steps": config.wander_steps,
                "perturbation_method": config.perturbation_method,
                "recovery_window": config.recovery_window,
                "n_domain_a_inputs": len(config.domain_a_inputs),
                "n_domain_b_inputs": len(config.domain_b_inputs),
                "n_domain_a_prime_inputs": len(config.domain_a_prime_inputs),
            },
        },
    )
    battery_result.overall_passed = battery_result.compute_overall()

    return battery_result

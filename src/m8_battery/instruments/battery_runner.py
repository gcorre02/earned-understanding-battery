"""Battery runner — orchestrates all five instruments + provenance.

Runs the full Emergent Understanding Battery against a single system.
All five instruments must pass + provenance constraint for overall PASS.

Includes reset discrimination diagnostic (§6.7) — supplementary test
that distinguishes earned, received, and transient structure by measuring
persistence and regrowth after system reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import networkx as nx

from m8_battery.core.test_system import TestSystem
from m8_battery.core.types import BatteryResult, InstrumentResult, SystemClass
from m8_battery.core.provenance import ProvenanceLog
from m8_battery.instruments.developmental_trajectory import run_developmental_trajectory
from m8_battery.instruments.integration import run_integration
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
    provenance = ProvenanceLog()
    results: dict[str, InstrumentResult] = {}

    # --- Phase 0: Baseline measurement (supplementary diagnostic) ---
    # Three-phase structure metric: fresh → post-training → during battery.
    # The system arrives already trained. Fresh baseline from a clone.
    baseline = _collect_baseline(system, control_factory)

    # --- Phase 1: Train on domain A + measure developmental trajectory ---
    results["developmental_trajectory"] = run_developmental_trajectory(
        system=system,
        inputs=config.domain_a_inputs,
        measurement_interval=config.measurement_interval,
        control_factory=control_factory,
        provenance=provenance,
    )

    # Record reference metric after domain A training
    reference_metric = system.get_structure_metric()

    # --- Phase 2: Integration ---
    results["integration"] = run_integration(
        system=system,
        probe_inputs=config.probe_inputs or config.domain_a_inputs[:10],
        partition_families=config.partition_families,
        provenance=provenance,
    )

    # --- Phase 3: Generativity (novel domain B) ---
    results["generativity"] = run_generativity(
        system=system,
        domain_b_inputs=config.domain_b_inputs,
        reference_metric=reference_metric,
        provenance=provenance,
    )

    # --- Phase 4: Transfer (domain A' vs naive) ---
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

    # --- Phase 5: Self-engagement (wander + perturb + recover) ---
    results["self_engagement"] = run_self_engagement(
        system=system,
        wander_steps=config.wander_steps,
        perturbation_method=config.perturbation_method,
        recovery_window=config.recovery_window,
        provenance=provenance,
    )

    # --- Phase 6: Provenance constraint ---
    prov_result = check_provenance(system, provenance)
    results["provenance_constraint"] = prov_result

    # --- Phase 7: Reset discrimination diagnostic (§6.7) ---
    # Supplementary — does not affect pass/fail. Reported in metadata.
    reset_inputs = config.domain_a_inputs or config.probe_inputs
    reset_diag = run_reset_discrimination(
        system=system,
        domain_inputs=reset_inputs,
        n_steps=min(20, len(reset_inputs)),
    )

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

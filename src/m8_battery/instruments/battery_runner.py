"""Battery runner — orchestrates all five instruments + provenance.

Runs the full Emergent Understanding Battery against a single system.
All five instruments must pass + provenance constraint for overall PASS.
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

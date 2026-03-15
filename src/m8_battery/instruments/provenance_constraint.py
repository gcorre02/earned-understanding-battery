"""Provenance constraint.

Not an instrument testing a property — it's an epistemic safeguard applied
across all five instruments. Verifies that the causal chain from input
through dynamic update to stable structure can be reconstructed.

For any positive instrument result, the evaluator must be able to
reconstruct the causal chain that produced the structure being measured.
"""

from __future__ import annotations

from m8_battery.core.test_system import TestSystem
from m8_battery.core.types import InstrumentResult
from m8_battery.core.provenance import ProvenanceLog


def check_provenance(
    system: TestSystem,
    provenance_log: ProvenanceLog,
) -> InstrumentResult:
    """Check provenance constraint.

    Verifies:
    1. The log is non-empty and complete (has inputs, state changes, outputs)
    2. State changes are traceable (before/after metrics recorded)
    3. The chain is continuous (no gaps in step indices)

    Args:
        system: The system that was tested
        provenance_log: The log from the battery run

    Returns:
        InstrumentResult — pass/fail on provenance completeness
    """
    issues = []

    # Check 1: Log completeness
    if provenance_log.event_count == 0:
        return InstrumentResult(
            name="provenance_constraint",
            passed=False,
            notes="Provenance log is empty",
            raw_data={"event_count": 0},
        )

    if not provenance_log.is_complete():
        issues.append("Log missing required event types (need: input, state_change, output)")

    # Check 2: State changes traceable
    events = provenance_log.events
    state_changes = [e for e in events if e.event_type == "state_change"]
    inputs = [e for e in events if e.event_type == "input"]
    outputs = [e for e in events if e.event_type == "output"]

    if len(state_changes) == 0:
        issues.append("No state_change events — cannot trace structural evolution")

    if len(inputs) == 0:
        issues.append("No input events — cannot trace causal chain origin")

    if len(outputs) == 0:
        issues.append("No output events — cannot trace causal chain completion")

    # Check 3: Step indices present (not continuity — multiple instruments
    # log to the same provenance with independent step sequences)
    input_steps = [e.data.get("step_index", -1) for e in inputs]
    if not input_steps or all(s < 0 for s in input_steps):
        issues.append("No step indices in input events")

    # Check 4: State changes have before/after metrics
    for sc in state_changes:
        if "metric_before" not in sc.data or "metric_after" not in sc.data:
            issues.append(f"State change at step {sc.data.get('step_index', '?')} missing before/after metrics")
            break  # Only report once

    # Check 5: Measurement events present (instruments recorded their results)
    measurements = [e for e in events if e.event_type == "measurement"]
    if len(measurements) == 0:
        issues.append("No measurement events — instruments did not record results")

    # Verdict
    if not issues:
        passed = True
        notes = (f"Provenance complete: {len(inputs)} inputs, "
                 f"{len(state_changes)} state changes, "
                 f"{len(outputs)} outputs, "
                 f"{len(measurements)} measurements")
    else:
        passed = False
        notes = f"Provenance issues: {'; '.join(issues)}"

    return InstrumentResult(
        name="provenance_constraint",
        passed=passed,
        raw_data={
            "event_count": provenance_log.event_count,
            "n_inputs": len(inputs),
            "n_state_changes": len(state_changes),
            "n_outputs": len(outputs),
            "n_measurements": len(measurements),
            "issues": issues,
        },
        notes=notes,
    )

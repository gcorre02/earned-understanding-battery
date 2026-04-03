"""Provenance logging for deterministic replay and auditability."""

from __future__ import annotations

import json
import time
from pathlib import Path

from m8_battery.core.types import ProvenanceEvent

class ProvenanceLog:
    """Append-only event log for a single battery run."""

    def __init__(self) -> None:
        self._events: list[ProvenanceEvent] = []
        self._start_time = time.monotonic()

    def log(self, event_type: str, **data) -> None:
        """Record an event."""
        self._events.append(ProvenanceEvent(
            timestamp=time.monotonic() - self._start_time,
            event_type=event_type,
            data=data,
        ))

    def log_input(self, input_data, step_index: int) -> None:
        """Record an input event."""
        self.log("input", step_index=step_index, input_repr=repr(input_data))

    def log_state_change(self, metric_before: float, metric_after: float,
                         step_index: int) -> None:
        """Record a state change with before/after metrics."""
        self.log("state_change",
                 step_index=step_index,
                 metric_before=metric_before,
                 metric_after=metric_after)

    def log_output(self, output_data, step_index: int) -> None:
        """Record an output event."""
        self.log("output", step_index=step_index, output_repr=repr(output_data))

    def log_measurement(self, instrument: str, result_summary: dict) -> None:
        """Record an instrument measurement."""
        self.log("measurement", instrument=instrument, **result_summary)

    @property
    def events(self) -> list[ProvenanceEvent]:
        """All recorded events (read-only view)."""
        return list(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def is_complete(self) -> bool:
        """Check that the log has inputs, state changes, and outputs."""
        types = {e.event_type for e in self._events}
        return {"input", "state_change", "output"}.issubset(types)

    def save(self, path: Path) -> None:
        """Save log to JSON file."""
        data = [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "data": e.data,
            }
            for e in self._events
        ]
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> ProvenanceLog:
        """Load log from JSON file."""
        log = cls()
        raw = json.loads(path.read_text())
        for entry in raw:
            log._events.append(ProvenanceEvent(
                timestamp=entry["timestamp"],
                event_type=entry["event_type"],
                data=entry.get("data", {}),
            ))
        return log

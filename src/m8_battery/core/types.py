"""Core data types for the Earned Understanding Battery."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SystemClass(Enum):
    """Antikythera spectrum classification."""
    CLASS_1 = 1  # Fixed mechanism — no learning
    CLASS_2 = 2  # Has structure, no operational learning
    CLASS_3 = 3  # Learns with external semantic objective
    CLASS_4 = 4  # Understanding-generating substrate (candidate)


@dataclass(frozen=True)
class Snapshot:
    """Serialised system state."""
    data: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Region:
    """An ablatable/perturbable structural region."""
    id: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InstrumentResult:
    """Result from a single battery instrument."""
    name: str
    passed: bool | None  # True, False, or None (ambiguous)
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    failure_mode: str = ""  # T1-05: named failure classification (e.g., "absent", "architectural")


@dataclass
class BatteryResult:
    """Aggregated result from running the full battery."""
    system_name: str
    system_class: SystemClass
    instrument_results: dict[str, InstrumentResult] = field(default_factory=dict)
    provenance_passed: bool | None = None
    overall_passed: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_overall(self) -> bool | None:
        """All five instruments + provenance must pass. No partial credit."""
        results = list(self.instrument_results.values())
        if not results:
            return None
        if any(r.passed is None for r in results):
            return None
        if self.provenance_passed is None:
            return None
        return all(r.passed for r in results) and self.provenance_passed


@dataclass
class ProvenanceEvent:
    """A single logged event in the provenance chain."""
    timestamp: float
    event_type: str  # "input", "state_change", "output", "measurement"
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainConfig:
    """Configuration for SBM domain generation."""
    n_nodes: int
    n_communities: int
    p_within: float
    p_between: float
    n_edge_types: int = 3
    weight_range: tuple[float, float] = (0.1, 1.0)
    n_node_features: int = 8
    seed: int | None = None

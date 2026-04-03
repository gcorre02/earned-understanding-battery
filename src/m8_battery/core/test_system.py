"""TestSystem ABC — the core interface all systems must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TestSystem(ABC):
    """Base class for all systems under test.

    Every calibration and candidate system must implement this interface.
    Instruments are defined entirely in terms of these methods.
    """

    @abstractmethod
    def reset(self) -> None:
        """Restore system to initial state."""

    @abstractmethod
    def step(self, input_data: Any) -> Any:
        """Process one interaction. Return output."""

    @abstractmethod
    def get_state(self) -> bytes:
        """Serialise complete internal state as bytes."""

    @abstractmethod
    def set_state(self, snapshot: bytes) -> None:
        """Restore from serialised state."""

    @abstractmethod
    def get_structure_metric(self) -> float:
        """Return scalar measure of structural organisation.

        Must be declared and justified per system before testing.
        """

    @abstractmethod
    def get_structure_distribution(self) -> dict[str, float]:
        """Return per-region structure metrics."""

    @abstractmethod
    def get_engagement_distribution(self) -> dict[str, float]:
        """Return per-region visit/activation frequency."""

    @abstractmethod
    def ablate(self, region_id: str) -> TestSystem:
        """Return new system instance with region removed.

        The original system is untouched. Critical for integration
        instrument which compares original vs ablated.
        """

    @abstractmethod
    def perturb(self, region_id: str, method: str) -> TestSystem:
        """Return new system instance with region modified (not removed)."""

    @abstractmethod
    def get_regions(self) -> list[str]:
        """Enumerate ablatable/perturbable structural regions.

        What a 'region' means is system-specific: cluster of graph nodes,
        attention heads, network layers, rule groups, etc. Each adapter
        documents its region semantics.
        """

    @abstractmethod
    def clone(self) -> TestSystem:
        """Return independent copy sharing initial conditions.

        Must behave identically to self from the same initial conditions
        but be fully independent (no shared mutable state).
        """

    def get_representation_state(self) -> Any:
        """Return a matrix snapshot of internal state for CKA computation.

        Returns None if not supported (Class 1 static, Class 2 frozen).
        Returns np.ndarray for systems with computable representations.

        Used by T1-02 metric bundle: CKA between early and late snapshots
        measures whether internal representations changed during operation.
        """
        return None  # Default: not supported

    def boost(self, region_id: str) -> TestSystem:
        """Return new system instance with boosted structure in target region.

        Used by T1-01f false-attractor control: after perturbation of the
        target region, boost a non-target region to create a decoy attractor.
        If the system reconstructs the ORIGINAL pattern instead of drifting
        to the decoy, that's stronger evidence of earned self-engagement.

        Default: return clone (no boost). Adapters can override with
        substrate-appropriate boosting (e.g., strengthen edges, boost weights).
        """
        return self.clone()

    def reset_engagement_tracking(self) -> None:
        """Reset engagement counters for windowed measurement.

        Called by self-engagement instrument before each measurement phase
        so that get_engagement_distribution() reflects recent activity only,
        not cumulative history. Default: no-op (systems whose engagement
        distribution is already instantaneous need not override).
        """

    def set_domain(self, graph: Any) -> None:
        """Switch the system's navigation domain while preserving learned state.

        For graph-based systems: replaces the underlying graph topology.
        Learned parameters (edge weights, preferences, policies) are preserved
        where applicable. Edges/nodes that don't exist in the new graph get
        default values.

        Used by generativity instrument: train on domain A, switch to domain B,
        navigate B with frozen A-trained structure. Default: no-op (systems that
        don't navigate graphs need not override).
        """

    def set_training(self, mode: bool) -> None:
        """Enable (True) or disable (False) learning during step().

        When mode=False, step() should operate the system WITHOUT updating
        any learned parameters. The system navigates using only what it
        has already earned. Default: no-op (systems that don't learn during
        step() need not override).

        Used by generativity instrument: domain B measurement must
        be frozen to distinguish structural influence from online adaptation.
        """

    def get_initial_position(self) -> Any:
        """Return the node the system started at (before any training).

        Used to sync fresh baseline starting position with trained system.
        The fresh baseline should start at the same position as the trained
        system did at the beginning of training, ensuring the ONLY difference
        is the training history, not the starting conditions.

        Default: None (systems without a graph position concept).
        """
        return None

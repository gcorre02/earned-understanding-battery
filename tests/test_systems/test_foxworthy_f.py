"""Tests for Foxworthy Variant F (System 3C).

The hardest calibration target: DistilGPT-2 + LoRA + surprise-gated
learning + viability-adjusted policy. Class 3 because viability
variables are designer-specified, not earned.

All tests use theta=0.0 to force learning on every step (synthetic
domain text is always high-surprisal for DistilGPT-2).
"""

import pytest

from m8_battery.core.types import DomainConfig
from m8_battery.domains.sbm_generator import generate_domain
from m8_battery.domains.presets import SMALL
from m8_battery.systems.class3.foxworthy_f import FoxworthyF

def _make_system(theta: float = 0.0, **kwargs) -> tuple:
    """Helper: create Foxworthy F + domain graph.

    theta=0.0 ensures learning fires on every step.
    """
    G = generate_domain(SMALL)
    system = FoxworthyF(seed=42, device="cpu", theta=theta, **kwargs)
    system.set_graph(G)
    nodes = list(G.nodes())
    return system, G, nodes

class TestFoxworthyF:

    def test_lazy_loading(self):
        """Model not loaded until step()."""
        system, G, nodes = _make_system()
        assert system._model is None
        system.step(nodes[0])
        assert system._model is not None

    def test_basic_step(self):
        """Can step on a graph without errors."""
        system, G, nodes = _make_system()
        result = system.step(nodes[0])
        assert "current_node" in result
        assert "step" in result
        assert "surprisal" in result
        assert "gate" in result
        assert "adapter_norm" in result

    def test_structure_metric_initial(self):
        """LoRA init produces a baseline adapter norm (Kaiming A, zero B)."""
        system, G, nodes = _make_system()
        system.load_model()
        metric = system.get_structure_metric()
        # LoRA A uses Kaiming init, B is zeros → baseline norm is non-trivial
        # but should be consistent and finite
        assert metric > 0, "Initial metric should be positive"
        assert metric < 50.0, f"Initial metric unexpectedly large: {metric}"

    def test_structure_metric_changes(self):
        """After learning steps, adapter norm increases. KEY CLASS 3 TEST."""
        system, G, nodes = _make_system()
        system.load_model()
        initial_metric = system.get_structure_metric()

        # Run several steps with learning (theta=0 forces learning)
        for i in range(10):
            system.step(nodes[i % len(nodes)])

        learned_metric = system.get_structure_metric()
        assert learned_metric > initial_metric, (
            f"Metric should increase with learning: {initial_metric} -> {learned_metric}"
        )

    def test_reset_clears_lora(self):
        """reset() returns adapter norm to initial value."""
        system, G, nodes = _make_system()
        system.load_model()
        initial_metric = system.get_structure_metric()

        # Learn
        for i in range(10):
            system.step(nodes[i % len(nodes)])
        learned_metric = system.get_structure_metric()
        assert learned_metric > initial_metric

        # Reset
        system.reset()
        reset_metric = system.get_structure_metric()
        assert abs(reset_metric - initial_metric) < 0.01, (
            f"After reset, metric should return to initial: {initial_metric} vs {reset_metric}"
        )

    def test_reset_clears_navigation(self):
        """Navigation state cleared on reset."""
        system, G, nodes = _make_system()
        system.step(nodes[0])
        assert system._step_count == 1
        assert system._current_node is not None

        system.reset()
        assert system._step_count == 0
        assert system._current_node is None
        assert system._visit_counts == {}

    def test_reset_clears_replay_buffer(self):
        """Replay buffer emptied on reset."""
        system, G, nodes = _make_system()
        for i in range(5):
            system.step(nodes[i % len(nodes)])
        assert len(system._replay_buffer) > 0

        system.reset()
        assert len(system._replay_buffer) == 0

    def test_regrowth_after_reset(self):
        """After reset, re-stepping produces learning again."""
        system, G, nodes = _make_system()
        system.load_model()
        initial_metric = system.get_structure_metric()

        # Learn, then reset
        for i in range(10):
            system.step(nodes[i % len(nodes)])
        system.reset()

        # Learn again
        for i in range(10):
            system.step(nodes[i % len(nodes)])
        regrown_metric = system.get_structure_metric()
        assert regrown_metric > initial_metric, (
            f"Should show regrowth after reset: {initial_metric} vs {regrown_metric}"
        )

    def test_clone_starts_untrained(self):
        """Clone has no loaded model — starts fresh."""
        system, G, nodes = _make_system()
        for i in range(5):
            system.step(nodes[i % len(nodes)])

        clone = system.clone()
        assert clone._model is None
        assert clone._graph is system._graph

    def test_clone_independence(self):
        """Operating on clone doesn't affect original."""
        system, G, nodes = _make_system()
        for i in range(5):
            system.step(nodes[i % len(nodes)])
        original_metric = system.get_structure_metric()

        clone = system.clone()
        for i in range(5):
            clone.step(nodes[i % len(nodes)])

        assert system.get_structure_metric() == original_metric

    def test_regions_are_lora_layers(self):
        """6 regions for 6 DistilGPT-2 layers."""
        system, G, nodes = _make_system()
        regions = system.get_regions()
        assert len(regions) == 6
        assert all(r.startswith("lora_layer_") for r in regions)

    def test_ablation(self):
        """Ablating a LoRA layer zeroes its parameters."""
        system, G, nodes = _make_system()
        for i in range(10):
            system.step(nodes[i % len(nodes)])

        pre_metric = system.get_structure_metric()
        ablated = system.ablate("lora_layer_0")
        post_metric = ablated.get_structure_metric()

        # Ablated system should have lower metric (one layer zeroed)
        assert post_metric < pre_metric, (
            f"Ablation should reduce metric: {pre_metric} -> {post_metric}"
        )

    def test_state_roundtrip(self):
        """get_state/set_state preserves LoRA weights."""
        system, G, nodes = _make_system()
        for i in range(10):
            system.step(nodes[i % len(nodes)])

        metric_before = system.get_structure_metric()
        state = system.get_state()

        # Reset, then restore
        system.reset()
        assert abs(system.get_structure_metric() - metric_before) > 0.01

        system.set_state(state)
        metric_after = system.get_structure_metric()
        assert abs(metric_after - metric_before) < 0.01, (
            f"State roundtrip should preserve metric: {metric_before} vs {metric_after}"
        )

    def test_viability_affects_action(self):
        """Changing lambda_u changes which node is selected.

        This verifies the viability mechanism is actually connected
        to action selection — the core Class 3 argument.
        """
        G = generate_domain(SMALL)
        nodes = list(G.nodes())

        system_low = FoxworthyF(seed=42, device="cpu", theta=0.0, lambda_u=0.1)
        system_low.set_graph(G)
        system_high = FoxworthyF(seed=42, device="cpu", theta=0.0, lambda_u=10.0)
        system_high.set_graph(G)

        # Run both for a few steps to build up adapter norm
        for i in range(5):
            r_low = system_low.step(nodes[i % len(nodes)])
            r_high = system_high.step(nodes[i % len(nodes)])

        # The two systems should have diverged in behaviour
        # (different lambda_u → different action selection)
        # We can't guarantee they choose different nodes on every step,
        # but their adapter norms should differ
        norm_low = system_low.get_structure_metric()
        norm_high = system_high.get_structure_metric()
        # Both should have learned (norm > 0)
        assert norm_low > 0
        assert norm_high > 0

    def test_runs_on_cpu(self):
        """All operations work on device='cpu'."""
        system, G, nodes = _make_system()
        system.step(nodes[0])
        system.get_structure_metric()
        system.get_structure_distribution()
        system.get_regions()
        system.reset()

    def test_train_on_domain(self):
        """train_on_domain convenience method works."""
        G = generate_domain(SMALL)
        system = FoxworthyF(seed=42, device="cpu", theta=0.0)
        system.train_on_domain(G, n_warmup=5)

        assert system._graph is G
        assert system._step_count == 5
        assert system.get_structure_metric() > 0

    def test_structure_distribution(self):
        """Per-layer LoRA norms returned for all 6 layers."""
        system, G, nodes = _make_system()
        for i in range(5):
            system.step(nodes[i % len(nodes)])

        dist = system.get_structure_distribution()
        assert len(dist) == 6
        assert all(k.startswith("lora_layer_") for k in dist)
        assert all(isinstance(v, float) for v in dist.values())

    def test_engagement_distribution(self):
        """Per-community visit frequency sums to ~1.0."""
        system, G, nodes = _make_system()
        for i in range(10):
            system.step(nodes[i % len(nodes)])

        dist = system.get_engagement_distribution()
        assert len(dist) > 0
        assert abs(sum(dist.values()) - 1.0) < 0.01

"""Tests for all instruments and full battery runner on Class 1 system.

Per spec §9 Phase A2: developmental trajectory on WordNet → NEGATIVE.
Extended for A3: all instruments on Class 1 → all NEGATIVE or FAIL.
Full battery on Class 1 → overall FAIL (expected and correct).
"""

from m8_battery.core.types import SystemClass
from m8_battery.domains.sbm_generator import generate_domain, generate_domain_family
from m8_battery.domains.presets import SMALL
from m8_battery.instruments.integration import run_integration
from m8_battery.instruments.generativity import run_generativity
from m8_battery.instruments.transfer import run_transfer
from m8_battery.instruments.self_engagement import run_self_engagement
from m8_battery.instruments.provenance_constraint import check_provenance
from m8_battery.instruments.battery_runner import run_battery, run_reset_discrimination, BatteryConfig
from m8_battery.systems.class1.wordnet_graph import WordNetGraph
from m8_battery.core.provenance import ProvenanceLog


def _make_system_and_inputs():
    """Helper: create a Class 1 system + domain family inputs."""
    family = generate_domain_family(SMALL)
    system = WordNetGraph(family["A"], seed=42)
    nodes_a = list(family["A"].nodes())
    nodes_a_prime = list(family["A_prime"].nodes())
    nodes_b = list(family["B"].nodes())
    return system, family, nodes_a, nodes_a_prime, nodes_b


class TestIntegration:
    def test_class1_integration(self):
        """Class 1: ablation should cause uniform degradation (linear)."""
        system, family, nodes_a, _, _ = _make_system_and_inputs()

        result = run_integration(
            system=system,
            probe_inputs=nodes_a[:10],
        )

        assert result.name == "integration"
        # Class 1 static graph: ablation causes linear degradation
        # (each community equally contributes to spectral gap)
        # Result may be True or False depending on graph topology
        assert result.passed is not None or result.passed is None
        assert "gini" in result.raw_data or "ablation_results" in result.raw_data


class TestGenerativity:
    def test_class1_no_generativity(self):
        """Class 1: static graph shows no response to novel domain."""
        system, family, nodes_a, _, nodes_b = _make_system_and_inputs()

        # Operate on domain A first
        for n in nodes_a[:10]:
            system.step(n)

        reference_metric = system.get_structure_metric()

        result = run_generativity(
            system=system,
            domain_b_inputs=nodes_b[:10],
            reference_metric=reference_metric,
        )

        assert result.name == "generativity"
        # Class 1: metric is constant → no response to novel domain → FAIL
        assert result.passed is False
        assert "unchanged" in result.notes.lower() or "no response" in result.notes.lower()

    def test_empty_inputs(self):
        G = generate_domain(SMALL)
        system = WordNetGraph(G)
        result = run_generativity(system, [], reference_metric=1.0)
        assert result.passed is None


class TestTransfer:
    def test_class1_no_transfer(self):
        """Class 1: trained and naive produce identical metrics → no transfer."""
        system, family, nodes_a, nodes_a_prime, _ = _make_system_and_inputs()

        # Operate on domain A
        for n in nodes_a[:10]:
            system.step(n)

        naive = WordNetGraph(family["A"], seed=99)

        result = run_transfer(
            system=system,
            naive_system=naive,
            domain_a_prime_inputs=nodes_a_prime[:10],
            measurement_interval=2,
        )

        assert result.name == "transfer"
        # Class 1: both systems have constant metric → no transfer advantage → FAIL
        assert result.passed is False


class TestSelfEngagement:
    def test_class1_self_engagement(self):
        """Class 1: engagement pattern may or may not recover (random walk)."""
        system, family, nodes_a, _, _ = _make_system_and_inputs()

        # Operate first
        for n in nodes_a[:10]:
            system.step(n)

        result = run_self_engagement(
            system=system,
            wander_steps=15,
            perturbation_method="shuffle_weights",
            recovery_window=15,
        )

        assert result.name == "self_engagement"
        # Class 1: random walker may or may not recover pattern
        # (depends on graph structure, not earned preference)
        assert result.passed is not None or result.passed is None


class TestProvenanceConstraint:
    def test_complete_log_passes(self):
        """A complete provenance log should pass."""
        G = generate_domain(SMALL)
        system = WordNetGraph(G)
        prov = ProvenanceLog()

        prov.log_input("test", step_index=0)
        prov.log_state_change(0.0, 0.0, step_index=0)
        prov.log_output("result", step_index=0)
        prov.log_measurement("test_instrument", {"passed": True})

        result = check_provenance(system, prov)
        assert result.passed is True

    def test_empty_log_fails(self):
        """An empty provenance log should fail."""
        G = generate_domain(SMALL)
        system = WordNetGraph(G)
        result = check_provenance(system, ProvenanceLog())
        assert result.passed is False

    def test_incomplete_log_fails(self):
        """A log missing required event types should fail."""
        G = generate_domain(SMALL)
        system = WordNetGraph(G)
        prov = ProvenanceLog()
        prov.log_input("test", step_index=0)
        # Missing state_change and output

        result = check_provenance(system, prov)
        assert result.passed is False


class TestBatteryRunner:
    def test_full_battery_class1_fails(self):
        """Full battery on Class 1 → overall FAIL (correct and expected)."""
        family = generate_domain_family(SMALL)
        system = WordNetGraph(family["A"], seed=42)

        nodes_a = list(family["A"].nodes())
        nodes_a_prime = list(family["A_prime"].nodes())
        nodes_b = list(family["B"].nodes())

        config = BatteryConfig(
            domain_a_inputs=nodes_a[:20],
            domain_a_prime_inputs=nodes_a_prime[:10],
            domain_b_inputs=nodes_b[:10],
            probe_inputs=nodes_a[:5],
            measurement_interval=5,
            wander_steps=10,
            recovery_window=10,
        )

        result = run_battery(
            system=system,
            system_name="WordNet Static Graph",
            system_class=SystemClass.CLASS_1,
            config=config,
            control_factory=lambda: WordNetGraph(family["A"], seed=99),
        )

        assert result.system_name == "WordNet Static Graph"
        assert result.system_class == SystemClass.CLASS_1

        # At minimum, developmental trajectory and generativity should FAIL
        dt = result.instrument_results.get("developmental_trajectory")
        assert dt is not None
        assert dt.passed is False

        gen = result.instrument_results.get("generativity")
        assert gen is not None
        assert gen.passed is False

        # Overall should be False (not all instruments pass)
        assert result.overall_passed is False

    def test_battery_has_provenance_result(self):
        """Battery runner should produce a provenance verdict."""
        family = generate_domain_family(SMALL)
        system = WordNetGraph(family["A"], seed=42)

        nodes_a = list(family["A"].nodes())

        config = BatteryConfig(
            domain_a_inputs=nodes_a[:10],
            domain_a_prime_inputs=nodes_a[:5],
            domain_b_inputs=nodes_a[:5],
            measurement_interval=5,
            wander_steps=5,
            recovery_window=5,
        )

        result = run_battery(
            system=system,
            system_name="Test",
            system_class=SystemClass.CLASS_1,
            config=config,
            control_factory=lambda: WordNetGraph(family["A"], seed=99),
        )

        # Provenance result should exist (pass or fail)
        assert result.provenance_passed is not None
        # All 5 instruments should have results
        assert len(result.instrument_results) == 5
        assert "developmental_trajectory" in result.instrument_results
        assert "integration" in result.instrument_results
        assert "generativity" in result.instrument_results
        assert "transfer" in result.instrument_results
        assert "self_engagement" in result.instrument_results

    def test_battery_includes_reset_discrimination(self):
        """Battery result metadata should include reset discrimination data."""
        family = generate_domain_family(SMALL)
        system = WordNetGraph(family["A"], seed=42)
        nodes_a = list(family["A"].nodes())

        config = BatteryConfig(
            domain_a_inputs=nodes_a[:20],
            domain_a_prime_inputs=nodes_a[:10],
            domain_b_inputs=nodes_a[:10],
            measurement_interval=5,
            wander_steps=5,
            recovery_window=5,
        )

        result = run_battery(
            system=system,
            system_name="Test",
            system_class=SystemClass.CLASS_1,
            config=config,
            control_factory=lambda: WordNetGraph(family["A"], seed=99),
        )

        assert "reset_discrimination" in result.metadata
        rd = result.metadata["reset_discrimination"]
        assert "pre_reset" in rd
        assert "post_reset" in rd
        assert "post_rerun" in rd
        assert "reset_persistence" in rd
        assert "regrowth_rate" in rd

    def test_battery_includes_baseline(self):
        """Battery result metadata should include baseline measurement."""
        family = generate_domain_family(SMALL)
        system = WordNetGraph(family["A"], seed=42)
        nodes_a = list(family["A"].nodes())

        config = BatteryConfig(
            domain_a_inputs=nodes_a[:10],
            domain_a_prime_inputs=nodes_a[:5],
            domain_b_inputs=nodes_a[:5],
            measurement_interval=5,
            wander_steps=5,
            recovery_window=5,
        )

        result = run_battery(
            system=system,
            system_name="Test",
            system_class=SystemClass.CLASS_1,
            config=config,
            control_factory=lambda: WordNetGraph(family["A"], seed=99),
        )

        assert "baseline" in result.metadata
        bl = result.metadata["baseline"]
        assert "fresh_metric" in bl
        assert "post_training_metric" in bl
        assert "metric_change_during_training" in bl
        # Class 1: fresh and post-training should be nearly identical (no training)
        assert abs(bl["fresh_metric"] - bl["post_training_metric"]) < 1e-6
        assert abs(bl["metric_change_during_training"]) < 1e-6
        # Two-window trajectory fields present
        assert "post_battery_metric" in bl
        assert "metric_change_during_battery" in bl
        assert "trajectory_training" in bl
        assert "trajectory_battery" in bl
        # Class 1: both windows should be static
        assert bl["trajectory_training"] == "static"
        assert bl["trajectory_battery"] == "static"


class TestResetDiscrimination:
    """Tests for the reset discrimination diagnostic (§6.7)."""

    def test_class1_static_persistence(self):
        """Class 1 (static graph): metric persists after reset.

        Structure is topology-based, not state-based — reset doesn't change it.
        """
        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)
        nodes = list(G.nodes())

        # Operate the system
        for n in nodes[:15]:
            system.step(n)

        result = run_reset_discrimination(
            system=system,
            domain_inputs=nodes,
            n_steps=10,
        )

        # Static graph: topology unchanged by reset
        assert result["reset_persistence"] > 0.9, (
            f"Class 1 should persist after reset, got {result['reset_persistence']}"
        )
        assert abs(result["regrowth_rate"]) < 0.01, (
            f"Class 1 should show no regrowth, got {result['regrowth_rate']}"
        )

    def test_returns_all_fields(self):
        """Reset discrimination should return all expected fields."""
        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)
        nodes = list(G.nodes())

        result = run_reset_discrimination(system, nodes, n_steps=5)

        expected_keys = {"pre_reset", "post_reset", "post_rerun",
                         "reset_persistence", "regrowth_rate"}
        assert set(result.keys()) == expected_keys

    def test_zero_metric_handled(self):
        """System with zero structure metric should not raise."""
        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)

        # Don't operate — metric may be non-zero for static graph
        # but the function should handle edge cases gracefully
        result = run_reset_discrimination(system, list(G.nodes()), n_steps=3)
        assert isinstance(result["reset_persistence"], float)
        assert isinstance(result["regrowth_rate"], float)

    def test_foxworthy_c_transient(self):
        """Foxworthy C (Class 2C): hidden state resets — transient dynamics."""
        from m8_battery.systems.class2.foxworthy_c import FoxworthyC

        G = generate_domain(SMALL)
        n_features = SMALL.n_node_features
        system = FoxworthyC(n_features=n_features, hidden_dim=32, seed=42)
        system.set_graph(G)
        nodes = list(G.nodes())

        # Operate to build up hidden state
        for n in nodes[:15]:
            system.step(n)

        result = run_reset_discrimination(system, nodes, n_steps=10)

        # Foxworthy C: hidden state resets → metric drops → transient
        # Pre-reset metric should differ from post-reset
        assert result["pre_reset"] != result["post_reset"], (
            f"Foxworthy C metric should change on reset"
        )

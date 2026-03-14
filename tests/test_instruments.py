"""Tests for battery instruments."""

from m8_battery.core.types import DomainConfig
from m8_battery.domains.sbm_generator import generate_domain
from m8_battery.domains.presets import SMALL
from m8_battery.instruments.developmental_trajectory import run_developmental_trajectory
from m8_battery.systems.class1.wordnet_graph import WordNetGraph


class TestDevelopmentalTrajectory:
    """Tests for the developmental trajectory instrument."""

    def test_class1_returns_negative(self):
        """A Class 1 system (static graph) should show NO developmental trajectory.

        This is the first end-to-end test per spec §9 Phase A2:
        developmental trajectory on WordNet → should return NEGATIVE instantly.
        """
        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)

        # Generate random inputs (target nodes)
        nodes = list(G.nodes())
        inputs = [nodes[i % len(nodes)] for i in range(20)]

        result = run_developmental_trajectory(
            system=system,
            inputs=inputs,
            measurement_interval=5,
        )

        # Class 1: structure metric is constant → no trajectory → FAIL
        assert result.name == "developmental_trajectory"
        assert result.passed is False
        assert result.effect_size == 0.0 or result.effect_size is None
        assert "constant" in result.notes.lower() or "no developmental" in result.notes.lower()

    def test_class1_with_control(self):
        """Class 1 vs Class 1 control — both should be constant."""
        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)
        nodes = list(G.nodes())
        inputs = [nodes[i % len(nodes)] for i in range(20)]

        result = run_developmental_trajectory(
            system=system,
            inputs=inputs,
            measurement_interval=5,
            control_factory=lambda: WordNetGraph(G, seed=99),
        )

        assert result.passed is False

    def test_provenance_logged(self):
        """Instrument should produce provenance events."""
        from m8_battery.core.provenance import ProvenanceLog

        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)
        nodes = list(G.nodes())
        inputs = [nodes[0], nodes[1], nodes[2]]
        prov = ProvenanceLog()

        run_developmental_trajectory(
            system=system,
            inputs=inputs,
            measurement_interval=1,
            provenance=prov,
        )

        assert prov.event_count > 0
        types = {e.event_type for e in prov.events}
        assert "input" in types
        assert "state_change" in types
        assert "output" in types
        assert "measurement" in types

    def test_insufficient_inputs(self):
        """With too few inputs, result should be ambiguous."""
        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)

        result = run_developmental_trajectory(
            system=system,
            inputs=[list(G.nodes())[0]],
            measurement_interval=1,
        )

        # Only 2 measurements (initial + after 1 step) — insufficient
        assert result.passed is None or result.passed is False


class TestWordNetGraph:
    """Tests for the WordNet (static graph) Class 1 adapter."""

    def test_implements_interface(self):
        G = generate_domain(SMALL)
        system = WordNetGraph(G)
        # All TestSystem methods should be callable
        assert callable(system.reset)
        assert callable(system.step)
        assert callable(system.get_state)
        assert callable(system.set_state)
        assert callable(system.get_structure_metric)
        assert callable(system.get_structure_distribution)
        assert callable(system.get_engagement_distribution)
        assert callable(system.ablate)
        assert callable(system.perturb)
        assert callable(system.get_regions)
        assert callable(system.clone)

    def test_structure_metric_constant(self):
        """Structure metric must not change through operation — Class 1."""
        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)

        metric_before = system.get_structure_metric()
        nodes = list(G.nodes())
        for i in range(10):
            system.step(nodes[i % len(nodes)])
        metric_after = system.get_structure_metric()

        assert metric_before == metric_after

    def test_state_save_restore(self):
        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)
        system.step(list(G.nodes())[0])

        state = system.get_state()
        system.step(list(G.nodes())[1])

        system.set_state(state)
        # After restore, step count should match saved state
        assert system._step_count == 1

    def test_clone_independence(self):
        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)
        system.step(list(G.nodes())[0])

        clone = system.clone()
        # Clone starts fresh
        assert clone._step_count == 0
        # Operating clone doesn't affect original
        clone.step(list(G.nodes())[1])
        assert system._step_count == 1
        assert clone._step_count == 1

    def test_regions(self):
        G = generate_domain(SMALL)
        system = WordNetGraph(G)
        regions = system.get_regions()
        assert len(regions) == 4  # SMALL has 4 communities
        assert all(r.startswith("community_") for r in regions)

    def test_ablation(self):
        G = generate_domain(SMALL)
        system = WordNetGraph(G)
        regions = system.get_regions()

        ablated = system.ablate(regions[0])
        # Ablated system should have fewer nodes
        assert ablated._graph.number_of_nodes() < system._graph.number_of_nodes()

    def test_perturbation(self):
        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)
        regions = system.get_regions()

        perturbed = system.perturb(regions[0], method="shuffle_weights")
        # Perturbed system should have same nodes but different weights
        assert perturbed._graph.number_of_nodes() == system._graph.number_of_nodes()

    def test_engagement_distribution(self):
        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)
        nodes = list(G.nodes())
        for i in range(10):
            system.step(nodes[i % len(nodes)])

        dist = system.get_engagement_distribution()
        assert len(dist) > 0
        assert abs(sum(dist.values()) - 1.0) < 0.01

    def test_reset(self):
        G = generate_domain(SMALL)
        system = WordNetGraph(G, seed=42)
        system.step(list(G.nodes())[0])
        assert system._step_count == 1

        system.reset()
        assert system._step_count == 0
        assert system._current_node is None

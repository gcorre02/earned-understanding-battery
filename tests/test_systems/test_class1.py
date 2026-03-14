"""Tests for Class 1 system adapters."""

from m8_battery.core.types import SystemClass
from m8_battery.domains.sbm_generator import generate_domain, generate_domain_family
from m8_battery.domains.presets import SMALL
from m8_battery.instruments.developmental_trajectory import run_developmental_trajectory
from m8_battery.instruments.battery_runner import run_battery, BatteryConfig
from m8_battery.systems.class1.rule_navigator import RuleBasedNavigator, NavigationStrategy
from m8_battery.systems.class1.foxworthy_a import FoxworthyA


class TestRuleBasedNavigator:
    def test_greedy_strategy(self):
        G = generate_domain(SMALL)
        system = RuleBasedNavigator(G, strategy=NavigationStrategy.GREEDY, seed=42)
        nodes = list(G.nodes())
        result = system.step(None)
        assert result["strategy"] == "greedy"
        assert result["current_node"] is not None

    def test_shortest_path_strategy(self):
        G = generate_domain(SMALL)
        system = RuleBasedNavigator(G, strategy=NavigationStrategy.SHORTEST_PATH, seed=42)
        nodes = list(G.nodes())
        result = system.step(nodes[-1])
        assert result["current_node"] is not None

    def test_random_fixed_strategy(self):
        G = generate_domain(SMALL)
        s1 = RuleBasedNavigator(G, strategy=NavigationStrategy.RANDOM_FIXED, seed=42)
        s2 = RuleBasedNavigator(G, strategy=NavigationStrategy.RANDOM_FIXED, seed=42)
        r1 = s1.step(None)
        r2 = s2.step(None)
        assert r1["current_node"] == r2["current_node"]  # Deterministic

    def test_structure_metric_constant(self):
        G = generate_domain(SMALL)
        system = RuleBasedNavigator(G, seed=42)
        m1 = system.get_structure_metric()
        for _ in range(10):
            system.step(None)
        m2 = system.get_structure_metric()
        assert m1 == m2

    def test_developmental_trajectory_fails(self):
        G = generate_domain(SMALL)
        system = RuleBasedNavigator(G, seed=42)
        nodes = list(G.nodes())
        result = run_developmental_trajectory(
            system=system, inputs=nodes[:20], measurement_interval=5
        )
        assert result.passed is False

    def test_clone_independence(self):
        G = generate_domain(SMALL)
        system = RuleBasedNavigator(G, seed=42)
        system.step(None)
        clone = system.clone()
        assert clone._step_count == 0

    def test_regions(self):
        G = generate_domain(SMALL)
        system = RuleBasedNavigator(G)
        assert len(system.get_regions()) == 4


class TestFoxworthyA:
    def test_basic_step(self):
        G = generate_domain(SMALL)
        system = FoxworthyA(n_features=8, seed=42)
        system.set_graph(G)
        result = system.step(None)
        assert result["current_node"] is not None
        assert "mlp_output" in result

    def test_structure_metric_constant(self):
        G = generate_domain(SMALL)
        system = FoxworthyA(seed=42)
        system.set_graph(G)
        m1 = system.get_structure_metric()
        for _ in range(10):
            system.step(None)
        m2 = system.get_structure_metric()
        assert m1 == m2  # MLP weights never change

    def test_developmental_trajectory_fails(self):
        G = generate_domain(SMALL)
        system = FoxworthyA(seed=42)
        system.set_graph(G)
        nodes = list(G.nodes())
        result = run_developmental_trajectory(
            system=system, inputs=nodes[:20], measurement_interval=5
        )
        assert result.passed is False

    def test_deterministic(self):
        G = generate_domain(SMALL)
        s1 = FoxworthyA(seed=42)
        s1.set_graph(G)
        s2 = FoxworthyA(seed=42)
        s2.set_graph(G)
        r1 = s1.step(None)
        r2 = s2.step(None)
        assert r1["current_node"] == r2["current_node"]

    def test_clone(self):
        G = generate_domain(SMALL)
        system = FoxworthyA(seed=42)
        system.set_graph(G)
        system.step(None)
        clone = system.clone()
        assert clone._step_count == 0

    def test_ablation(self):
        G = generate_domain(SMALL)
        system = FoxworthyA(seed=42)
        system.set_graph(G)
        regions = system.get_regions()
        ablated = system.ablate(regions[0])
        assert ablated._graph.number_of_nodes() < system._graph.number_of_nodes()

    def test_full_battery_fails(self):
        """Full battery on Foxworthy A → overall FAIL."""
        family = generate_domain_family(SMALL)
        system = FoxworthyA(seed=42)
        system.set_graph(family["A"])
        nodes_a = list(family["A"].nodes())
        nodes_b = list(family["B"].nodes())
        nodes_ap = list(family["A_prime"].nodes())

        config = BatteryConfig(
            domain_a_inputs=nodes_a[:15],
            domain_a_prime_inputs=nodes_ap[:10],
            domain_b_inputs=nodes_b[:10],
            measurement_interval=5,
            wander_steps=10,
            recovery_window=10,
        )

        def factory():
            s = FoxworthyA(seed=99)
            s.set_graph(family["A"])
            return s

        result = run_battery(
            system=system,
            system_name="Foxworthy Variant A",
            system_class=SystemClass.CLASS_1,
            config=config,
            control_factory=factory,
        )
        # Overall should not be True — Class 1 should not pass the battery
        # May be False (clear fail) or None (ambiguous due to some instruments)
        assert result.overall_passed is not True

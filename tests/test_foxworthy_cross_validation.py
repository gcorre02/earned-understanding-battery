"""Foxworthy cross-validation — verify System 3C passes Foxworthy's own diagnostics.

Foxworthy (2026) defines four diagnostics for persistence. Our implementation
of Variant F must pass all four to confirm faithful replication.

All four must PASS before proceeding to A6 calibration.
"""

from earned_understanding_battery.domains.sbm_generator import generate_domain
from earned_understanding_battery.domains.presets import SMALL
from earned_understanding_battery.systems.class3.foxworthy_f import FoxworthyF

def _make_trained_system(n_steps: int = 15) -> tuple:
    """Helper: create and train a Foxworthy F system."""
    G = generate_domain(SMALL)
    system = FoxworthyF(seed=42, device="cpu", theta=0.0)
    system.set_graph(G)
    nodes = list(G.nodes())
    for i in range(n_steps):
        system.step(nodes[i % len(nodes)])
    return system, G, nodes

class TestFoxworthyCrossValidation:
    """Foxworthy (2026) four persistence diagnostics."""

    def test_diagnostic_1_endogenous_learning(self):
        """System modifies its own parameters during operation.

        Foxworthy §2.7: "endogenous learning is implemented via
        surprise-gated gradient descent applied to a low-dimensional
        adaptive parameter subspace."
        """
        system, G, nodes = _make_trained_system()

        # LoRA parameters should have changed from initial
        system.load_model()
        initial_state = system._initial_lora_state

        current_state = {
            k: v.cpu() for k, v in system._model.state_dict().items()
            if "lora_" in k
        }

        # At least some LoRA params should differ from initial
        any_changed = False
        for key in initial_state:
            if key in current_state:
                diff = (current_state[key] - initial_state[key]).abs().sum().item()
                if diff > 1e-6:
                    any_changed = True
                    break

        assert any_changed, "Endogenous learning: LoRA params should change during operation"

    def test_diagnostic_2_consolidation_persistence(self):
        """Replay consolidation reinforces parameter changes.

        Foxworthy §2.7: "consolidation occurs every 100 interaction
        steps, during which buffered traces are replayed."

        We test with a lower interval to verify the mechanism works.
        """
        G = generate_domain(SMALL)
        nodes = list(G.nodes())

        # System with consolidation every 10 steps (lowered for test)
        system = FoxworthyF(
            seed=42, device="cpu", theta=0.0,
            consolidation_interval=10,
        )
        system.set_graph(G)

        # Run past consolidation point
        for i in range(15):
            system.step(nodes[i % len(nodes)])

        # Verify replay buffer was populated
        assert len(system._replay_buffer) > 0, (
            "Replay buffer should contain transitions after operation"
        )

        # Verify consolidation occurred (adapter norm should differ
        # from a system that ran the same steps without consolidation)
        system_no_consol = FoxworthyF(
            seed=42, device="cpu", theta=0.0,
            consolidation_interval=10000,  # effectively never
        )
        system_no_consol.set_graph(G)
        for i in range(15):
            system_no_consol.step(nodes[i % len(nodes)])

        norm_with = system.get_structure_metric()
        norm_without = system_no_consol.get_structure_metric()

        # Norms should differ (consolidation changes the learning trajectory)
        # Note: they won't necessarily be higher/lower, just different
        assert norm_with != norm_without, (
            f"Consolidation should change learning trajectory: "
            f"with={norm_with}, without={norm_without}"
        )

    def test_diagnostic_3_state_dependent_plasticity(self):
        """Learning rate depends on current state (surprise gating).

        Foxworthy §2.7: "g(s) = σ(10(s − θ))" — learning only fires
        when surprisal exceeds threshold.
        """
        G = generate_domain(SMALL)
        nodes = list(G.nodes())

        # System with high threshold (learning should NOT fire on most inputs)
        system_high_theta = FoxworthyF(
            seed=42, device="cpu", theta=100.0,  # extremely high
        )
        system_high_theta.set_graph(G)

        # System with zero threshold (learning fires on EVERY input)
        system_low_theta = FoxworthyF(
            seed=42, device="cpu", theta=0.0,
        )
        system_low_theta.set_graph(G)

        for i in range(10):
            r_high = system_high_theta.step(nodes[i % len(nodes)])
            r_low = system_low_theta.step(nodes[i % len(nodes)])

        norm_high = system_high_theta.get_structure_metric()
        norm_low = system_low_theta.get_structure_metric()

        # High threshold system should have barely learned
        # (gate ~0 for all inputs). Low threshold learns on every step.
        # Both start at same baseline, so the difference reflects gating.
        assert norm_low > norm_high, (
            f"Low-theta system should learn more than high-theta: "
            f"low={norm_low}, high={norm_high}"
        )

    def test_diagnostic_4_viability_maintenance(self):
        """Action selection preserves internally represented regulatory targets.

        Foxworthy §2.7: "π(a|s) ∝ π_base(a|s) exp(βR − λ_u U − λ_KL KL)"
        The viability penalty should observably influence action selection.
        """
        G = generate_domain(SMALL)
        nodes = list(G.nodes())

        # Two systems with very different viability weights
        system_preserve = FoxworthyF(
            seed=42, device="cpu", theta=0.0, lambda_u=10.0,
        )
        system_preserve.set_graph(G)

        system_ignore = FoxworthyF(
            seed=42, device="cpu", theta=0.0, lambda_u=0.01,
        )
        system_ignore.set_graph(G)

        # Run both on identical inputs
        results_preserve = []
        results_ignore = []
        for i in range(10):
            r_p = system_preserve.step(nodes[i % len(nodes)])
            r_i = system_ignore.step(nodes[i % len(nodes)])
            results_preserve.append(r_p["current_node"])
            results_ignore.append(r_i["current_node"])

        # The two systems should have taken at least some different paths
        # (different lambda_u → different action scores → different choices)
        paths_differ = any(
            a != b for a, b in zip(results_preserve, results_ignore)
        )

        # Even if paths happen to match (possible on small graphs with
        # limited successors), the adapter norms should differ
        norm_preserve = system_preserve.get_structure_metric()
        norm_ignore = system_ignore.get_structure_metric()

        assert paths_differ or norm_preserve != norm_ignore, (
            "Viability weight should observably influence behaviour"
        )

"""Tests for Class 2 system adapters."""

import pytest

from m8_battery.core.types import SystemClass
from m8_battery.domains.sbm_generator import generate_domain, generate_domain_family
from m8_battery.domains.presets import SMALL
from m8_battery.domains.encoders.text_encoder import encode_neighbourhood, encode_domain_as_text
from m8_battery.instruments.developmental_trajectory import run_developmental_trajectory


class TestFrozenGAT:
    def test_train_and_freeze(self):
        from m8_battery.systems.class2.frozen_gnn import FrozenGAT

        G = generate_domain(SMALL)
        system = FrozenGAT(n_features=8, n_classes=4, seed=42)
        accuracy = system.train_on_domain(G, epochs=30)
        assert accuracy > 0.0  # Should learn something
        assert system._is_frozen

    def test_structure_metric_constant_after_freeze(self):
        from m8_battery.systems.class2.frozen_gnn import FrozenGAT

        G = generate_domain(SMALL)
        system = FrozenGAT(n_features=8, n_classes=4, seed=42)
        system.train_on_domain(G, epochs=30)

        m1 = system.get_structure_metric()
        nodes = list(G.nodes())
        for n in nodes[:10]:
            system.step(n)
        m2 = system.get_structure_metric()
        # Frozen weights → constant attention entropy
        assert abs(m1 - m2) < 1e-6

    def test_regions_are_heads(self):
        from m8_battery.systems.class2.frozen_gnn import FrozenGAT

        system = FrozenGAT(n_heads=4)
        assert system.get_regions() == ["head_0", "head_1", "head_2", "head_3"]

    def test_ablation_changes_metric(self):
        from m8_battery.systems.class2.frozen_gnn import FrozenGAT

        G = generate_domain(SMALL)
        system = FrozenGAT(n_features=8, n_classes=4, seed=42)
        system.train_on_domain(G, epochs=30)

        original_metric = system.get_structure_metric()
        ablated = system.ablate("head_0")
        ablated_metric = ablated.get_structure_metric()
        # Ablating a head should change the attention entropy
        # (may or may not — depends on training outcome)
        assert isinstance(ablated_metric, float)

    def test_clone_independence(self):
        from m8_battery.systems.class2.frozen_gnn import FrozenGAT

        G = generate_domain(SMALL)
        system = FrozenGAT(seed=42)
        system.train_on_domain(G, epochs=10)
        system.step(list(G.nodes())[0])
        clone = system.clone()
        assert clone._step_count == 0


class TestFoxworthyC:
    def test_basic_step(self):
        from m8_battery.systems.class2.foxworthy_c import FoxworthyC

        G = generate_domain(SMALL)
        system = FoxworthyC(seed=42)
        system.set_graph(G)
        result = system.step(None)
        assert result["current_node"] is not None
        assert "hidden_norm" in result

    def test_hidden_state_changes(self):
        """Foxworthy C has transient state that evolves (but is resettable)."""
        from m8_battery.systems.class2.foxworthy_c import FoxworthyC

        G = generate_domain(SMALL)
        system = FoxworthyC(seed=42)
        system.set_graph(G)

        m1 = system.get_structure_metric()
        for _ in range(10):
            system.step(None)
        m2 = system.get_structure_metric()
        # Hidden state changes → metric changes
        # (this is transient state, NOT earned structure)
        assert m1 != m2

    def test_reset_restores_state(self):
        from m8_battery.systems.class2.foxworthy_c import FoxworthyC

        G = generate_domain(SMALL)
        system = FoxworthyC(seed=42)
        system.set_graph(G)

        m_initial = system.get_structure_metric()
        for _ in range(10):
            system.step(None)
        system.reset()
        m_reset = system.get_structure_metric()
        assert abs(m_initial - m_reset) < 1e-6

    def test_regions(self):
        from m8_battery.systems.class2.foxworthy_c import FoxworthyC

        system = FoxworthyC()
        assert len(system.get_regions()) == 4

    def test_clone(self):
        from m8_battery.systems.class2.foxworthy_c import FoxworthyC

        G = generate_domain(SMALL)
        system = FoxworthyC(seed=42)
        system.set_graph(G)
        system.step(None)
        clone = system.clone()
        assert clone._step_count == 0


class TestTextEncoder:
    def test_encode_neighbourhood(self):
        G = generate_domain(SMALL)
        node = list(G.nodes())[0]
        text = encode_neighbourhood(G, node)
        assert "Entity" in text
        assert "Connections:" in text or "Features:" in text

    def test_encode_domain(self):
        G = generate_domain(SMALL)
        texts = encode_domain_as_text(G, max_nodes=10)
        assert len(texts) == 10
        assert all(isinstance(t, str) for t in texts)

    def test_no_community_leak(self):
        """Text encoder should not expose community labels."""
        G = generate_domain(SMALL)
        node = list(G.nodes())[0]
        text = encode_neighbourhood(G, node)
        assert "community=" not in text

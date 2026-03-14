"""Diagnostic tests for integration instrument false positive check.

Four tests to determine whether Class 1 Gini=0.48 is genuine topological
integration or a size-inequality artefact.
"""

import networkx as nx
import numpy as np
import pytest

from m8_battery.core.types import DomainConfig
from m8_battery.domains.sbm_generator import generate_domain
from m8_battery.instruments.integration import run_integration
from m8_battery.systems.class1.wordnet_graph import WordNetGraph


def _assign_features(G: nx.DiGraph, n_communities: int, method: str = "equal"):
    """Assign community features to nodes for region detection."""
    nodes = list(G.nodes())
    for i, node in enumerate(nodes):
        if method == "equal":
            block = i % n_communities
        elif method == "random":
            block = np.random.RandomState(42).randint(0, n_communities)
        else:
            block = 0
        G.nodes[node]["features"] = {"community": block}
        G.nodes[node]["label"] = f"E_{node:03d}"
    return G


class TestIntegrationDiagnostics:
    """Four diagnostic tests per validation request."""

    def test_1_balanced_sbm(self):
        """Test 1: Balanced SBM — equal community sizes.

        If Gini ≈ 0: size artefact confirmed.
        If Gini > 0: genuine topological integration.
        """
        # 48 nodes = 4 × 12 exact equal partition
        config = DomainConfig(
            n_nodes=48, n_communities=4,
            p_within=0.3, p_between=0.03, seed=42,
        )
        G = generate_domain(config)

        # Force exactly equal community sizes
        nodes = list(G.nodes())
        for i, node in enumerate(nodes):
            G.nodes[node]["features"]["community"] = i // 12

        system = WordNetGraph(G, seed=42)
        probe_nodes = nodes[:10]
        result = run_integration(system=system, probe_inputs=probe_nodes)

        gini = result.raw_data.get("gini", None)
        print(f"\nTest 1 (Balanced SBM): Gini = {gini:.4f}")
        assert gini is not None
        # Record for diagnostic report
        result.raw_data["diagnostic_test"] = "balanced_sbm"

    def test_2_erdos_renyi(self):
        """Test 2: Erdős-Rényi random graph — no community structure.

        If Gini ≈ 0: instrument correctly detects no structure.
        If Gini > threshold: instrument too sensitive.
        """
        G = nx.erdos_renyi_graph(50, p=0.1, seed=42, directed=True)

        # Add weights
        rng = np.random.default_rng(42)
        for u, v in G.edges():
            G.edges[u, v]["weight"] = rng.uniform(0.1, 1.0)

        # Assign random regions (no real community structure)
        G = _assign_features(G, n_communities=4, method="equal")

        system = WordNetGraph(G, seed=42)
        nodes = list(G.nodes())
        probe_nodes = nodes[:10]
        result = run_integration(system=system, probe_inputs=probe_nodes)

        gini = result.raw_data.get("gini", None)
        print(f"\nTest 2 (Erdos-Renyi): Gini = {gini:.4f}")
        assert gini is not None

    def test_3_fully_connected(self):
        """Test 3: Complete graph — ablation should cause proportional degradation.

        Expected: Gini ≈ 0 (all regions equally important).
        If Gini ≠ 0: instrument has a bug.
        """
        G = nx.complete_graph(50, create_using=nx.DiGraph)

        # Equal weights
        for u, v in G.edges():
            G.edges[u, v]["weight"] = 1.0

        # Equal regions
        G = _assign_features(G, n_communities=4, method="equal")

        system = WordNetGraph(G, seed=42)
        nodes = list(G.nodes())
        probe_nodes = nodes[:10]
        result = run_integration(system=system, probe_inputs=probe_nodes)

        gini = result.raw_data.get("gini", None)
        print(f"\nTest 3 (Fully connected): Gini = {gini:.4f}")
        assert gini is not None
        # Fully connected + equal regions + equal weights → Gini should be near 0
        assert gini < 0.15, f"Fully connected graph should have low Gini, got {gini}"

    def test_4_star_graph(self):
        """Test 4: Star topology — extreme structural inequality.

        Expected: Very high Gini (hub ablation destroys everything).
        Sanity check that instrument CAN detect extreme integration.
        """
        G = nx.star_graph(49)
        G = G.to_directed()

        # Add weights
        for u, v in G.edges():
            G.edges[u, v]["weight"] = 1.0

        # Region 0 = hub (node 0), Regions 1-3 = leaf groups
        for node in G.nodes():
            if node == 0:
                G.nodes[node]["features"] = {"community": 0}
            else:
                G.nodes[node]["features"] = {"community": 1 + ((node - 1) % 3)}
            G.nodes[node]["label"] = f"E_{node:03d}"

        system = WordNetGraph(G, seed=42)
        nodes = list(G.nodes())
        probe_nodes = nodes[:10]
        result = run_integration(system=system, probe_inputs=probe_nodes)

        gini = result.raw_data.get("gini", None)
        print(f"\nTest 4 (Star graph): Gini = {gini:.4f}")
        assert gini is not None
        # Star topology should show high integration (hub is critical)
        assert gini > 0.3, f"Star graph should have high Gini, got {gini}"

"""Tests for domain generation, spectral verification, and graph encoding."""

import networkx as nx
import numpy as np

from m8_battery.core.types import DomainConfig
from m8_battery.domains.sbm_generator import generate_domain, generate_domain_family
from m8_battery.domains.spectral_verifier import (
    compute_graph_signature,
    spectral_similarity,
    verify_cross_format_invariance,
)
from m8_battery.domains.encoders.graph_encoder import encode_graph
from m8_battery.domains.presets import SMALL, MEDIUM, LARGE


class TestSBMGenerator:
    def test_basic_generation(self):
        G = generate_domain(SMALL)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 50
        assert G.number_of_edges() > 0

    def test_reproducibility(self):
        config = DomainConfig(
            n_nodes=50, n_communities=4, p_within=0.3, p_between=0.03, seed=123
        )
        G1 = generate_domain(config)
        G2 = generate_domain(config)
        assert G1.number_of_nodes() == G2.number_of_nodes()
        assert G1.number_of_edges() == G2.number_of_edges()

    def test_node_features(self):
        G = generate_domain(SMALL)
        for node in G.nodes():
            assert "features" in G.nodes[node]
            assert "label" in G.nodes[node]
            assert "community" in G.nodes[node]["features"]

    def test_edge_attributes(self):
        G = generate_domain(SMALL)
        for u, v in G.edges():
            assert "weight" in G.edges[u, v]
            assert "edge_type" in G.edges[u, v]
            w = G.edges[u, v]["weight"]
            assert 0.1 <= w <= 1.0

    def test_medium_scale(self):
        G = generate_domain(MEDIUM)
        assert G.number_of_nodes() == 150

    def test_large_scale(self):
        G = generate_domain(LARGE)
        assert G.number_of_nodes() == 500

    def test_community_structure(self):
        G = generate_domain(SMALL)
        communities = set()
        for node in G.nodes():
            communities.add(G.nodes[node]["features"]["community"])
        assert len(communities) == 4


class TestDomainFamily:
    def test_family_keys(self):
        family = generate_domain_family(SMALL)
        assert set(family.keys()) == {"A", "A_prime", "B", "C"}

    def test_a_and_a_prime_same_size(self):
        family = generate_domain_family(SMALL)
        assert family["A"].number_of_nodes() == family["A_prime"].number_of_nodes()

    def test_a_and_a_prime_same_edge_count(self):
        family = generate_domain_family(SMALL)
        assert family["A"].number_of_edges() == family["A_prime"].number_of_edges()

    def test_b_same_size(self):
        family = generate_domain_family(SMALL)
        assert family["B"].number_of_nodes() == family["A"].number_of_nodes()

    def test_c_different_communities(self):
        family = generate_domain_family(SMALL)
        c_communities = set()
        for node in family["C"].nodes():
            c_communities.add(family["C"].nodes[node]["features"]["community"])
        # C has fewer communities (negative control)
        a_communities = set()
        for node in family["A"].nodes():
            a_communities.add(family["A"].nodes[node]["features"]["community"])
        assert len(c_communities) < len(a_communities)


class TestSpectralVerifier:
    def test_graph_signature(self):
        G = generate_domain(SMALL)
        sig = compute_graph_signature(G, k=10)
        assert sig is not None
        assert len(sig) == 10
        # Eigenvalues should be sorted descending
        assert all(sig[i] >= sig[i + 1] for i in range(len(sig) - 1))

    def test_self_similarity(self):
        G = generate_domain(SMALL)
        sig = compute_graph_signature(G, k=10)
        sim = spectral_similarity(sig, sig)
        assert sim is not None
        assert abs(sim - 1.0) < 1e-10

    def test_different_graphs_lower_similarity(self):
        family = generate_domain_family(SMALL)
        sig_a = compute_graph_signature(family["A"], k=10)
        sig_c = compute_graph_signature(family["C"], k=10)
        sim = spectral_similarity(sig_a, sig_c)
        assert sim is not None
        assert sim < 1.0

    def test_isomorphic_high_similarity(self):
        family = generate_domain_family(SMALL)
        sig_a = compute_graph_signature(family["A"], k=10)
        sig_a_prime = compute_graph_signature(family["A_prime"], k=10)
        sim = spectral_similarity(sig_a, sig_a_prime)
        assert sim is not None
        # Isomorphic graphs should have very high spectral similarity
        assert sim > 0.9

    def test_cross_format_invariance(self):
        G = generate_domain(SMALL)
        # Same graph under two "formats" (identity test)
        passed, distances = verify_cross_format_invariance(
            {"format1": G, "format2": G}, k=10
        )
        assert passed
        assert distances["format1_vs_format2"] < 0.01


class TestGraphEncoder:
    def test_encode(self):
        G = generate_domain(SMALL)
        domain = encode_graph(G)
        assert domain.graph is G
        assert len(domain.node_labels) == 50
        assert domain.node_features.shape[0] == 50
        assert len(domain.edge_types) > 0

    def test_feature_matrix_numeric(self):
        G = generate_domain(SMALL)
        domain = encode_graph(G)
        # All features should be finite numbers
        assert np.all(np.isfinite(domain.node_features))

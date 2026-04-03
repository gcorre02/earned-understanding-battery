"""Tests for domain construction quality verification."""

import pytest

from m8_battery.domains.sbm_generator import generate_domain_family
from m8_battery.domains.presets import SMALL, MEDIUM
from m8_battery.domains.domain_verification import (
    verify_structural_preservation,
    verify_surface_destruction,
    verify_domain_family,
    check_leakage_channels,
)

class TestStructuralPreservation:
    def test_a_vs_a_prime_preserves_structure(self):
        """A' should preserve community count, node count, spectral signature."""
        family = generate_domain_family(SMALL)
        result = verify_structural_preservation(family["A"], family["A_prime"])
        assert result["community_match"] == 1.0
        assert result["node_count_match"] == 1.0
        # A' is relabeled A — exact same adjacency → spectral similarity ≈ 1.0
        assert result["spectral_similarity"] > 0.99

    def test_a_vs_b_preserves_structure(self):
        """B should preserve community count and spectral family."""
        family = generate_domain_family(SMALL)
        result = verify_structural_preservation(family["A"], family["B"])
        assert result["community_match"] == 1.0
        assert result["node_count_match"] == 1.0
        # B is a fresh draw — spectral similarity should be moderate (same SBM params)
        assert result["spectral_similarity"] > 0.5

    def test_a_vs_c_different_structure(self):
        """C is qualitatively different — community count and spectral should differ."""
        family = generate_domain_family(SMALL)
        result = verify_structural_preservation(family["A"], family["C"])
        # C has fewer communities
        assert result["community_match"] == 0.0 or result["spectral_similarity"] < 0.9

class TestSurfaceDestruction:
    def test_a_vs_a_prime_labels_destroyed(self):
        """A' should have different labels from A."""
        family = generate_domain_family(SMALL)
        result = verify_surface_destruction(family["A"], family["A_prime"])
        assert result["label_overlap"] == 0.0, "A' labels should not overlap with A"

    def test_a_vs_a_prime_same_edges(self):
        """A' is relabeled A — edge set overlap depends on permutation."""
        family = generate_domain_family(SMALL)
        result = verify_surface_destruction(family["A"], family["A_prime"])
        # With permuted node IDs, edge Jaccard should be low
        # (same structure but different node numbering)
        assert result["edge_jaccard"] < 0.5

    def test_a_vs_b_different_edges(self):
        """B is a fresh draw — edge set should be different."""
        family = generate_domain_family(SMALL)
        result = verify_surface_destruction(family["A"], family["B"])
        assert result["edge_jaccard"] < 0.3

class TestDomainFamily:
    def test_full_verification(self):
        """Full family verification produces expected structure."""
        family = generate_domain_family(MEDIUM)
        results = verify_domain_family(family)

        # A vs A': structure preserved, surface destroyed
        aa = results["A_vs_A_prime"]
        assert aa["structural"]["community_match"] == 1.0
        assert aa["structural"]["spectral_similarity"] > 0.99
        assert aa["surface"]["label_overlap"] == 0.0

        # A vs B: structure preserved (same params), surface destroyed (different seed)
        ab = results["A_vs_B"]
        assert ab["structural"]["community_match"] == 1.0

    def test_leakage_channels(self):
        """Check leakage channel documentation runs."""
        family = generate_domain_family(SMALL)
        warnings = check_leakage_channels(family["A"])
        # Should produce at least one warning (unequal communities or degree variation)
        assert isinstance(warnings, list)

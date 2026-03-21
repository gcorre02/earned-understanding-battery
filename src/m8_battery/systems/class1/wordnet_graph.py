"""System 1A — WordNet static graph.

A Class 1 system: fixed structure, no learning, no adaptation.
Navigates a static subgraph of WordNet via deterministic lookup.

Expected battery result: NEGATIVE on all instruments.
Structure metric is constant — no developmental trajectory.
"""

from __future__ import annotations

import pickle
from typing import Any

import igraph as ig
import networkx as nx
import numpy as np

from m8_battery.core.test_system import TestSystem


def _nx_to_igraph(G: nx.DiGraph) -> tuple[ig.Graph, dict[int, int], dict[int, int]]:
    """Convert networkx DiGraph to igraph, returning node mappings.

    Returns:
        (ig_graph, nx_to_ig_map, ig_to_nx_map)
    """
    nodes = sorted(G.nodes())
    nx_to_ig = {n: i for i, n in enumerate(nodes)}
    ig_to_nx = {i: n for i, n in enumerate(nodes)}

    edges = [(nx_to_ig[u], nx_to_ig[v]) for u, v in G.edges()]
    weights = [G.edges[u, v].get("weight", 1.0) for u, v in G.edges()]

    ig_g = ig.Graph(n=len(nodes), edges=edges, directed=True)
    ig_g.es["weight"] = weights

    return ig_g, nx_to_ig, ig_to_nx


class WordNetGraph(TestSystem):
    """Static graph navigator over a provided networkx graph.

    Although named 'WordNet', this adapter works with any networkx graph.
    For calibration, we pass SBM-generated graphs (synthetic entities).
    The WordNet name reflects its role: a static knowledge graph with
    fixed structure that never changes through operation.

    Uses igraph (C backend) for shortest-path computation. Networkx is
    retained for graph storage and attribute access. See F-020 governance.
    """

    def __init__(self, graph: nx.DiGraph, seed: int = 42) -> None:
        self._original_graph = graph.copy()
        self._graph = graph.copy()
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._current_node: int | None = None
        self._visit_counts: dict[int, int] = {}
        self._step_count = 0

        # Pre-compute structure metric (constant for Class 1)
        self._structure_metric = self._compute_structure_metric()

        # F-020: Build igraph representation + all-pairs next-hop cache.
        # igraph uses C for shortest paths — orders of magnitude faster
        # than networkx on pathological graphs. The graph is static so
        # the cache is valid for the lifetime of this instance.
        self._ig_graph, self._nx_to_ig, self._ig_to_nx = _nx_to_igraph(self._graph)
        self._next_hop_cache: dict[tuple, int | None] = {}
        self._build_next_hop_cache()

    def _build_next_hop_cache(self) -> None:
        """Pre-compute next hop for all reachable (source, target) pairs.

        Uses igraph's C-compiled shortest_paths for bulk computation.
        500-node graph: ~0.1s with igraph vs ~hours with networkx on
        pathological topologies.
        """
        n = self._ig_graph.vcount()
        if n == 0:
            return

        # igraph bulk shortest paths — returns matrix of path objects
        for source_ig in range(n):
            try:
                paths = self._ig_graph.get_shortest_paths(
                    source_ig, weights="weight", mode="out"
                )
                source_nx = self._ig_to_nx[source_ig]
                for target_ig, path in enumerate(paths):
                    if len(path) > 1:
                        self._next_hop_cache[(source_nx, self._ig_to_nx[target_ig])] = self._ig_to_nx[path[1]]
                    elif len(path) == 1:
                        self._next_hop_cache[(source_nx, self._ig_to_nx[target_ig])] = source_nx
            except Exception:
                pass

    def set_domain(self, graph: nx.DiGraph) -> None:
        """Switch to a new graph domain (near re-init for Class 1).

        Replaces the graph, rebuilds igraph cache and next-hop cache,
        resets navigation state. No learned state to preserve.
        """
        self._original_graph = graph.copy()
        self._graph = graph.copy()
        self._structure_metric = self._compute_structure_metric()
        self._ig_graph, self._nx_to_ig, self._ig_to_nx = _nx_to_igraph(self._graph)
        self._next_hop_cache = {}
        self._build_next_hop_cache()
        # Reset navigation
        nodes = list(self._graph.nodes())
        self._current_node = nodes[0] if nodes else None
        self._visit_counts = {}
        self._step_count = 0

    def reset(self) -> None:
        """Restore to initial state."""
        self._graph = self._original_graph.copy()
        self._rng = np.random.default_rng(self._seed)
        self._current_node = None
        self._visit_counts = {}
        self._step_count = 0

    def step(self, input_data: Any) -> Any:
        """Process one interaction — navigate to a node.

        input_data: target node ID (int) or None for random walk.
        Returns: dict with current_node, path, neighbours.
        """
        nodes = list(self._graph.nodes())
        if not nodes:
            return {"error": "empty graph"}

        if self._current_node is None:
            self._current_node = nodes[0]

        if input_data is not None and input_data in self._graph:
            # Navigate toward target — use cached next-hop (igraph, F-020)
            next_hop = self._next_hop_cache.get(
                (self._current_node, input_data)
            )
            if next_hop is not None:
                self._current_node = next_hop
        else:
            # Random walk — choose a random neighbour
            neighbours = list(self._graph.successors(self._current_node))
            if neighbours:
                self._current_node = self._rng.choice(neighbours)

        self._visit_counts[self._current_node] = (
            self._visit_counts.get(self._current_node, 0) + 1
        )
        self._step_count += 1

        return {
            "current_node": self._current_node,
            "step": self._step_count,
            "neighbours": list(self._graph.successors(self._current_node)),
        }

    def get_state(self) -> bytes:
        """Serialise state."""
        state = {
            "current_node": self._current_node,
            "visit_counts": self._visit_counts,
            "step_count": self._step_count,
            "rng_state": self._rng.bit_generator.state,
        }
        return pickle.dumps(state)

    def set_state(self, snapshot: bytes) -> None:
        """Restore from serialised state."""
        state = pickle.loads(snapshot)
        self._current_node = state["current_node"]
        self._visit_counts = state["visit_counts"]
        self._step_count = state["step_count"]
        self._rng.bit_generator.state = state["rng_state"]

    def get_structure_metric(self) -> float:
        """Return structure metric — CONSTANT for Class 1.

        Returns spectral gap of the graph (second-smallest Laplacian
        eigenvalue). This never changes because the graph is static.
        """
        return self._structure_metric

    def get_structure_distribution(self) -> dict[str, float]:
        """Per-community structure metrics — all constant."""
        communities: dict[int, list[int]] = {}
        for node in self._graph.nodes():
            block = self._graph.nodes[node].get("features", {}).get("community", 0)
            communities.setdefault(block, []).append(node)

        result = {}
        for comm_id, nodes in communities.items():
            subgraph = self._graph.subgraph(nodes)
            if subgraph.number_of_edges() > 0:
                result[f"community_{comm_id}"] = float(
                    nx.density(subgraph)
                )
            else:
                result[f"community_{comm_id}"] = 0.0
        return result

    def get_engagement_distribution(self) -> dict[str, float]:
        """Per-community visit frequency."""
        communities: dict[int, list[int]] = {}
        for node in self._graph.nodes():
            block = self._graph.nodes[node].get("features", {}).get("community", 0)
            communities.setdefault(block, []).append(node)

        total = sum(self._visit_counts.values()) or 1
        result = {}
        for comm_id, nodes in communities.items():
            visits = sum(self._visit_counts.get(n, 0) for n in nodes)
            result[f"community_{comm_id}"] = visits / total
        return result

    def ablate(self, region_id: str) -> TestSystem:
        """Remove a community from the graph."""
        comm_id = int(region_id.split("_")[-1])
        nodes_to_remove = [
            n for n in self._graph.nodes()
            if self._graph.nodes[n].get("features", {}).get("community", -1) == comm_id
        ]
        new_graph = self._graph.copy()
        new_graph.remove_nodes_from(nodes_to_remove)
        return WordNetGraph(new_graph, seed=self._seed)

    def perturb(self, region_id: str, method: str = "shuffle_weights") -> TestSystem:
        """Perturb a community's edge weights."""
        comm_id = int(region_id.split("_")[-1])
        new_graph = self._graph.copy()
        rng = np.random.default_rng(self._seed + 999)

        comm_nodes = set(
            n for n in new_graph.nodes()
            if new_graph.nodes[n].get("features", {}).get("community", -1) == comm_id
        )

        for u, v in new_graph.edges():
            if u in comm_nodes or v in comm_nodes:
                if method == "shuffle_weights":
                    new_graph.edges[u, v]["weight"] = rng.uniform(0.1, 1.0)
                elif method == "zero_weights":
                    new_graph.edges[u, v]["weight"] = 0.0

        return WordNetGraph(new_graph, seed=self._seed)

    def get_regions(self) -> list[str]:
        """Return community IDs as regions."""
        communities = set()
        for node in self._graph.nodes():
            block = self._graph.nodes[node].get("features", {}).get("community", 0)
            communities.add(block)
        return [f"community_{c}" for c in sorted(communities)]

    def clone(self) -> TestSystem:
        """Return independent copy."""
        new = WordNetGraph(self._original_graph, seed=self._seed)
        return new

    def _compute_structure_metric(self) -> float:
        """Compute spectral gap (algebraic connectivity).

        For a static graph this is constant — exactly what we want for
        a Class 1 system.

        Uses igraph + numpy for eigendecomposition. networkx's
        algebraic_connectivity uses ARPACK which hangs on certain
        graph topologies (F-020: seed 123 never completed).
        """
        if self._graph.number_of_nodes() < 2:
            return 0.0

        try:
            # Convert networkx undirected graph to igraph for eigendecomposition.
            # Must use networkx's to_undirected() to match historical metric values.
            G_und = self._graph.to_undirected()
            ig_und, _, _ = _nx_to_igraph(nx.DiGraph(G_und))
            ig_und_graph = ig_und.as_undirected(combine_edges="first")
            laplacian = np.array(ig_und_graph.laplacian(weights="weight"))
            eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
            return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        except Exception:
            return 0.0

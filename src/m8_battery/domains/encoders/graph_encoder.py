"""Graph encoder — native networkx format for graph-native systems."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class GraphDomain:
    """A domain encoded as native graph structures."""
    graph: nx.DiGraph
    node_features: np.ndarray  # (n_nodes, n_features)
    node_labels: list[str]
    edge_types: list[str]  # unique edge type labels


def encode_graph(G: nx.DiGraph) -> GraphDomain:
    """Encode a generated SBM graph as a GraphDomain.

    Extracts node features into a numpy array and catalogues edge types.
    """
    nodes = sorted(G.nodes())
    node_labels = [G.nodes[n].get("label", f"E_{n:03d}") for n in nodes]

    # Extract feature matrix
    feature_keys = None
    rows = []
    for n in nodes:
        feats = G.nodes[n].get("features", {})
        if feature_keys is None:
            feature_keys = sorted(
                k for k in feats.keys()
                if isinstance(feats[k], (int, float))
            )
        row = [float(feats.get(k, 0.0)) for k in feature_keys]
        rows.append(row)

    node_features = np.array(rows) if rows else np.empty((len(nodes), 0))

    # Catalogue edge types
    edge_types = sorted({
        G.edges[u, v].get("edge_type", "default")
        for u, v in G.edges()
    })

    return GraphDomain(
        graph=G,
        node_features=node_features,
        node_labels=node_labels,
        edge_types=edge_types,
    )

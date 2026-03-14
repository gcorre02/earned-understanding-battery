"""Text encoder — structured text descriptions for LLM-based systems."""

from __future__ import annotations

import networkx as nx


def encode_neighbourhood(G: nx.DiGraph, node: int, max_neighbours: int = 10) -> str:
    """Encode a node's local neighbourhood as structured text.

    Returns a text description suitable for LLM processing:
    node label, features, and connections to neighbours.
    """
    label = G.nodes[node].get("label", f"E_{node:03d}")
    features = G.nodes[node].get("features", {})

    lines = [f"Entity {label}:"]

    # Node features
    feat_parts = []
    for k, v in sorted(features.items()):
        if k == "community":
            continue  # Don't leak community labels
        if isinstance(v, float):
            feat_parts.append(f"{k}={v:.2f}")
        else:
            feat_parts.append(f"{k}={v}")
    if feat_parts:
        lines.append(f"  Features: {', '.join(feat_parts)}")

    # Connections
    successors = list(G.successors(node))[:max_neighbours]
    if successors:
        lines.append("  Connections:")
        for s in successors:
            s_label = G.nodes[s].get("label", f"E_{s:03d}")
            edge_data = G.edges[node, s]
            weight = edge_data.get("weight", 0.0)
            etype = edge_data.get("edge_type", "connected")
            lines.append(f"    -> {s_label} ({etype}, weight={weight:.2f})")

    return "\n".join(lines)


def encode_domain_as_text(G: nx.DiGraph, max_nodes: int = 50) -> list[str]:
    """Encode an entire domain as a list of text descriptions.

    Each element describes one node's neighbourhood.
    """
    nodes = sorted(G.nodes())[:max_nodes]
    return [encode_neighbourhood(G, n) for n in nodes]

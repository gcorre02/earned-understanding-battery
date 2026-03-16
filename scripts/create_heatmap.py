"""Static 3x3 heatmap figure — 9 system traversal comparison.

Each panel shows one system's final state: node visit frequency as
colour/size, edge traversal frequency as line thickness. Dark background.

Usage:
    .venv/bin/python scripts/create_heatmap.py

Output:
    results/animation/m8-calibration-heatmap-9panel.png
    results/animation/m8-calibration-heatmap-3c-detail.png
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_PATH = Path(__file__).parent.parent / "results" / "animation" / "traversal_data.json"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "animation"

BG = "#06060f"

SYSTEMS = ["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C"]

SYSTEM_NAMES = {
    "1A": "1A WordNet", "1B": "1B Rule Nav", "1C": "1C Fox A",
    "2A": "2A TinyLlama", "2B": "2B Frozen GAT", "2C": "2C Fox C",
    "3A": "3A DQN", "3B": "3B Curiosity", "3C": "3C Fox F",
}

PALETTES = {
    "1A": "Greys", "1B": "Greys", "1C": "Purples",
    "2A": "Blues", "2B": "Blues", "2C": "Oranges",
    "3A": "Reds", "3B": "Greens", "3C": "YlOrBr",
}

EDGE_COLOURS = {
    "1A": "#888888", "1B": "#888888", "1C": "#9070a0",
    "2A": "#4080b0", "2B": "#4080b0", "2C": "#c07020",
    "3A": "#c02030", "3B": "#209040", "3C": "#c09020",
}

CLASS_LABELS = {0: "CLASS 1\nFIXED", 1: "CLASS 2\nFROZEN", 2: "CLASS 3\nLEARNING"}


def community_layout(graph_data, width=10, height=10):
    """3x2 grid communities, grid nodes within."""
    comms = {}
    for n in graph_data["nodes"]:
        comms.setdefault(n["community"], []).append(n["id"])

    margin = 0.5
    uw, uh = width - 2 * margin, height - 2 * margin
    cw, ch = uw / 3, uh / 2

    positions = {}
    for i, cid in enumerate(sorted(comms)):
        cx = -uw / 2 + cw * (i % 3 + 0.5)
        cy = -uh / 2 + ch * (i // 3 + 0.5)
        nids = comms[cid]
        n = len(nids)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = max(1, math.ceil(n / cols))
        pad = cw * 0.1
        aw, ah = cw - 2 * pad, ch - 2 * pad
        sx, sy = aw / max(cols, 1), ah / max(rows, 1)
        for j, nid in enumerate(nids):
            r, c = j // cols, j % cols
            positions[nid] = (
                cx - aw / 2 + c * sx + sx / 2,
                cy - ah / 2 + r * sy + sy / 2,
            )
    return positions


def compute_traversal_stats(traversal):
    """Count node visits and edge traversals."""
    node_visits = Counter()
    edge_traversals = Counter()

    for i, nid in enumerate(traversal):
        if nid < 0:
            continue
        node_visits[nid] += 1
        if i > 0:
            prev = traversal[i - 1]
            if prev >= 0 and prev != nid:
                ek = (min(prev, nid), max(prev, nid))
                edge_traversals[ek] += 1

    return node_visits, edge_traversals


def draw_panel(ax, sys_id, graph_data, traversal, layout):
    """Draw one system's panel."""
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    node_visits, edge_traversals = compute_traversal_stats(traversal)
    all_nodes = [n["id"] for n in graph_data["nodes"]]
    cmap = plt.cm.get_cmap(PALETTES[sys_id])
    edge_color = EDGE_COLOURS[sys_id]

    # Draw traversed edges
    for (u, v), count in edge_traversals.items():
        if u not in layout or v not in layout:
            continue
        x0, y0 = layout[u]
        x1, y1 = layout[v]
        lw = 0.3 + count * 0.4
        alpha = min(0.8, 0.15 + count * 0.1)
        ax.plot([x0, x1], [y0, y1], linewidth=lw, alpha=alpha, color=edge_color,
                solid_capstyle="round")

    # Separate visited and unvisited
    visited_ids = [n for n in all_nodes if node_visits.get(n, 0) > 0 and n in layout]
    unvisited_ids = [n for n in all_nodes if node_visits.get(n, 0) == 0 and n in layout]

    # Unvisited: faint dots
    if unvisited_ids:
        ux = [layout[n][0] for n in unvisited_ids]
        uy = [layout[n][1] for n in unvisited_ids]
        ax.scatter(ux, uy, s=2, c="#222222", alpha=0.25, zorder=1)

    # Visited: colour + size by visit count
    if visited_ids:
        max_v = max(node_visits.values())
        vx = [layout[n][0] for n in visited_ids]
        vy = [layout[n][1] for n in visited_ids]
        counts = [node_visits[n] for n in visited_ids]
        normed = [c / max_v for c in counts]
        sizes = [max(5, n * 180) for n in normed]

        ax.scatter(vx, vy, s=sizes, c=normed, cmap=PALETTES[sys_id],
                   vmin=0, vmax=1, alpha=0.9, edgecolors="none", zorder=2)

    ax.set_title(SYSTEM_NAMES[sys_id], color="white", fontsize=10, pad=4,
                 fontfamily="sans-serif")


def create_9panel(graph_data, systems_data, layout):
    """Create the 3x3 figure."""
    fig = plt.figure(figsize=(16, 16), facecolor=BG)
    gs = gridspec.GridSpec(3, 3, hspace=0.18, wspace=0.08,
                           left=0.06, right=0.98, top=0.95, bottom=0.02)

    for idx, sys_id in enumerate(SYSTEMS):
        row, col = idx // 3, idx % 3
        ax = fig.add_subplot(gs[row, col])
        traversal = systems_data.get(sys_id, {}).get("traversal", [])
        draw_panel(ax, sys_id, graph_data, traversal, layout)

    # Class labels
    for row, label in CLASS_LABELS.items():
        fig.text(0.015, 0.82 - row * 0.33, label, color="#555555",
                 fontsize=8, va="center", ha="left", fontfamily="sans-serif",
                 linespacing=1.4)

    out = OUTPUT_DIR / "m8-calibration-heatmap-9panel.png"
    fig.savefig(out, dpi=300, facecolor=BG)
    print(f"Saved: {out}")
    plt.close(fig)
    return out


def create_3c_detail(graph_data, systems_data, layout):
    """Close-up of 3C panel."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor=BG)
    traversal = systems_data.get("3C", {}).get("traversal", [])
    draw_panel(ax, "3C", graph_data, traversal, layout)
    ax.set_title("System 3C — Foxworthy Variant F", color="#ffd040",
                 fontsize=14, pad=8, fontfamily="sans-serif")

    out = OUTPUT_DIR / "m8-calibration-heatmap-3c-detail.png"
    fig.savefig(out, dpi=300, facecolor=BG, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    data = json.loads(DATA_PATH.read_text())
    graph_data = data["graph"]
    systems_data = data["systems"]

    print(f"Loaded: {len(graph_data['nodes'])} nodes, {len(systems_data)} systems")

    layout = community_layout(graph_data)

    create_9panel(graph_data, systems_data, layout)
    create_3c_detail(graph_data, systems_data, layout)
    print("Done.")


if __name__ == "__main__":
    main()

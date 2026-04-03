"""Manim animation — System 3C (Foxworthy F) Growing Graph of Light.

Single panel: nodes appear on discovery, edges grow on traversal.
Gold light on dark background. The exploration → consolidation transition.

Usage:
    cd m8-battery
    manim -pql scripts/manim_3c.py GrowingGraph3C   # low quality preview
    manim -pqh scripts/manim_3c.py GrowingGraph3C   # high quality
"""
from manim import *
import json
import math
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "results" / "animation" / "traversal_data.json"

# 3C gold colour
GOLD = "#ffd040"
GOLD_DIM = "#aa8020"
GOLD_BRIGHT = "#ffee80"
BG = "#06060f"
BORDER = "#151520"

MAX_STEPS = 120  # enough to see exploration + some consolidation
STEP_DURATION = 0.3  # seconds per step

def community_layout_2d(graph_data, width=12, height=7):
    """Communities in 3x2 grid, nodes in grid within. Centered at origin."""
    comms = {}
    for n in graph_data["nodes"]:
        comms.setdefault(n["community"], []).append(n["id"])

    margin = 0.5
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    grid_cols, grid_rows = 3, 2
    cell_w = usable_w / grid_cols
    cell_h = usable_h / grid_rows

    centroids = {}
    for i, cid in enumerate(sorted(comms)):
        gc = i % grid_cols
        gr = i // grid_cols
        cx = -usable_w / 2 + cell_w * (gc + 0.5)
        cy = -usable_h / 2 + cell_h * (gr + 0.5)
        centroids[cid] = (cx, cy)

    pad = cell_w * 0.1
    positions = {}
    for cid, nids in comms.items():
        cx, cy = centroids[cid]
        n = len(nids)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows_n = max(1, math.ceil(n / cols))
        area_w = cell_w - 2 * pad
        area_h = cell_h - 2 * pad
        sp_x = area_w / max(cols, 1)
        sp_y = area_h / max(rows_n, 1)
        for j, nid in enumerate(nids):
            r = j // cols
            c = j % cols
            px = cx - area_w / 2 + c * sp_x + sp_x / 2
            py = cy - area_h / 2 + r * sp_y + sp_y / 2
            positions[nid] = (px, py)

    return positions

class GrowingGraph3C(Scene):
    def construct(self):
        self.camera.background_color = BG

        # Load data
        data = json.loads(DATA_PATH.read_text())
        graph_data = data["graph"]
        sdata = data["systems"]["3C"]
        traversal = sdata["traversal"][:MAX_STEPS]

        # Layout
        positions = community_layout_2d(graph_data)

        # Border
        border = Rectangle(width=13, height=7.5, stroke_color=BORDER,
                           stroke_width=1, fill_opacity=0)
        self.add(border)

        # Track state
        node_dots = {}        # nid -> Dot mobject
        edge_lines = {}       # (a,b) -> Line mobject
        discovered = set()
        seen_edges = set()

        # Process traversal step by step
        for i, nid in enumerate(traversal):
            if nid < 0 or nid not in positions:
                continue

            anims = []

            # Discover new node
            if nid not in discovered:
                discovered.add(nid)
                px, py = positions[nid]
                dot = Dot(point=[px, py, 0], radius=0.06, color=GOLD)
                dot.set_opacity(0)
                self.add(dot)
                node_dots[nid] = dot
                anims.append(dot.animate.set_opacity(0.8))

            # Form edge from previous node
            if i > 0:
                prev = traversal[i - 1]
                if prev >= 0 and prev != nid and prev in positions:
                    edge_key = (min(prev, nid), max(prev, nid))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        pa = positions[prev]
                        pb = positions[nid]
                        line = Line(
                            start=[pa[0], pa[1], 0],
                            end=[pb[0], pb[1], 0],
                            stroke_color=GOLD_DIM,
                            stroke_width=1.5,
                            stroke_opacity=0.7,
                        )
                        self.add(line)
                        edge_lines[edge_key] = line
                        anims.append(Create(line, run_time=STEP_DURATION))
                    else:
                        # Reinforce existing edge — thicken
                        existing = edge_lines[edge_key]
                        new_width = min(existing.stroke_width + 0.3, 5.0)
                        anims.append(
                            existing.animate.set_stroke(
                                color=GOLD, width=new_width, opacity=min(0.9, existing.stroke_opacity + 0.05)
                            )
                        )

            # Move marker (bright glow at current position)
            if anims:
                self.play(*anims, run_time=STEP_DURATION, rate_func=smooth)
            else:
                self.wait(STEP_DURATION * 0.3)

        # Final hold
        self.wait(2)

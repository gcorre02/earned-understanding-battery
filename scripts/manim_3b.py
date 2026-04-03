"""Manim — System 3B (Curiosity Agent) Growing Graph of Light.

Uses Manim's native Graph class. Vertices and edges added incrementally
as the curiosity agent explores. Green expanding frontier on dark canvas.

Usage:
    manim -pql scripts/manim_3b.py GrowingGraph3B
    manim -pqh scripts/manim_3b.py GrowingGraph3B
"""
from manim import *
import json
import math
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "results" / "animation" / "traversal_data.json"

GREEN = "#40e060"
GREEN_DIM = "#1a5028"
GREEN_GLOW = "#90ffa0"
BG = "#06060f"

def community_positions(graph_data, width=12.5, height=6.8):
    """3x2 grid communities, grid nodes within. Returns {nid: [x, y, 0]}."""
    comms = {}
    for n in graph_data["nodes"]:
        comms.setdefault(n["community"], []).append(n["id"])

    margin = 0.3
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
            positions[nid] = [
                cx - aw / 2 + c * sx + sx / 2,
                cy - ah / 2 + r * sy + sy / 2,
                0,
            ]
    return positions

def get_meaningful_events(traversal, positions):
    """Extract events where new nodes/edges appear. Group into waves."""
    discovered = set()
    seen_edges = set()
    events = []

    for i, nid in enumerate(traversal):
        if nid < 0 or nid not in positions:
            continue

        new_nodes = []
        new_edges = []

        if nid not in discovered:
            discovered.add(nid)
            new_nodes.append(nid)

        if i > 0:
            prev = traversal[i - 1]
            if prev >= 0 and prev != nid and prev in positions:
                ek = (min(prev, nid), max(prev, nid))
                if ek not in seen_edges:
                    seen_edges.add(ek)
                    new_edges.append((prev, nid))

        if new_nodes or new_edges:
            events.append({"nodes": new_nodes, "edges": new_edges, "current": nid})

    # Group into waves of 3-5 events for smoother pacing
    waves = []
    wave_size = 4
    for i in range(0, len(events), wave_size):
        chunk = events[i:i + wave_size]
        wave_nodes = []
        wave_edges = []
        last_current = chunk[-1]["current"]
        for ev in chunk:
            wave_nodes.extend(ev["nodes"])
            wave_edges.extend(ev["edges"])
        waves.append({"nodes": wave_nodes, "edges": wave_edges, "current": last_current})

    return waves

class GrowingGraph3B(Scene):
    def construct(self):
        self.camera.background_color = BG

        data = json.loads(DATA_PATH.read_text())
        graph_data = data["graph"]
        traversal = data["systems"]["3B"]["traversal"][:200]
        pos = community_positions(graph_data)

        waves = get_meaningful_events(traversal, pos)
        print(f"Animation waves: {len(waves)}")

        # We'll manage vertices and edges manually (Graph class
        # doesn't handle incremental addition well with custom positions)

        # Bright marker for current position
        marker = Dot(radius=0.09, color=GREEN_GLOW, fill_opacity=0.9)
        marker.set_z_index(10)
        self.add(marker)

        self.wait(0.5)

        node_dots = {}

        for wave in waves:
            anims = []

            # New vertices — small dots fading in
            for nid in wave["nodes"]:
                if nid not in node_dots:
                    p = pos[nid]
                    dot = Dot(point=p, radius=0.035, color=GREEN,
                              fill_opacity=0, stroke_width=0)
                    dot.set_z_index(5)
                    self.add(dot)
                    node_dots[nid] = dot
                    anims.append(dot.animate.set_fill(opacity=0.65))

            # New edges — grow from source
            for src, tgt in wave["edges"]:
                # Ensure source dot exists
                for v in (src, tgt):
                    if v not in node_dots:
                        dot = Dot(point=pos[v], radius=0.035, color=GREEN,
                                  fill_opacity=0.65, stroke_width=0)
                        dot.set_z_index(5)
                        self.add(dot)
                        node_dots[v] = dot

                line = Line(
                    pos[src], pos[tgt],
                    stroke_color=GREEN_DIM,
                    stroke_width=1.5,
                    stroke_opacity=0.5,
                )
                line.set_z_index(1)
                self.add(line)
                anims.append(
                    GrowFromPoint(line, point=pos[src], run_time=0.5)
                )

            # Move marker
            cp = pos.get(wave["current"])
            if cp:
                anims.append(marker.animate.move_to(cp))

            if anims:
                self.play(
                    AnimationGroup(*anims, lag_ratio=0.15),
                    run_time=0.6,
                    rate_func=smooth,
                )

        # Fade marker, hold final state
        self.play(marker.animate.set_opacity(0), run_time=0.8)
        self.wait(2)

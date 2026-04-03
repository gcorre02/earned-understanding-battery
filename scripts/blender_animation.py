"""Blender — M8 Battery: 9 Systems, Growing Graphs of Light.

Each panel starts empty. Nodes appear on first visit, edges grow on
traversal. Only 3C develops internal structure after exploration.

Run:
    blender --background --python scripts/blender_animation.py
    blender --background --python scripts/blender_animation.py -- --render

Reads: results/animation/traversal_data.json
"""
import bpy
import bmesh
import json
import math
import sys
from pathlib import Path
from mathutils import Vector

DATA_PATH = Path(__file__).parent.parent / "results" / "animation" / "traversal_data.json"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "animation"

FPS = 30
RES_X, RES_Y = 1920, 1080

# Pacing: ~4 steps/sec during fast phase, ~2 steps/sec during Class 3 focus
STEPS_FAST = 100       # steps shown during 0:03-0:20 (17s, ~6 steps/s)
STEPS_CLASS3 = 100     # additional steps during 0:38-0:55 (17s, ~6 steps/s)

# Timing (seconds)
T = {
    "title_end": 3,
    "all_start": 3,
    "all_end": 20,
    "c1_dim": 20,
    "c2_dim": 28,
    "c3_start": 38,
    "c3_end": 55,
    "hold_start": 55,
    "hold_end": 60,
    "total": 60,
}
# Convert to frames
F = {k: int(v * FPS) for k, v in T.items()}

BOARD_SIZE = 3.2
BOARD_GAP = 0.2
GRID_W = 3 * BOARD_SIZE + 2 * BOARD_GAP  # horizontal span
GRID_H = 3 * BOARD_SIZE + 2 * BOARD_GAP  # vertical span
GRID_OX = -GRID_W / 2 + BOARD_SIZE / 2
GRID_OY = GRID_H / 2 - BOARD_SIZE / 2

NODE_RADIUS = 0.035
EDGE_BEVEL = 0.008  # curve bevel depth for traces

SYSTEM_GRID = {
    "1A": (0, 0), "1B": (1, 0), "1C": (2, 0),
    "2A": (0, 1), "2B": (1, 1), "2C": (2, 1),
    "3A": (0, 2), "3B": (1, 2), "3C": (2, 2),
}

SYSTEM_COLOURS = {
    "1A": (0.878, 0.910, 0.941, 1),  # #e0e8f0
    "1B": (0.753, 0.816, 0.910, 1),  # #c0d0e8
    "1C": (0.816, 0.690, 0.878, 1),  # #d0b0e0
    "2A": (0.502, 0.878, 0.941, 1),  # #80e0f0
    "2B": (0.376, 0.565, 0.753, 1),  # #6090c0
    "2C": (0.941, 0.627, 0.188, 1),  # #f0a030
    "3A": (1.000, 0.188, 0.251, 1),  # #ff3040
    "3B": (0.251, 0.878, 0.376, 1),  # #40e060
    "3C": (1.000, 0.816, 0.251, 1),  # #ffd040
}

COL_BG = (0.024, 0.024, 0.059, 1)       # #06060f
COL_BORDER = (0.082, 0.082, 0.125, 1)   # #151520

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in [bpy.data.meshes, bpy.data.materials, bpy.data.curves,
                  bpy.data.cameras, bpy.data.lights]:
        for item in block:
            block.remove(item)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)

def setup_render():
    scene = bpy.context.scene
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = name
            break
        except TypeError:
            continue
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = F["total"]
    if hasattr(scene.eevee, "use_bloom"):
        scene.eevee.use_bloom = True
        scene.eevee.bloom_threshold = 0.5
        scene.eevee.bloom_intensity = 0.5
    scene.eevee.taa_render_samples = 32
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(OUTPUT_DIR / "frames" / "frame_")
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    if world.use_nodes:
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = COL_BG
            bg.inputs["Strength"].default_value = 0.1
    scene.world = world

def board_origin(sid):
    c, r = SYSTEM_GRID[sid]
    return Vector((c * (BOARD_SIZE + BOARD_GAP) + GRID_OX,
                   -r * (BOARD_SIZE + BOARD_GAP) + GRID_OY, 0))

GRID_TOTAL = max(GRID_W, GRID_H)  # for camera

def community_layout(graph_data):
    """Communities in 2x3 grid within each board. Nodes in grid within each community."""
    comms = {}
    for n in graph_data["nodes"]:
        comms.setdefault(n["community"], []).append(n["id"])

    margin = BOARD_SIZE * 0.08
    usable = BOARD_SIZE - 2 * margin

    # Place communities in a 3-col x 2-row grid filling the board
    nc = len(comms)
    grid_cols, grid_rows = 3, 2
    cell_w = usable / grid_cols
    cell_h = usable / grid_rows

    centroids = {}
    for i, cid in enumerate(sorted(comms)):
        gc = i % grid_cols
        gr = i // grid_cols
        cx = -usable / 2 + cell_w * (gc + 0.5)
        cy = -usable / 2 + cell_h * (gr + 0.5)
        centroids[cid] = (cx, cy)

    # Place nodes in a grid within each community cell
    pad = cell_w * 0.1  # padding within cell
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

def mat(name, color, emission=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = 0.4
    if emission > 0:
        b.inputs["Emission Color"].default_value = color
        b.inputs["Emission Strength"].default_value = emission
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
        if hasattr(m, "blend_method"):
            m.blend_method = "BLEND"
    return m

def make_node_mesh():
    """Flat disc — just a circle, no sphere."""
    m = bpy.data.meshes.new("node_mesh")
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, segments=8, radius=NODE_RADIUS)
    bm.to_mesh(m)
    bm.free()
    return m

def make_edge_curve(name, src_pos, tgt_pos, material):
    """Create a thin glowing curve between two points."""
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = EDGE_BEVEL
    curve.bevel_resolution = 1
    spline = curve.splines.new("POLY")
    spline.points.add(1)  # 2 points total
    spline.points[0].co = (src_pos[0], src_pos[1], 0.005, 1)
    spline.points[1].co = (tgt_pos[0], tgt_pos[1], 0.005, 1)
    curve.materials.append(material)
    obj = bpy.data.objects.new(name, curve)
    return obj

def create_panel_border(sid, col):
    """Thin border around each panel."""
    origin = board_origin(sid)
    half = BOARD_SIZE / 2 + 0.05
    verts = [(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)]

    curve = bpy.data.curves.new(f"border_{sid}", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.01
    spline = curve.splines.new("POLY")
    spline.points.add(4)
    for i, (x, y, z) in enumerate(verts):
        spline.points[i].co = (x, y, z, 1)
    spline.points[4].co = (verts[0][0], verts[0][1], 0, 1)  # close
    spline.use_cyclic_u = True

    obj = bpy.data.objects.new(f"border_{sid}", curve)
    obj.location = origin
    border_mat = mat(f"border_mat_{sid}", COL_BORDER, emission=0.1)
    curve.materials.append(border_mat)
    col.objects.link(obj)

def build_system(sid, sdata, graph_data, node_pos, node_mesh, col):
    """Build one system panel: nodes appear on discovery, edges grow on traversal."""
    origin = board_origin(sid)
    colour = SYSTEM_COLOURS[sid]
    traversal = sdata.get("traversal", [])
    if not traversal:
        return

    sys_class = int(sid[0])

    # Determine frame range for this system's animation
    if sys_class <= 2:
        f_start = F["all_start"]
        f_end = F["all_end"]
        steps = traversal[:STEPS_FAST]
    else:
        f_start = F["all_start"]
        f_end = F["c3_end"]
        steps = traversal[:STEPS_FAST + STEPS_CLASS3]

    n_steps = len(steps)
    if n_steps == 0:
        return

    # Node material
    node_mat = mat(f"nmat_{sid}", colour, emission=3.0)

    # Edge material
    edge_mat = mat(f"emat_{sid}", colour, emission=2.0)

    # Track discovery order
    discovered_nodes = {}  # nid -> step_index
    edge_list = []  # (step_index, a, b)
    seen_edges = set()

    for i, nid in enumerate(steps):
        if nid >= 0 and nid not in discovered_nodes:
            discovered_nodes[nid] = i
        if i > 0:
            a, b = steps[i - 1], steps[i]
            if a >= 0 and b >= 0 and a != b:
                edge_key = (min(a, b), max(a, b))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edge_list.append((i, a, b))

    # Create node objects — hidden, appear on discovery
    for nid, disc_step in discovered_nodes.items():
        if nid not in node_pos:
            continue
        px, py = node_pos[nid]
        obj = bpy.data.objects.new(f"n_{sid}_{nid}", node_mesh.copy())
        obj.location = (origin.x + px, origin.y + py, 0.02)
        obj.data.materials.append(node_mat)
        col.objects.link(obj)

        # Fade in: scale 0 -> 1 over ~0.3s (9 frames)
        appear_frame = int(f_start + (disc_step / n_steps) * (f_end - f_start))
        obj.scale = (0, 0, 0)
        obj.keyframe_insert(data_path="scale", frame=max(1, appear_frame - 1))
        obj.scale = (1, 1, 1)
        obj.keyframe_insert(data_path="scale", frame=appear_frame + 9)

    # Create edges as curves — hidden, appear on traversal
    for step_i, a, b in edge_list:
        if a not in node_pos or b not in node_pos:
            continue
        pa, pb = node_pos[a], node_pos[b]
        src = (origin.x + pa[0], origin.y + pa[1])
        tgt = (origin.x + pb[0], origin.y + pb[1])
        if abs(src[0] - tgt[0]) + abs(src[1] - tgt[1]) < 0.001:
            continue

        obj = make_edge_curve(f"e_{sid}_{a}_{b}", src, tgt, edge_mat)
        col.objects.link(obj)

        appear_frame = int(f_start + (step_i / n_steps) * (f_end - f_start))
        obj.scale = (0, 0, 0)
        obj.keyframe_insert(data_path="scale", frame=max(1, appear_frame - 1))
        obj.scale = (1, 1, 1)
        obj.keyframe_insert(data_path="scale", frame=appear_frame + 5)

    # Current position marker — bright flat disc
    mk_mat = mat(f"mkm_{sid}", colour, emission=10.0)
    mk_mesh = bpy.data.meshes.new(f"mk_{sid}")
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, segments=10, radius=NODE_RADIUS * 2.5)
    bm.to_mesh(mk_mesh)
    bm.free()
    marker = bpy.data.objects.new(f"mk_{sid}", mk_mesh)
    marker.data.materials.append(mk_mat)
    col.objects.link(marker)

    # Smooth movement (default bezier interpolation)
    for i, nid in enumerate(steps):
        if nid < 0 or nid not in node_pos:
            continue
        px, py = node_pos[nid]
        frame = int(f_start + (i / n_steps) * (f_end - f_start))
        marker.location = (origin.x + px, origin.y + py, 0.05)
        marker.keyframe_insert(data_path="location", frame=frame)

    # Dimming for Class 1/2
    if sys_class == 1:
        dim_frame = F["c1_dim"]
    elif sys_class == 2:
        dim_frame = F["c2_dim"]
    else:
        dim_frame = None

    if dim_frame:
        # Hide marker
        marker.hide_render = False
        marker.keyframe_insert(data_path="hide_render", frame=dim_frame)
        marker.hide_render = True
        marker.keyframe_insert(data_path="hide_render", frame=dim_frame + 1)

    return len(discovered_nodes), len(edge_list)

def create_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type = "ORTHO"
    # ortho_scale = vertical extent visible. For 16:9, horizontal = 1.78x wider.
    # Grid is ~10.4 square. Height is the tight constraint.
    # Also check if width needs more room: GRID_W / (RES_X/RES_Y)
    scale_for_height = GRID_H + 0.8
    scale_for_width = (GRID_W + 0.8) / (RES_X / RES_Y)
    cam_data.ortho_scale = max(scale_for_height, scale_for_width)
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (0, 0, 20)
    cam.rotation_euler = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

def create_lights():
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = 0.5  # dim — let emission do the work
    sun.color = (0.9, 0.9, 1.0)
    obj = bpy.data.objects.new("Sun", sun)
    obj.location = (0, 0, 15)
    bpy.context.scene.collection.objects.link(obj)

def main():
    render = "--render" in sys.argv

    print("=== M8: Growing Graphs of Light ===")

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found.")
        sys.exit(1)

    data = json.loads(DATA_PATH.read_text())
    graph_data = data["graph"]
    systems_data = data["systems"]

    print(f"Loaded: {len(graph_data['nodes'])} nodes, {len(systems_data)} systems")

    node_pos = community_layout(graph_data)

    clear_scene()
    setup_render()

    node_mesh = make_node_mesh()

    panels_col = bpy.data.collections.new("Panels")
    bpy.context.scene.collection.children.link(panels_col)

    print("Building panels...")
    for sid in SYSTEM_GRID:
        create_panel_border(sid, panels_col)
        sdata = systems_data.get(sid, {})
        result = build_system(sid, sdata, graph_data, node_pos, node_mesh, panels_col)
        if result:
            nodes, edges = result
            print(f"  [{sid}] {nodes} nodes discovered, {edges} edges formed")
        else:
            print(f"  [{sid}] No data")

    create_camera()
    create_lights()

    blend_path = OUTPUT_DIR / "m8_growing_graphs.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"Saved: {blend_path}")
    print(f"Frames: {F['total']} ({T['total']}s)")

    if render:
        bpy.ops.render.render(animation=True)

if __name__ == "__main__":
    main()

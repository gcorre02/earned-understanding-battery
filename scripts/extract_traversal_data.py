"""Extract traversal data from all 9 M8 battery systems for Blender animation.

Generates a MEDIUM SBM graph, runs each system through 200 steps,
captures per-step node visits, computes 3D layout, exports to JSON.

Usage:
    .venv/bin/python scripts/extract_traversal_data.py
    .venv/bin/python scripts/extract_traversal_data.py --lightweight-only
    .venv/bin/python scripts/extract_traversal_data.py --steps 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import gc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import networkx as nx
import torch
from m8_battery.core.types import DomainConfig
from m8_battery.domains.sbm_generator import generate_domain
from m8_battery.domains.presets import MEDIUM

RESULTS_DIR = Path(__file__).parent.parent / "results" / "animation"

SYSTEM_META = {
    "1A": {"label": "Static Graph", "class": 1, "colour": "#FFFFFF", "heavy": False},
    "1B": {"label": "Rule Navigator", "class": 1, "colour": "#C0C0C0", "heavy": False},
    "1C": {"label": "Foxworthy A", "class": 1, "colour": "#B39DDB", "heavy": False},
    "2A": {"label": "Frozen TinyLlama", "class": 2, "colour": "#00BCD4", "heavy": True},
    "2B": {"label": "Frozen GAT", "class": 2, "colour": "#1565C0", "heavy": False},
    "2C": {"label": "Foxworthy C", "class": 2, "colour": "#FF9800", "heavy": False},
    "3A": {"label": "DQN Agent", "class": 3, "colour": "#F44336", "heavy": True},
    "3B": {"label": "Curiosity Agent", "class": 3, "colour": "#4CAF50", "heavy": True},
    "3C": {"label": "Foxworthy F", "class": 3, "colour": "#FFD700", "heavy": True},
}

SEED = 42
N_FEATURES = MEDIUM.n_node_features
N_COMMUNITIES = MEDIUM.n_communities

def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def make_system(system_id: str, graph: nx.DiGraph):
    """Instantiate and prepare a system for stepping."""
    device = _get_device()

    if system_id == "1A":
        from m8_battery.systems.class1.wordnet_graph import WordNetGraph
        return WordNetGraph(graph=graph, seed=SEED)

    elif system_id == "1B":
        from m8_battery.systems.class1.rule_navigator import RuleBasedNavigator
        return RuleBasedNavigator(graph=graph, seed=SEED)

    elif system_id == "1C":
        from m8_battery.systems.class1.foxworthy_a import FoxworthyA
        s = FoxworthyA(n_features=N_FEATURES, seed=SEED)
        s.set_graph(graph)
        return s

    elif system_id == "2A":
        from m8_battery.systems.class2.frozen_llm import FrozenLLM
        s = FrozenLLM(seed=SEED, device=device)
        s.set_graph(graph)
        s.load_model()
        return s

    elif system_id == "2B":
        from m8_battery.systems.class2.frozen_gnn import FrozenGAT
        s = FrozenGAT(n_features=N_FEATURES, n_classes=N_COMMUNITIES, seed=SEED)
        s.train_on_domain(graph, epochs=50)
        return s

    elif system_id == "2C":
        from m8_battery.systems.class2.foxworthy_c import FoxworthyC
        s = FoxworthyC(n_features=N_FEATURES, seed=SEED)
        s.set_graph(graph)
        return s

    elif system_id == "3A":
        from m8_battery.systems.class3.dqn_agent import DQNAgent
        s = DQNAgent(seed=SEED)
        s.train_on_domain(graph)
        return s

    elif system_id == "3B":
        from m8_battery.systems.class3.curiosity_agent import CuriosityAgent
        s = CuriosityAgent(seed=SEED)
        s.train_on_domain(graph)
        return s

    elif system_id == "3C":
        from m8_battery.systems.class3.foxworthy_f import FoxworthyF
        s = FoxworthyF(seed=SEED, device=device, theta=0.0)
        s.set_graph(graph)
        s.load_model()
        s.train_on_domain(graph, n_warmup=50)
        return s

    raise ValueError(f"Unknown system: {system_id}")

def run_traversal(system, n_steps: int) -> list[int]:
    """Run a system for n_steps, returning list of visited node IDs."""
    system.reset()
    path = []
    for _ in range(n_steps):
        result = system.step(None)
        node = result.get("current_node")
        if node is not None:
            path.append(int(node))
        else:
            path.append(-1)
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lightweight-only", action="store_true",
                        help="Skip heavy model systems (2A, 3A, 3B, 3C)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== M8 Traversal Data Extraction ===")
    print(f"Steps: {args.steps}, Lightweight: {args.lightweight_only}")
    print()

    # Generate domain
    print("Generating MEDIUM domain (seed=42)...")
    G = generate_domain(MEDIUM)
    nodes = sorted(G.nodes())
    print(f"  {len(nodes)} nodes, {G.number_of_edges()} edges")

    # Compute 3D layout
    print("Computing 3D spring layout...")
    pos_3d = nx.spring_layout(G.to_undirected(), dim=3, seed=42, k=2.0, iterations=100)

    # Build graph data
    graph_data = {
        "nodes": [],
        "edges": [],
    }
    for n in nodes:
        feat = G.nodes[n].get("features", {})
        community = feat.get("community", 0)
        p = pos_3d[n]
        graph_data["nodes"].append({
            "id": int(n),
            "community": int(community),
            "position": [float(p[0]), float(p[1]), float(p[2])],
        })

    for u, v, d in G.edges(data=True):
        graph_data["edges"].append({
            "source": int(u),
            "target": int(v),
            "weight": float(d.get("weight", 1.0)),
        })

    # Run systems
    systems_data = {}
    run_order = ["1A", "1B", "1C", "2C", "2B", "2A", "3A", "3B", "3C"]

    for sid in run_order:
        meta = SYSTEM_META[sid]
        if args.lightweight_only and meta["heavy"]:
            print(f"[{sid}] Skipping (lightweight mode) — generating random walk")
            rng = np.random.default_rng(SEED + hash(sid) % 1000)
            path = []
            current = int(rng.choice(nodes))
            for _ in range(args.steps):
                path.append(current)
                neighbours = list(G.successors(current))
                if neighbours:
                    current = int(rng.choice(neighbours))
                else:
                    current = int(rng.choice(nodes))
            systems_data[sid] = {
                "label": meta["label"],
                "class": meta["class"],
                "colour": meta["colour"],
                "traversal": path,
                "synthetic": True,
            }
            continue

        print(f"[{sid}] {meta['label']} (Class {meta['class']})...")
        t0 = time.time()
        try:
            system = make_system(sid, G)
            path = run_traversal(system, args.steps)
            elapsed = time.time() - t0
            print(f"  Done: {elapsed:.1f}s, visited {len(set(path))} unique nodes")

            systems_data[sid] = {
                "label": meta["label"],
                "class": meta["class"],
                "colour": meta["colour"],
                "traversal": path,
                "synthetic": False,
            }

            # Free memory
            if hasattr(system, "unload_model"):
                system.unload_model()
            del system
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")
            systems_data[sid] = {
                "label": meta["label"],
                "class": meta["class"],
                "colour": meta["colour"],
                "traversal": [],
                "error": str(e),
            }

    # Export
    output = {
        "graph": graph_data,
        "systems": systems_data,
        "metadata": {
            "n_nodes": len(nodes),
            "n_communities": N_COMMUNITIES,
            "n_steps": args.steps,
            "seed": SEED,
            "preset": "MEDIUM",
        },
    }

    out_path = RESULTS_DIR / "traversal_data.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nExported: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()

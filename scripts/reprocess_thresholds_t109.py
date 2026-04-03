#!/usr/bin/env python3
"""T1-09: Reprocess thresholds from final recalibration data.

Applies methodology (three-zone classification) to all instruments
including earned ratio thresholds. Validates conjunction for all
systems including Hebbian (internal) and STDP (4A-anchor).

Produces:
- Per-instrument threshold analysis
- Partial pass matrix
- Conjunction validation
- Sensitivity analysis
- Failure mode taxonomy per system
"""

import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results" / "recalibration_m5"

CLASS_MAP = {
    "1A": 1, "1B": 1, "1C": 1,
    "2A": 2, "2B": 2, "2C": 2,
    "3A": 3, "3B": 3, "3C": 3,
    "3D": 3, "3E": 3,
    "HEB": "anchor", "STDP": "anchor",
}

INSTRUMENTS = [
    "developmental_trajectory",
    "integration",
    "generativity",
    "transfer",
    "self_engagement",
]

def load_results():
    """Load all recalibration JSON files."""
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results

def analyse_instrument(runs, inst_name):
    """Analyse one instrument across all runs."""
    print(f"\n{'='*60}")
    print(f"  {inst_name.upper()}")
    print(f"{'='*60}")

    # Collect per-system data
    by_system = {}
    for r in runs:
        sid = r["system_id"]
        inst = r["instruments"].get(inst_name, {})
        es = inst.get("effect_size") or 0.0
        passed = inst.get("passed")
        notes = inst.get("notes", "")
        by_system.setdefault(sid, []).append({
            "es": es, "passed": passed, "seed": r["seed"], "notes": notes,
        })

    # Per-system summary
    print(f"\n  Per-system results:")
    for sid in ["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C", "3D", "3E", "HEB", "STDP"]:
        if sid not in by_system:
            continue
        vals = by_system[sid]
        es_vals = [v["es"] for v in vals]
        pass_vals = [v["passed"] for v in vals]
        cls = CLASS_MAP.get(sid, "?")
        mean_es = np.mean(es_vals)
        print(f"    {sid:4s} (class {cls}): ES={[round(v,4) for v in es_vals]} "
              f"mean={mean_es:.4f} passed={pass_vals}")

    # Class-level distributions (excluding anchors)
    print(f"\n  Class distributions:")
    for c in [1, 2, 3]:
        vals = []
        for sid, data in by_system.items():
            if CLASS_MAP.get(sid) == c:
                vals.extend([d["es"] for d in data])
        if vals:
            print(f"    Class {c}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} "
                  f"[{min(vals):.4f}, {max(vals):.4f}]")

    # Anchor results
    for sid in ["HEB", "STDP"]:
        if sid in by_system:
            vals = by_system[sid]
            es_vals = [v["es"] for v in vals]
            pass_vals = [v["passed"] for v in vals]
            print(f"    {sid}: ES={[round(v,4) for v in es_vals]} passed={pass_vals}")

def conjunction_validation(runs):
    """Validate conjunction: no Class 1-3 system passes all 5."""
    print(f"\n{'='*60}")
    print(f"  CONJUNCTION VALIDATION")
    print(f"{'='*60}")

    # Group by system+seed
    by_key = {}
    for r in runs:
        key = f"{r['system_id']}_s{r['seed']}"
        by_key[key] = r

    print(f"\n  Partial pass matrix:")
    print(f"  {'System':8s} {'Class':6s} {'Traj':5s} {'Integ':6s} {'Gen':5s} {'Trans':6s} {'Self':5s} {'Total':6s}")
    print(f"  {'-'*50}")

    any_class13_pass = False
    for sid in ["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C", "3D", "3E", "HEB", "STDP"]:
        for seed in [42, 123, 456]:
            key = f"{sid}_s{seed}"
            if key not in by_key:
                continue
            r = by_key[key]
            inst = r["instruments"]
            passes = {}
            for iname in INSTRUMENTS:
                passes[iname] = inst.get(iname, {}).get("passed") is True

            n_pass = sum(passes.values())
            cls = CLASS_MAP.get(sid, "?")

            marks = []
            for iname in INSTRUMENTS:
                marks.append("PASS" if passes[iname] else "    ")

            conj = "PASS" if n_pass == 5 else f"FAIL"
            print(f"  {sid:4s} s{seed} {str(cls):6s} {marks[0]:5s} {marks[1]:6s} "
                  f"{marks[2]:5s} {marks[3]:6s} {marks[4]:5s} {n_pass}/5 {conj}")

            if n_pass == 5 and cls in [1, 2, 3]:
                any_class13_pass = True

    print(f"\n  Conjunction validation: ", end="")
    if any_class13_pass:
        print("FAIL — a Class 1-3 system passes all 5!")
    else:
        print("PASS — no Class 1-3 system passes all 5")

    # Check anchor requirements
    for sid in ["STDP"]:
        se_passes = []
        conj_passes = []
        for seed in [42, 123, 456]:
            key = f"{sid}_s{seed}"
            if key not in by_key:
                continue
            r = by_key[key]
            inst = r["instruments"]
            se_pass = inst.get("self_engagement", {}).get("passed") is True
            all_pass = all(inst.get(i, {}).get("passed") is True for i in INSTRUMENTS)
            se_passes.append(se_pass)
            conj_passes.append(all_pass)

        print(f"\n  STDP anchor check:")
        print(f"    Self-engagement passes: {se_passes}")
        print(f"    Conjunction passes: {conj_passes}")
        if any(se_passes) and not any(conj_passes):
            print(f"    REQUIREMENT MET: passes self-engagement, fails conjunction")
        elif not any(se_passes):
            print(f"    WARNING: STDP does not pass self-engagement on any seed")
        elif any(conj_passes):
            print(f"    WARNING: STDP passes full conjunction on some seeds")

def failure_taxonomy(runs):
    """Report failure modes from taxonomy."""
    print(f"\n{'='*60}")
    print(f"  FAILURE MODE TAXONOMY")
    print(f"{'='*60}")

    for iname in INSTRUMENTS:
        modes = {}
        for r in runs:
            sid = r["system_id"]
            inst = r["instruments"].get(iname, {})
            notes = inst.get("notes", "")
            # Extract failure mode from notes (heuristic — proper field not in JSON yet)
            passed = inst.get("passed")
            if passed is True:
                mode = "earned"
            elif "Precondition" in notes:
                mode = "precondition-fail"
            elif "not earned" in notes or "" in notes:
                mode = "architectural"
            elif "No response" in notes or "constant" in notes or "unchanged" in notes:
                mode = "absent"
            elif "Ambiguous" in notes:
                mode = "noise"
            elif "Linear" in notes or "uniform" in notes:
                mode = "modular"
            else:
                mode = "absent"

            modes.setdefault(mode, []).append(sid)

        print(f"\n  {iname}:")
        for mode, systems in sorted(modes.items()):
            unique = sorted(set(systems))
            print(f"    {mode}: {unique}")

def main():
    print("T1-09: Threshold Reprocessing + Conjunction Validation")
    print(f"Data directory: {RESULTS_DIR}")
    print()

    runs = load_results()
    print(f"Loaded {len(runs)} calibration results")

    # Check which systems are present
    systems_found = sorted(set(r["system_id"] for r in runs))
    print(f"Systems: {systems_found}")

    # Per-instrument analysis
    for inst_name in INSTRUMENTS:
        analyse_instrument(runs, inst_name)

    # Conjunction validation
    conjunction_validation(runs)

    # Failure taxonomy
    failure_taxonomy(runs)

    print(f"\n{'='*60}")
    print(f"  RECALIBRATION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

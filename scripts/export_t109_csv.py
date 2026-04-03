#!/usr/bin/env python
"""Export T109 calibration data from recalibration JSON results to CSV.

Reads all 39 JSON files from results/recalibration_m5/ and produces
results/t109_calibration_data.csv plus summary info about SBM parameters
and generativity measurement details.
"""

import csv
import json
import glob
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "recalibration_m5"
OUTPUT_CSV = ROOT / "results" / "t109_calibration_data.csv"


def extract_earned_ratio(notes: str) -> str:
    """Extract earned_ratio from notes string if present."""
    m = re.search(r"earned_ratio=([\d]+\.[\d]+)", notes)
    return m.group(1) if m else ""


def extract_failure_mode(classifications: dict, instrument: str) -> str:
    """Get failure mode from classifications dict."""
    return classifications.get(instrument, "")


def main():
    json_files = sorted(glob.glob(str(RESULTS_DIR / "*.json")))
    print(f"Found {len(json_files)} JSON files\n")

    rows = []
    for fpath in json_files:
        with open(fpath) as f:
            data = json.load(f)

        system_id = data["system_id"]
        seed = data["seed"]
        classifications = data.get("classifications", {})

        for inst_name, inst_data in data["instruments"].items():
            rows.append({
                "system_id": system_id,
                "seed": seed,
                "instrument": inst_name,
                "passed": inst_data["passed"],
                "effect_size": inst_data.get("effect_size", ""),
                "earned_ratio": extract_earned_ratio(inst_data.get("notes", "")),
                "failure_mode": extract_failure_mode(classifications, inst_name),
                "notes": inst_data.get("notes", ""),
            })

    # Write CSV
    fieldnames = ["system_id", "seed", "instrument", "passed", "effect_size",
                  "earned_ratio", "failure_mode", "notes"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written: {OUTPUT_CSV}")
    print(f"  {len(rows)} rows ({len(json_files)} files x {len(rows) // len(json_files)} instruments)\n")

    # --- SBM Parameters (MEDIUM preset) ---
    print("=" * 60)
    print("SBM Parameters (MEDIUM preset from presets.py)")
    print("=" * 60)
    print(f"  n_nodes:          150")
    print(f"  n_communities:    6")
    print(f"  p_within:         0.3")
    print(f"  p_between:        0.02")
    print(f"  n_node_features:  8  (DomainConfig default)")
    print(f"  seed:             42")
    print()

    # --- Generativity instrument details ---
    print("=" * 60)
    print("Generativity Instrument Measurement Details")
    print("=" * 60)
    print()
    print("Structural generativity (instruments/generativity.py):")
    print("  Metric: system.get_structure_metric() — delta between")
    print("          metric_before and metric_after novel-domain exposure")
    print("  Decision: passed if metric changed (|delta| > 1e-6) AND")
    print("            structure retained (retention > 0.5)")
    print("  Effect size: |relative_delta| = |delta / reference_metric|")
    print()
    print("Behavioural generativity (analysis/behavioural_generativity.py):")
    print("  JSD: Jensen-Shannon divergence over node visit distributions")
    print("  Log base: natural log (np.log, base e)")
    print("  Method: JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q)")
    print("          with epsilon=1e-10 smoothing to avoid log(0)")
    print()
    print("Domain-B inputs used (from run_m5_recalibration.py):")
    print("  Standard systems: 50 (nodes_b[:50])")
    print("  STDP:             160")
    print("  LLM systems:      20")


if __name__ == "__main__":
    main()

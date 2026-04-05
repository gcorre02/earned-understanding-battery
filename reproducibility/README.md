# Reproducibility Bundle — Generativity Instrument v3

## How to Reproduce

1. Clone the repository at commit `388a90f`
2. Install dependencies: `pip install -r reproducibility/requirements-v3.txt`
3. Run: `python scripts/reproduce_v3_tables.py`

The script reads the raw artefacts and recomputes all headline tables from the v3 data package.

## Raw Artefacts

| File | Contents |
|------|----------|
| `results/harmonised_results.json` | Per-system per-seed per-domain results (63 runs) + null summaries + bootstrap CIs + ROC AUC |
| `results/null_distribution_raw_samples.csv` | 500 raw null-pair values (5 types x 2 domains x 50 pairs) |

### CSV Schema
```
system_type, seed_a, seed_b, marginal_jsd, transition_jsd, structural_consistency, domain_variant
```

### JSON Schema (per result entry)
```json
{
  "system_name": "string",
  "seed": "int",
  "domain_variant": "string (B0/B1/B2)",
  "marginal_jsd": "float",
  "transition_jsd": "float",
  "structural_consistency": "float",
  "marginal_coherence": "float",
  "transition_coherence": "float",
  "trained_self_transition": "float",
  "fresh_self_transition": "float",
  "trained_entropy": "float",
  "fresh_entropy": "float",
  "trained_visited": "int",
  "fresh_visited": "int",
  "edge_jaccard": "float",
  "signal_type": "string",
  "commit": "string"
}
```

## Environment

- Python 3.12.8
- See `requirements-v3.txt` for exact package versions
- Platform: macOS (Darwin), Apple Silicon (M5 Max)

## Paper 1

DOI: https://doi.org/10.5281/zenodo.19178410 (v2 — current, the version this battery is built against)

v1 (original, superseded): https://doi.org/10.5281/zenodo.19011223

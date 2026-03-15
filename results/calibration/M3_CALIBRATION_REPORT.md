# M3 Max Calibration Report — Systems 2A + 3C

**Date:** 2026-03-15
**Machine:** Mac M3 Max (36GB unified memory, MPS backend)
**Branch:** m3-calibration
**Scale:** MEDIUM (150 nodes, 5 communities)
**Seeds:** 42, 123, 456

---

## Code Changes

### MPS Device Support (SD-012)

The calibration script and two system adapters were modified to use Apple Silicon MPS
instead of CPU-only inference:

1. **`scripts/run_m3_calibration.py`** — Auto-detect MPS/CUDA/CPU via `_get_device()`.
   Previously hardcoded `device="cpu"`.

2. **`src/m8_battery/systems/class2/frozen_llm.py:258-259`** — `torch.randn` for
   perturbation noise generated on CPU then `.to(device)`. MPS does not support
   `torch.Generator(device="mps")`.

3. **`src/m8_battery/systems/class3/foxworthy_f.py:508-509`** — Same fix as above.

**Why:** CPU inference on TinyLlama 1.1B was estimated at ~90 min per system.
MPS reduced total run from ~2 hours to ~21 minutes (1244s).

**Determinism:** `torch.Generator` seeding is preserved (CPU generator with fixed seed).
Noise is identical across runs; only the device transfer is added.

---

## Results Summary

| System | Class | Seed | Overall | Battery Time | Provenance |
|--------|-------|------|---------|-------------|------------|
| 2A (Frozen TinyLlama 1.1B) | 2 | 42  | FAIL | 360.1s | FAIL |
| 2A (Frozen TinyLlama 1.1B) | 2 | 123 | FAIL | 293.9s | FAIL |
| 2A (Frozen TinyLlama 1.1B) | 2 | 456 | FAIL | 275.8s | FAIL |
| 3C (Foxworthy Variant F)   | 3 | 42  | FAIL | 101.2s | FAIL |
| 3C (Foxworthy Variant F)   | 3 | 123 | FAIL |  92.3s | FAIL |
| 3C (Foxworthy Variant F)   | 3 | 456 | FAIL |  68.8s | FAIL |

**Total runtime:** 1244s (~21 min)

---

## Per-Instrument Breakdown

### 2A — Frozen TinyLlama 1.1B (Class 2)

Completely flat across all three seeds — consistent with a frozen model that does not
learn or reorganise.

| Instrument | Passed | Effect Size | Notes |
|-----------|--------|-------------|-------|
| Developmental Trajectory | FAIL | 0.0 | Structure metric constant — no trajectory |
| Integration | FAIL | 0.100 | Linear/uniform degradation (Gini=0.10, CV=0.18) |
| Generativity | FAIL | 0.0 | No response to novel domain (delta=0.0) |
| Transfer | FAIL | 0.0 | Trained and naive produce identical metric |
| Self-Engagement | FAIL | 0.0 | Engagement pattern disrupted (cosine_sim=0.0) |

**Reset discrimination:** No discrimination at all. Pre/post/rerun metrics identical
(0.831). Persistence=1.0, regrowth=0.0. Frozen weights are completely inert.

### 3C — Foxworthy Variant F (Class 3)

More interesting — shows partial capability, consistent with an adaptive system that
lacks genuine emergent organisation.

| Instrument | Seed 42 | Seed 123 | Seed 456 |
|-----------|---------|----------|----------|
| Developmental Trajectory | FAIL (slope=-0.18, R²=0.21) | AMBIGUOUS (slope=-0.18, R²=0.48) | **PASS** (slope=0.42, R²=0.74) |
| Integration | FAIL (Gini=0.05) | FAIL (Gini=0.09) | FAIL (Gini=0.20) |
| Generativity | **PASS** (delta=2.25) | **PASS** (delta=11.64) | **PASS** (delta=4.86) |
| Transfer | AMBIGUOUS (adv=-0.38) | AMBIGUOUS (adv=-0.12) | AMBIGUOUS (adv=-0.31) |
| Self-Engagement | FAIL (cosine=0.0) | FAIL (cosine=0.0) | FAIL (cosine=0.0) |

**Key observations:**
- **Generativity passes consistently** (3/3 seeds) — Foxworthy F responds to novel
  domains, which is expected for a LoRA-tuned model with curiosity-driven training.
- **Transfer is ambiguous** (3/3 seeds) — negative advantage means naive system
  sometimes outperforms trained. Not genuine structural transfer.
- **Integration fails** — low Gini scores indicate uniform/linear degradation rather
  than non-linear interdependence between regions.
- **Self-engagement fails uniformly** — no recovery pattern after perturbation.
- **Developmental trajectory is seed-sensitive** — one pass, one ambiguous, one fail.
  The positive slope in seed 456 (R²=0.74) warrants investigation on Razer with more
  seeds.

**Reset discrimination:** Non-trivial dynamics. Persistence ranges 0.90-1.40,
regrowth 0.03-0.10. The system does reorganise after reset, but not in a way that
demonstrates earned structure.

---

## Findings

- **F-M3-01:** MPS gives ~5x speedup over CPU on M3 Max for TinyLlama inference.
  No numerical divergence observed in pass/fail outcomes (deterministic seeding preserved).
- **F-M3-02:** 3C generativity passes consistently — this is the first instrument where
  a Class 3 system reliably passes. The conjunction gate (all instruments must pass) still
  correctly rejects the system overall.
- **F-M3-03:** 3C developmental trajectory is seed-sensitive (1 pass, 1 ambiguous, 1 fail).
  May need more seeds or adjusted thresholds for stable discrimination.
- **F-M3-04:** 3C transfer instrument returns ambiguous for all seeds — the transfer
  instrument may need threshold tuning to produce clean pass/fail for Class 3 systems.

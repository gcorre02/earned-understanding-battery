## Transfer Instrument Protocol

### 1. Purpose

The transfer instrument tests whether structural organisation learned in one domain accelerates acquisition in an isomorphic but statistically altered domain. A system that has genuinely earned relational structure on domain A should, when exposed to domain A' (same relational invariants, destroyed surface statistics), converge faster than a naive system encountering A' for the first time. This is the empirical operationalisation of structural transfer as described in Paper 1 section 5 and section 3.5, which defines transfer as traceable to shared relational invariants rather than statistical familiarity.

### 2. Construct Definition

**Transfer** (Paper 1 section 5):

> "Transfer must be traceable to structural correspondence (shared relational invariants), not statistical familiarity."

Domain A' preserves the relational structure of domain A while destroying surface statistics. The transfer instrument measures whether a trained system's learned dynamics accelerate structural convergence on A' relative to a naive baseline. If acceleration is present, and the naive baseline does not show comparable convergence, the transfer is earned -- it reflects internalised relational invariants, not distributional overlap.

### 3. Metric

The primary metric is the area under curve (AUC) of the structure metric trajectory, comparing a trained system on A' against a naive system on A'.

**Trajectory collection.** Both trained and naive systems are stepped through domain A' inputs. The structure metric is recorded at intervals of `measurement_interval` steps (default 5), beginning with the initial structure metric before any A' input. This produces a trajectory of structure metric values over time.

**AUC computation.** The area under each trajectory is computed via the trapezoidal rule (`np.trapezoid`).

**Transfer advantage:**

```
transfer_advantage = (trained_AUC - naive_AUC) / |naive_AUC|
```

When `|naive_AUC| < 1e-10`, the advantage is set to 0.0 (no meaningful baseline to compare against).

**Effect size.** Cohen's d is computed from the final 3 trajectory points of each system:

```
pooled_std = sqrt((std(trained[-3:])^2 + std(naive[-3:])^2) / 2)
effect_size = |trained_final - naive_final| / pooled_std
```

When fewer than 3 trajectory points exist or pooled standard deviation is below 1e-10, effect size is not computed.

### 4. Threshold

Both conditions must hold:

| Condition | Criterion |
|-----------|-----------|
| `has_advantage` | `transfer_advantage > 0.1` |
| `metrics_differ` | `|trained_final - naive_final| > 1e-6` |

The advantage threshold of 0.1 (10% improvement over naive) prevents noise-level differences from registering as transfer. The metrics-differ check prevents degenerate cases where both systems produce identical final metrics but differ in trajectory shape.

### 5. Earned Ratio

```
earned_ratio = trained_AUC / |naive_AUC|
```

DN-22 requires `earned_ratio > 1.0`. This ensures the trained system's cumulative structural signal exceeds the naive system's -- not merely that the endpoint differs, but that the trained system maintained a structural advantage throughout exposure to A'.

Edge cases:
- When `|naive_AUC| < 1e-10` but `|trained_AUC| > 1e-10`: ratio is capped at `min(trained_AUC / 1e-10, 1e6)`.
- When both AUCs are near zero: ratio defaults to 1.0 (no earned advantage).
- Global cap: `1e6` (ratio-based metric; higher cap acceptable compared to geometric-mean instruments).

### 6. Pass Condition

```
passed = has_advantage AND metrics_differ AND passes_earned
```

All three conditions are required. The conjunction ensures that (a) the trained system shows meaningful acceleration, (b) the systems arrive at different structural states, and (c) the acceleration is earned relative to the naive baseline.

### 7. Failure Modes

| Mode | Condition | Interpretation |
|------|-----------|---------------|
| `earned` | All three conditions met | PASS. Trained system shows earned structural transfer to A'. |
| `statistical` | `has_advantage` and `metrics_differ` but not `passes_earned` | Transfer present but not earned. Naive system shows similar AUC, suggesting distributional overlap rather than structural correspondence. |
| `absent` | `metrics_differ` is false | No transfer. Trained and naive systems produce identical final metrics on A'. The system's learned structure does not influence behaviour on the altered domain. |
| `shortcut` | Remaining cases (advantage exists but ambiguous) | Ambiguous. Some advantage is present but does not meet the full threshold. Returns `passed=None` (indeterminate). |

### 8. Positive Control Evidence

**3E (Active Inference, Friston).** The architectural ground truth for transfer: the Dirichlet transition model learns relational structure over domain A that generalises to A' because the learned transitions encode community-to-community relationships, not node-level statistics. When surface statistics are destroyed, the relational model still applies.

Calibration results (medium scale, 3 seeds):

| Seed | Passed | Advantage | Earned Ratio | Trained Final | Naive Final |
|------|--------|-----------|--------------|---------------|-------------|
| 42 | True | 9.18 | 10.18 | 48.41 | 7.07 |
| 123 | True | 9.02 | 10.02 | 47.73 | 7.07 |
| 456 | True | 9.02 | 10.02 | 47.60 | 7.07 |

3E passes 3/3 seeds with earned ratios of approximately 10x -- the trained system accumulates ten times the structural signal of a naive system on A'. The naive final metric (7.07) is consistent across seeds (expected: the naive system has no domain A experience to transfer).

**Class 1 and Class 2 systems correctly fail.** Static graphs (1A, 1B) show zero advantage (no learning mechanism). Frozen networks (2A, 2B) show no transfer (no plasticity to develop transferable structure). This is the expected negative control behaviour: systems without a learning mechanism that operates over relational structure cannot exhibit relational transfer.

### 9. Calibration Results

Results from the Phase A recalibration run (medium scale, seed 42 shown; multi-seed summary in pass rate column).

| System | Class | Seed 42 Result | Earned Ratio | Pass Rate (3 seeds) |
|--------|-------|----------------|--------------|---------------------|
| 1A (WordNet Static) | 1 | FAIL (absent) | -- | 0/3 |
| 1B (Rule-Based) | 1 | FAIL (absent) | -- | 0/3 |
| 1C (Foxworthy A) | 1 | None (shortcut) | -- | 0/3 |
| 2A (Frozen TinyLlama) | 2 | FAIL (absent) | -- | 0/3 |
| 2B (Frozen GAT) | 2 | None (shortcut) | -- | 0/3 |
| 2C (Foxworthy C) | 2 | PASS | 1.26 | 2/3 |
| 3A (DQN/MaskablePPO) | 3 | None (shortcut) | -- | 0/3 |
| 3B (Curiosity/RND) | 3 | None (shortcut) | -- | 1/3 |
| 3C (Foxworthy F) | 3 | None (shortcut) | -- | 0/3 |
| 3D (Empowerment) | 3 | None (shortcut) | -- | 0/3 |
| 3E (Active Inference) | 3 | PASS | 10.18 | 3/3 |
| HEB (Hebbian Walker) | -- | PASS | 84.40 | 3/3 |
| STDP (Brian2 STDP) | -- | PASS | 1.27 | 3/3 |

Key observations:
- **3E** is the designated positive control with a 10x earned ratio, stable across seeds.
- **HEB** shows the highest earned ratio (~83x) but this reflects Hebbian weight accumulation rather than relational model transfer. HEB's transfer is mechanistically different from 3E's (weight magnitude vs. relational structure).
- **STDP** passes with modest earned ratios (~1.3x), consistent with spike-timing-dependent plasticity creating transferable edge-weight structure.
- **2C** passes 2/3 seeds with marginal earned ratios (~1.2x). This is a borderline case worth monitoring in Phase C.
- All Class 1 systems fail or return indeterminate. All pure Class 2 systems (2A, 2B) fail.

### 10. Known Limitations

1. **Single true positive for relational transfer.** 3E is the only system whose transfer is architecturally grounded in relational model generalisation. HEB and STDP pass via weight-based mechanisms. AUC = 1.0 for per-instrument ROC reflects strict separation between the positive and negative pools, not statistical power. Confidence in the instrument's sensitivity depends on expanding the positive control set in Phase C.

2. **Domain A' construction quality.** A' preserves relational structure while destroying surface statistics. Spectral similarity between A and A' is verified by M-04 calibration (spectral basin signatures, k=10, full-basin, symmetrised Laplacian). However, the verification is internal to the battery -- it has not been independently audited. If A' inadvertently preserves surface statistics that a system can exploit, the instrument would produce false positives.

3. **Trajectory sampling.** `measurement_interval=5` means the structure metric trajectory is sampled every 5 steps, not recorded continuously. Rapid structural changes between measurement points are invisible. The AUC computed from sampled trajectories is an approximation of the true area under the continuous trajectory.

4. **Earned ratio cap.** The `1e6` cap is necessary to prevent division-by-near-zero from producing astronomically large ratios, but it means that systems with truly negligible naive AUC will all cluster at the cap regardless of their actual trained AUC magnitude.

5. **No trajectory precondition.** Unlike self-engagement, the transfer instrument does not require the system to have passed the developmental trajectory instrument. A system can show transfer without showing earned trajectory development (e.g., 3E passes transfer but fails trajectory in the recalibration run because its trajectory earned ratio is 0.18). This is by design -- transfer tests a different property -- but it means transfer results are interpretable only within the conjunction, not in isolation.

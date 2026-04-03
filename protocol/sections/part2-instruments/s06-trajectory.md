## Developmental Trajectory Instrument Protocol

**Instrument:** Developmental Trajectory (structure from dynamics)

---

### 1. Purpose

The developmental trajectory instrument tests whether structural organisation develops through operation -- compression from diffuse to stable over time. A Class 4 candidate should show measurable structural compression during training that is statistically distinguishable from both random fluctuation and architectural pre-specification. This is the empirical operationalisation of emergence as described in Paper 1 section 3.1: structure arises from dynamics, not from prior specification.

---

### 2. Construct Definition

**Emergence** (Paper 1 section 3.1):

> "Structure arises from dynamics, not prior specification."

The instrument measures whether the system's internal structure changes measurably and monotonically over operational time. A system whose structure is fixed at initialisation (Class 1) will show a flat trajectory. A system whose structure changes but in a way indistinguishable from a fresh instance running the same inputs has architectural trajectory (topology-driven), not earned trajectory. Only systems whose structural development exceeds what the architecture alone provides pass the earned-ratio gate.

---

### 3. Metric

The primary metrics are computed from the structure metric time series collected at each training step:

1. **Linear regression slope** of `structure_metric` over training steps. Quantifies the rate of structural change.
2. **R-squared** of the linear regression. Quantifies how well a monotonic trend explains the observed trajectory.
3. **Monotonicity fraction.** Fraction of consecutive step-pairs where the structure metric moves in the dominant direction. Computed as `max(n_positive, n_negative) / len(diffs)` where `diffs = np.diff(trajectory)`.

**Supplementary diagnostic:** Lempel-Ziv compression ratio (via bz2) of the structure metric time series, split into early and late halves. The compression trend (late_ratio minus early_ratio) indicates whether the system develops more compressible (structured) behaviour over time. A negative compression trend means the late trajectory is more compressible than the early trajectory. Literature basis: TERL (2025) and brain entropy trajectories (PMC 2022).

---

### 4. Threshold

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| slope | > 1e-6 | Structure metric must change measurably (not constant within floating-point noise) |
| R-squared | > 0.3 | Trend must explain at least 30% of variance (not random walk) |
| monotonicity | > 0.6 | At least 60% of consecutive pairs must move in the dominant direction |

All three thresholds must be met simultaneously for `has_trend` and `has_monotonicity` to be true. A system can have a strong slope but low monotonicity (noisy oscillation) or high monotonicity but low R-squared (small consistent changes swamped by variance) -- both are classified as ambiguous.

---

### 5. Earned Ratio

The earned ratio compares the trained system's metric range against a fresh (untrained) control system's metric range on the same inputs:

```
earned_ratio = metric_range(trained) / metric_range(fresh)
```

where `metric_range = max(trajectory) - min(trajectory)`.

** requirement:** earned_ratio must be > 1.0. If a fresh system shows the same trajectory magnitude, the structural development is architectural (topology-driven), not earned through interaction.

** observation:** Topology-driven Hebbian walkers (e.g., HEB) correctly show earned_ratio >> 1.0 when tested through the battery (trained on domain A, fresh never exposed). However, when tested via direct instrument invocation with identical input sequences, systems whose learning is purely topology-driven show earned_ratio near 1.0 because the fresh system develops the same trajectory from the same topology. This is correct behaviour: the earned ratio correctly identifies that the trajectory is not path-dependent. PC-SE and PC-INT are tested via direct invocation precisely because the trajectory precondition correctly gates them.

---

### 6. Pass Condition

```
trajectory = has_trend AND has_monotonicity AND passes_earned
```

where:
- `has_trend` = `abs(slope) > 1e-6 AND r_squared > 0.3`
- `has_monotonicity` = `monotonicity > 0.6`
- `passes_earned` = `earned_ratio is None OR earned_ratio > 1.0`

All three conditions are required. If any fails, the system does not pass trajectory. The earned ratio condition is only enforced when a control factory is available (battery mode); in direct-invocation mode without a control, `passes_earned` defaults to true.

---

### 7. Failure Modes

| Mode | Condition | Interpretation |
|------|-----------|----------------|
| `earned` | has_trend AND has_monotonicity AND passes_earned | Genuine developmental trajectory. Structure develops through operation and exceeds architectural baseline. |
| `architectural` | has_trend AND has_monotonicity AND NOT passes_earned | Trajectory present but not earned. Fresh system shows comparable development -- structure is topology-driven. |
| `noise` | has_trend OR has_monotonicity (but not both) | Inconsistent signal. Some evidence of trend or monotonicity but not both. Classified as ambiguous (passed=None). |
| `absent` | NOT has_trend AND NOT has_monotonicity | No metric change. Structure is constant or fluctuates randomly. |

---

### 8. Positive Control Evidence

**STDP (Brian2 spiking network, 4A-anchor).** Three seeds, all PASS:

| Seed | Slope | R-squared | Monotonicity | Earned Ratio |
|------|-------|-----------|--------------|--------------|
| 42 | 0.000476 | 0.8734 | 0.67 | 2.18 |
| 123 | 0.000294 | 0.7218 | 0.63 | 3.43 |
| 456 | 0.000620 | 0.9523 | 0.70 | 2.00 |

STDP shows consistent developmental trajectory across all seeds with earned ratios well above 1.0. The spike-timing-dependent plasticity mechanism produces genuine path-dependent structural compression. Self-engagement also passes on all three seeds (resistance_ratio 43-478, recovery_ratio=inf), confirming that STDP's learned structure is self-maintaining.

**Foxworthy Variant F (3C) training phase.** During training, 3C shows adapter norm increase (+7 adapter norm), confirming that structural change occurs during training. However, the battery-measured trajectory is ambiguous (R-squared=0.37, monotonicity=0.50) because the structure metric does not capture adapter-space changes with sufficient resolution.

** validation.** Trajectory correctly detects path-dependent development (STDP, HEB in battery mode) and correctly identifies non-path-dependent development when systems are tested via direct invocation with identical inputs.

---

### 9. Calibration Results

Recalibration data from `results/recalibration_m5/`, medium scale (150 nodes), three seeds per system.

| System | Class | Seed 42 | Seed 123 | Seed 456 | Consensus |
|--------|-------|---------|----------|----------|-----------|
| 1A -- WordNet Static Graph | 1 | FAIL | FAIL | FAIL | FAIL |
| 1B -- Rule-Based Navigator | 1 | FAIL | FAIL | FAIL | FAIL |
| 1C -- Foxworthy Variant A | 1 | FAIL | FAIL | FAIL | FAIL |
| 2A -- Frozen TinyLlama 1.1B | 2 | FAIL | FAIL | FAIL | FAIL |
| 2B -- Frozen GAT | 2 | FAIL | FAIL | FAIL | FAIL |
| 2C -- Foxworthy Variant C | 2 | FAIL | FAIL | AMB | FAIL |
| 3A -- DQN (MaskablePPO) | 3 | FAIL | FAIL | FAIL | FAIL |
| 3B -- Curiosity (RND) | 3 | FAIL | FAIL | FAIL | FAIL |
| 3C -- Foxworthy Variant F | 3 | AMB | FAIL | FAIL | FAIL |
| 3D -- Empowerment (Klyubin) | 3 | FAIL | FAIL | FAIL | FAIL |
| 3E -- Active Inference (Friston) | 3 | FAIL | FAIL | FAIL | FAIL |
| HEB -- Hebbian Walker | 3 | PASS | PASS | PASS | PASS |
| STDP -- Brian2 STDP (4A-anchor) | 3 | PASS | PASS | PASS | PASS |

**Discrimination:** 2/13 systems pass (HEB, STDP). All Class 1 and Class 2 systems fail. All Class 3 systems except HEB and STDP fail. The two passing systems are the only ones with genuine plasticity mechanisms that operate on the structure metric. 3E (Active Inference) shows trajectory but fails the earned-ratio gate (earned_ratio=0.18 -- fresh shows stronger development). 3C shows ambiguous trajectory on one seed (R-squared just above 0.3, monotonicity at 0.5).

---

### 10. Known Limitations

1. ** observation: topology-driven walkers.** Topology-driven Hebbian walkers show earned_ratio near 1.0 when tested via direct instrument invocation with identical input sequences. This is correct behaviour -- the trajectory is not path-dependent -- not an instrument flaw. The battery-mode earned ratio is higher because trained vs fresh systems receive different histories (trained on domain A, fresh never exposed).

2. **Input count.** The default 50 `domain_a_inputs` may be insufficient for slow learners. Systems that require hundreds or thousands of training steps to develop structure may show flat trajectories within the 50-step measurement window. Mitigation: system-specific training protocols can provide longer input sequences; the instrument accepts arbitrary-length input lists.

3. **Structure metric resolution.** The trajectory instrument depends entirely on `get_structure_metric()` as a scalar summary of internal structure. Systems whose structural development occurs in dimensions not captured by this scalar (e.g., 3C's adapter norm changes) will show absent or ambiguous trajectories even if genuine development is occurring. This is a known limitation of scalar summarisation, not of the trajectory analysis itself.

4. **Linear regression assumption.** The analysis fits a linear model to what may be a nonlinear trajectory (e.g., sigmoid, logarithmic). A system with rapid early development followed by a plateau will have a lower R-squared than one with constant-rate development, even if the total structural change is greater. The monotonicity fraction partially compensates for this.

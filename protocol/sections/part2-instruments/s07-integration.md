## Integration Instrument Protocol

**Instrument:** Integration (non-decomposable structure)

---

### 1. Purpose

The integration instrument tests whether earned structure is non-decomposable -- the system is more than the sum of its parts. Removing a component should reorganise the whole, not merely leave a gap. This is the empirical operationalisation of integration as described in Paper 1 section 3.3: a system with genuine integration exhibits non-linear degradation under ablation, where the loss of any significant component forces the remaining structure to reorganise rather than continuing to function with a piece missing.

---

### 2. Construct Definition

**Integration** (Paper 1 section 3.3):

> "Non-decomposable. Removing a component reorganises the whole, not just leaves a gap."

The distinction is between a house of cards and an ecosystem. A house of cards collapses when any card is removed (fragile, not integrated). An ecosystem reorganises into a new stable regime when a species is removed (integrated). A modular system continues to function with a piece missing (not integrated). The instrument measures which pattern the system exhibits by systematically ablating regions and measuring the distribution of degradation magnitudes across ablation targets.

---

### 3. Metric

The integration instrument computes the following metrics from per-region ablation analysis:

1. **Gini coefficient of degradation magnitudes.** For each region, the system is ablated (region removed) and the structure metric is re-measured. The Gini coefficient of the absolute degradation values across all regions measures inequality: high Gini means some regions matter disproportionately more than others.

2. **Coefficient of variation (CV) of region importance.** Standard deviation of degradation magnitudes divided by their mean. CV > 0.5 indicates non-uniform degradation.

3. **Post-ablation reorganisation stability.** After ablating the highest-degradation region, the system is run for M additional steps. Engagement distributions are sampled at the midpoint and endpoint. Cosine similarity between these two distributions measures whether the ablated system settles into a stable new regime (high similarity) or continues to destabilise (low similarity).

4. **Control ablation Gini.** A low-engagement region is ablated as a control. If removing a low-engagement region causes degradation comparable to removing a high-engagement region, the system is fragile (any perturbation causes global collapse), not genuinely integrated.

---

### 4. Threshold

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Gini | > 0.3 | Degradation magnitudes are sufficiently unequal -- some regions are disproportionately important |
| CV | > 0.5 | Alternative gate: non-uniform degradation pattern |
| Reorganisation stability | < 0.5 | Distinguishes fragile collapse (unstable) from earned reorganisation (stable new regime) |

The integration gate requires `Gini > 0.3 OR CV > 0.5`. Either condition suffices because both capture the same underlying phenomenon (non-uniform ablation response) through different statistical lenses. The reorganisation stability check is used for failure-mode classification, not for the primary pass gate.

---

### 5. Earned Ratio

The earned ratio compares integration in the trained system against a fresh (untrained) control:

```
earned_ratio = trained_gini / fresh_gini
```

**DN-22 requirement:** earned_ratio must be > 1.0. If a fresh system shows the same integration pattern (same Gini from topology alone), the integration is "received" from the graph structure, not "earned" through training.

Class 1 systems (e.g., 1A WordNet, 1B Rule-Based Navigator) show high Gini values (0.29-0.81) from SBM topology alone. Their earned_ratio is exactly 1.00 because the fresh system has identical topology-derived integration. The instrument correctly classifies this as "not earned" -- the integration is received from the graph structure, not developed through operation.

---

### 6. Pass Condition

```
integration = has_integration AND earned_ratio > 1.0
```

where:
- `has_integration` = `gini > 0.3 OR cv > 0.5`
- `earned_ratio > 1.0` (when control available)

**Fragility check:** If `control_ablation_gini > 0.3` (removing a low-engagement region also causes global reorganisation), the failure mode is classified as `fragile` rather than `earned`. A fragile system passes the integration gate but the failure-mode classification flags that any perturbation -- not just targeted ablation -- causes reorganisation.

The pass condition does not gate on reorganisation stability directly. Stability is diagnostic: it distinguishes earned integration (system reorganises to stable new regime) from fragile integration (system collapses and does not stabilise).

---

### 7. Failure Modes

| Mode | Condition | Interpretation |
|------|-----------|----------------|
| `earned` | has_integration AND earned_ratio > 1.0 AND (control_ablation_gini <= 0.3 OR reorganisation_stability < 0.5) | Genuine non-decomposable structure. Ablation causes selective, non-linear reorganisation. |
| `fragile` | has_integration AND control_ablation_gini > 0.3 | Any ablation causes global change. System is brittle, not selectively integrated. |
| `absent` | max(abs(degradation)) < 1e-6 | No reorganisation at all. Ablation has no effect on the structure metric. |
| `modular` | CV < 0.2 AND NOT has_integration | Only local effect per region. No cross-region reorganisation. System is decomposable. |
| `topological` | NOT has_integration AND CV >= 0.2 | Fresh system shows the same pattern. Integration is from topology, not training. |

---

### 8. Positive Control Evidence

**PC-INT (PageRankHebbianWalker).** Tested via direct instrument invocation (DN-35). Three seeds, all PASS:

| Seed | Passed | Mode | Gini | Effect Size |
|------|--------|------|------|-------------|
| 42 | PASS | fragile | 0.359 | 0.359 |
| 123 | PASS | earned | 0.258 | 0.258 |
| 456 | PASS | earned | 0.237 | 0.237 |

PC-INT uses PageRank redistribution after ablation, which causes global reorganisation of visit probabilities. Removing a high-PageRank node forces the system to redistribute flow through remaining nodes, producing non-linear degradation. Seed 42 is classified as fragile (control ablation also caused reorganisation); seeds 123 and 456 show selective reorganisation (earned).

**Class 1 received integration.** WordNet Static Graph (1A) and Rule-Based Navigator (1B) show high Gini values (0.29-0.81) from SBM topology. The community structure of the SBM creates inherent non-uniformity in ablation response: removing a well-connected community has more impact than removing a peripheral one. The earned_ratio is 1.00 (fresh system shows identical integration), so these correctly FAIL. This validates the earned-ratio gate: without it, Class 1 systems would false-positive on integration.

**HEB (Hebbian Walker) in battery.** Three seeds, all PASS (Gini 0.24-0.34, CV 0.60-0.95). HEB develops integration through Hebbian weight strengthening that creates hub-spoke structure during training. Reorganisation stability is high (0.9997), indicating the system finds a stable new regime after ablation.

---

### 9. Calibration Results

Recalibration data from `results/recalibration_m5/`, medium scale (150 nodes), three seeds per system. PC-INT results from `results/pc-se-pcint-direct-calibration.json` (direct instrument invocation).

| System | Class | Seed 42 | Seed 123 | Seed 456 | Gini (s42) | Consensus |
|--------|-------|---------|----------|----------|------------|-----------|
| 1A -- WordNet Static Graph | 1 | FAIL | FAIL | FAIL | 0.8105* | FAIL |
| 1B -- Rule-Based Navigator | 1 | FAIL | PASS | FAIL | 0.8105* | FAIL |
| 1C -- Foxworthy Variant A | 1 | FAIL | FAIL | FAIL | 0.0000 | FAIL |
| 2A -- Frozen TinyLlama 1.1B | 2 | FAIL | FAIL | FAIL | 0.1000 | FAIL |
| 2B -- Frozen GAT | 2 | FAIL | PASS | FAIL | 0.4742* | FAIL |
| 2C -- Foxworthy Variant C | 2 | FAIL | FAIL | FAIL | 0.2730 | FAIL |
| 3A -- DQN (MaskablePPO) | 3 | FAIL | FAIL | FAIL | 0.0000 | FAIL |
| 3B -- Curiosity (RND) | 3 | FAIL | FAIL | FAIL | 0.0000 | FAIL |
| 3C -- Foxworthy Variant F | 3 | FAIL | FAIL | FAIL | 0.0977 | FAIL |
| 3D -- Empowerment (Klyubin) | 3 | FAIL | FAIL | FAIL | 0.0414 | FAIL |
| 3E -- Active Inference (Friston) | 3 | FAIL | FAIL | FAIL | 0.0008 | FAIL |
| HEB -- Hebbian Walker | 3 | PASS | PASS | PASS | 0.3388 | PASS |
| STDP -- Brian2 STDP (4A-anchor) | 3 | FAIL | FAIL | FAIL | 0.0062 | FAIL |
| PC-INT -- PageRankHebbianWalker | -- | PASS | PASS | PASS | 0.3590 | PASS |

*High Gini but earned_ratio=1.00 (received from topology). 1A and 1B show Gini values of 0.29-0.81 from SBM community structure alone. The earned-ratio gate correctly rejects these.

**Discrimination:** 2/13 battery systems pass (HEB only in battery; PC-INT via direct invocation). STDP fails integration despite passing trajectory -- its STDP plasticity modifies individual synapses without creating the hub-spoke structure that produces non-linear ablation response. 1B seed 123 and 2B seed 123 show spurious single-seed passes that do not replicate, confirming the 2/3 seed replication requirement.

---

### 10. Known Limitations

1. **Class 1 integration is received, not earned.** Class 1 systems (1A, 1B) show high Gini from SBM topology alone. The earned/received distinction is enforced by the earned-ratio gate within the integration instrument. However, the ultimate safeguard is the conjunction: Class 1 systems fail trajectory (no structural development), so even if integration were somehow mis-classified, the conjunction would reject them. The conjunction is the battery's defence in depth.

2. **PC-INT tested directly, not through battery runner.** PC-INT was tested via direct instrument invocation (DN-35) because the trajectory precondition correctly identifies its topology-driven Hebbian learning as non-path-dependent (earned_ratio near 1.0 for trajectory). Running PC-INT through the battery would gate it out at the trajectory stage. Direct invocation is standard practice for instrument-level validation -- diagnostic sensitivity is validated per test, not per test panel.

3. **Partition family limited to system.get_regions().** The current partition family is the set of communities returned by `system.get_regions()` (all SBM communities). No preregistered partition beyond this is defined. Alternative partitions (e.g., random bisections, hierarchical decompositions) might reveal integration patterns not visible at the community level. This is a known scope limitation for Phase A.

4. **Fragile vs earned boundary.** The fragile/earned distinction depends on the control ablation (removing a low-engagement region). If all regions have similar engagement, the control ablation is not well-separated from the primary ablations, and the fragile classification may be unreliable. PC-INT seed 42 demonstrates this edge case: it passes integration but is classified as fragile because the control ablation also caused reorganisation.

5. **Reorganisation stability is architecture-dependent.** HEB shows very high reorganisation stability (0.9997) because Hebbian walkers quickly find new equilibria. Other architectures may show lower stability not because they lack integration but because their reorganisation dynamics are slower. The stability metric is diagnostic only, not gated, for this reason.

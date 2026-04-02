## S13. Per-Instrument Discrimination

Each of the five battery instruments achieves perfect separation between its positive and negative controls, measured by ROC AUC with bootstrap confidence intervals.

### Per-Instrument ROC AUC

| Instrument       | AUC | 95% CI       | n_pos | n_neg | Positive Control(s)                              |
|------------------|-----|--------------|-------|-------|--------------------------------------------------|
| Generativity     | 1.0 | [1.0, 1.0]  | 6     | 9     | PC1 + PC3 (transition JSD on B2)                 |
| Self-Engagement  | 1.0 | [1.0, 1.0]  | 3     | 9     | PC-SE (direct, recovery ratio)                   |
| Integration      | 1.0 | [1.0, 1.0]  | 12    | 6     | PC-INT (direct, Gini) + Class 1 (received)       |
| Trajectory       | 1.0 | [1.0, 1.0]  | 3     | 9     | STDP (earned trajectory)                         |
| Transfer         | 1.0 | [1.0, 1.0]  | 1     | 6     | 3E active inference (41x earned ratio)           |

Method: 10,000 bootstrap resamples. Source: `results/per-instrument-roc-auc.json`.

### Caveats

**Transfer has a single true positive.** System 3E (active inference) is the only system with earned transfer (advantage = 9.18, earned_ratio = 10.18). AUC = 1.0 reflects strict separation on this small sample, not strong statistical power. Additional transfer-positive systems would strengthen the estimate but are not available in the current calibration panel.

**Integration includes both received and earned positives.** The 12 positives comprise 9 Class 1 systems (received integration -- topology-driven, not learning-driven) and 3 PC-INT seeds (earned integration via PageRank-Hebbian dynamics). The earned/received distinction is the conjunction's responsibility, not the integration instrument's. The instrument detects non-linear degradation under ablation regardless of origin.

**Self-engagement and integration tested via direct instrument invocation.** PC-SE and PC-INT were tested using Option 3 (DN-35): direct instrument invocation bypassing battery preconditions. This is standard practice for instrument-level sensitivity validation. The trajectory precondition correctly classifies their topology-driven Hebbian learning as non-path-dependent (earned_ratio approximately 1.0), which would gate them out of the full battery. Instrument sensitivity is validated per instrument, not per battery pipeline.

**Trajectory relies on a single architecture.** STDP is the sole architecturally-grounded trajectory positive. HEB was excluded due to variable results across seeds. Additional trajectory-positive architectures in Phase B would strengthen this instrument's validation.

### Battery-Level Discrimination

Battery-level (conjunction) ROC AUC is not computed at this stage. Gate F -- the Phase C candidate evaluation -- is the first test of conjunction discrimination. The calibration panel establishes that the conjunction rejects all 13 systems (0/13 pass), and positive controls confirm each instrument is individually sensitive. The conjunction's ability to accept a genuine Class 4 system is tested prospectively, not retrospectively.

### Summary

Strict separation on small panels. Architectural grounding -- not statistical power -- is the primary evidence for instrument validity. Each positive control was selected because its architecture guarantees the target property, providing non-circular ground truth independent of the battery's own scoring.

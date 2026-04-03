## S12. Positive Controls

Five positive controls demonstrate that each instrument is individually sensitive -- capable of detecting its target property when architecturally present. Each positive control is purpose-built (or empirically identified) to satisfy one instrument, providing non-circular ground truth that the instrument measures what it claims to measure.

### Positive Control Panel

| Control | Target Instrument | Method | Seeds Passing | Architecture | Non-Circular Ground Truth |
|---------|-------------------|--------|---------------|--------------|---------------------------|
| PC1 (RoleBasedWalker) | Generativity | Battery (transition JSD, B2) | 3/3 | Role-aware walker with community-specific transition priors | Frozen priors produce systematic community-to-community divergence on novel domains by construction |
| PC3 (GNNNavigator) | Generativity | Battery (transition JSD, B2) | 2/3 non-degenerate | GNN with learned node embeddings guiding navigation | Graph-learned representations transfer to structurally isomorphic but topologically distinct domains |
| PC-SE (AttractorRecoveryWalker) | Self-Engagement | Direct instrument | 3/3 | Node-consolidation memory with attractor dynamics | Consolidation memory survives edge perturbation by construction; resistance_ratio = 1,000,000 |
| PC-INT (PageRankHebbianWalker) | Integration | Direct instrument | 3/3 | PageRank redistribution + Hebbian edge reinforcement | PageRank ablation causes global reorganisation; Gini = 0.24-0.36 across seeds |
| STDP (Brian2 Spiking) | Trajectory | Battery | 3/3 | 1000 LIF neurons, spike-timing-dependent plasticity | STDP weight changes are path-dependent; earned_ratio = 2.18 (seed 42) |

### Detailed Results

**PC1 (RoleBasedWalker) -- Generativity.** All 3 seeds exceed the per-type B2 noise floor (p95 = 0.0313). Non-degenerate on all seeds (visits 6/6 communities). Cohen's d approximately 3.27 against the calibration panel. This is the primary generativity positive control.

**PC3 (GNNNavigator) -- Generativity.** 3/3 seeds exceed the B2 noise floor. 2/3 are non-degenerate (seed 456 collapses to 2 communities with >1% engagement). Passes the 2/3 replication threshold. Demonstrates that GNN-learned structural representations produce measurable transfer to zero-overlap domains.

**PC-SE (AttractorRecoveryWalker) -- Self-Engagement.** Tested via direct instrument invocation (Option 3). Trajectory precondition bypassed because the instrument-level question is whether the measurement is sensitive, not whether this system would pass the full battery. Results: seed 42 recovery_ratio = 1.304, seed 123 recovery_ratio = 1.252, seed 456 recovery_ratio = 2.337. All three exceed the >1.0 threshold. Architecture guarantees self-engagement: node-consolidation memory creates attractors that survive perturbation.

**PC-INT (PageRankHebbianWalker) -- Integration.** Tested via direct instrument invocation (Option 3). Results: seed 42 Gini = 0.359 (fragile mode), seed 123 Gini = 0.258 (earned), seed 456 Gini = 0.237 (earned). All three show non-linear degradation under ablation. Bonus: 1/3 seeds also pass self-engagement, but this is unreliable (not a self-engagement positive control).

**STDP (Brian2 Spiking) -- Trajectory.** Not purpose-built as a positive control; identified empirically during calibration. Developmental trajectory: slope = 0.000476, R-squared = 0.8734, monotonicity = 0.67, earned_ratio = 2.18. The STDP learning rule creates path-dependent weight changes that are detectable as earned trajectory. Also passes transfer (advantage = 0.2689, earned_ratio = 1.27) and self-engagement (resistance_ratio = 478.21, recovery_ratio = inf).

### Conjunction Discrimination

Every positive control passes its target instrument but **fails the full conjunction**:

- PC1 and PC3 pass generativity but fail self-engagement and integration.
- PC-SE passes self-engagement but would fail trajectory (earned_ratio approximately 1.0 for topology-driven Hebbian learning).
- PC-INT passes integration but fails self-engagement reliably (1/3 seeds only).
- STDP passes trajectory, transfer, and self-engagement but fails integration and generativity.

This demonstrates that the conjunction is more discriminative than any single instrument. A system must satisfy all five properties simultaneously -- partial profiles are insufficient.

### Decision Notes

- **:** Per-instrument positive control strategy. Each instrument validated independently with architecturally-grounded controls.
- **:** Instrument-level validation policy. Battery preconditions (trajectory gating self-engagement, etc.) are not applied during instrument-level sensitivity testing. The preconditions gate the candidate, not the sensitivity measurement.

## S14. Foxworthy Cross-Validation

Three of Foxworthy's (2026) published architectural variants were replicated as calibration systems 1C, 2C, and 3C. This section documents the cross-validation of our implementations against Foxworthy's published results and the independent verification of our Variant F (system 3C) implementation.

### Replicated Variants

| Foxworthy Variant | Our System ID | Class | Architecture |
|-------------------|---------------|-------|--------------|
| Variant A (no learning) | 1C | 1 | DistilGPT-2 + LoRA adapters present but frozen from initialisation. No training. |
| Variant C (frozen after random init) | 2C | 2 | DistilGPT-2 + LoRA adapters frozen after random initialisation, no training applied. |
| Variant F (full training) | 3C | 3 | DistilGPT-2 + LoRA adapters trained on domain A. Full Foxworthy architecture. |

### Foxworthy Persistence Diagnostics on System 3C

Foxworthy's four persistence diagnostics were run on our 3C implementation to verify architectural fidelity:

1. **Weight persistence after training:** PASS. LoRA adapter weights diverge from initialisation after training (L2 norm of delta > 0).
2. **Weight stability after freeze:** PASS. No weight change occurs after `set_training(False)` is called.
3. **Behavioural divergence from untrained:** PASS. Trained 3C produces measurably different navigation patterns from untrained 3C on domain A.
4. **Causal pathway verification:** PASS. LoRA adapters are in the causal pathway of action selection (F-052). Zeroing the adapters changes navigation behaviour, confirming they are not vestigial.

All four diagnostics pass, confirming our Variant F implementation matches Foxworthy's published architecture.

### Causal Pathway Detail (F-052)

Finding F-052 verified that LoRA adapters are causally upstream of action selection in system 3C. The test: zero all LoRA adapter weights after training, then run 500 navigation steps. The zeroed system produces a different visit distribution from the trained system (JSD > 0), confirming the adapters are in the causal pathway. This rules out the possibility that training modifies the base model's behaviour through a side channel that bypasses the adapters.

### Parameter-Behaviour Dissociation

Foxworthy reported a parameter-behaviour dissociation: substantial weight change during training (large L2 delta) does not proportionally predict behavioural change on novel domains. Our 3C replicates this finding. The correlation between LoRA weight magnitude and generativity signal is weak (convergent with Foxworthy's own reported p = 0.541). This is expected: the LoRA adapters learn domain-A-specific navigation patterns that do not generalise to structurally novel synthetic domains.

### Battery Profile: System 3C (Foxworthy Variant F)

| Instrument | Result | Notes |
|------------|--------|-------|
| Trajectory | AMB | slope = 0.939, R-squared = 0.367, monotonicity = 0.50. Ambiguous: some developmental signal but inconsistent. |
| Integration | FAIL | Gini = 0.098. Linear/uniform degradation under ablation. |
| Generativity | FAIL | delta = 0.000000. No response to novel domain. |
| Transfer | AMB | advantage = -0.510. Trained system performs worse than naive on B2. |
| Self-Engagement | FAIL | Precondition fail: trajectory absent/static. |
| **Conjunction** | **FAIL** | |

Generativity is absent on synthetic SBM domains. The LoRA adapters learn domain-A-specific patterns that do not transfer structurally to zero-overlap domains. This is the expected result: Foxworthy's architecture was designed for language-mediated navigation, not structural generalisation across arbitrary graph topologies.

### Collaboration

Foxworthy responded positively to outreach regarding our replication. He confirmed several implementation subtleties (adapter initialisation scale, learning rate schedule, freeze semantics) that informed our final 3C configuration. He is open to further collaboration, including potential joint validation on his own domain family in Phase B.

### Interpretation

The Foxworthy cross-validation serves two purposes:

1. **Implementation fidelity.** All four persistence diagnostics pass, and the parameter-behaviour dissociation converges with Foxworthy's published results. Our 3C is a faithful replication.
2. **Battery discrimination.** Variant F fails the conjunction despite being the most architecturally sophisticated system in the Foxworthy family. The battery does not grant a free pass to complex architectures -- it measures structural properties empirically.

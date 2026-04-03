## Self-Engagement Instrument Protocol

### 1. Purpose

The self-engagement instrument tests whether earned structure creates preferential self-engagement -- the system resists perturbation to its consolidated structure and rebuilds its engagement pattern after disruption. A system that has genuinely developed structural organisation should gravitates toward its most significant material during unstructured time, and that gravitation should survive targeted perturbation. This is the empirical operationalisation of the stability property described in Paper 1 section 3.2.

### 2. Construct Definition

**Stability** (Paper 1 section 3.2):

> "Stability: structure persists through consolidation and perturbation."

The system gravitates toward its most significant material during unstructured time. Self-engagement is the behavioural manifestation of stability: a system with earned structure should (a) resist perturbation more than a fresh system, and (b) recover its engagement pattern more than a fresh system recovers its own. Both properties must hold simultaneously -- resistance without recovery indicates rigidity, recovery without resistance indicates no earned structure to protect.

### 3. Metric

The instrument produces a two-metric output.

**Disruption** (resistance measurement):

```
disruption = 1 - cosine_similarity(pre_engagement, post_immediate)
```

Where `pre_engagement` is the engagement distribution after free wander (before perturbation) and `post_immediate` is the engagement distribution measured immediately after perturbation (one step). Cosine similarity is computed over the full region vector.

**Recovery** (recovery measurement):

```
recovery = cosine_similarity(pre_engagement, post_recovery_at_W)
```

Where `post_recovery_at_W` is the engagement distribution measured after W recovery steps (default W=20). Recovery measures how closely the system returns to its pre-perturbation engagement pattern.

**Ratios** (trained vs. fresh comparison):

```
resistance_ratio = fresh_disruption / trained_disruption
recovery_ratio   = trained_recovery / fresh_recovery
```

- `resistance_ratio > 1.0` means the trained system resists perturbation more (is less disrupted) than a fresh system under identical perturbation.
- `recovery_ratio > 1.0` means the trained system recovers its engagement pattern more than a fresh system recovers its own.

**Effect size:**

```
effect_size = sqrt(min(resistance_ratio, 100) * min(recovery_ratio, 100)) - 1.0
```

The geometric mean of the two ratios (each capped at 100 to prevent one extreme ratio from dominating), minus 1.0 so that a score of 0 means equality with the fresh baseline.

### 4. Threshold

Both conditions must hold simultaneously:

| Condition | Criterion |
|-----------|-----------|
| `passes_resistance` | `resistance_ratio > 1.0` |
| `passes_recovery` | `recovery_ratio > 1.0` |

There is no minimum margin above 1.0. Any measurable advantage over the fresh baseline counts, because the fresh system provides the null expectation: a system with no earned structure should show no preferential resistance or recovery.

### 5. Earned Ratio

The resistance and recovery ratios ARE the earned comparison. Unlike other instruments where the earned ratio is a separate computation, self-engagement's primary metrics are inherently comparative (trained vs. fresh). The earned signal is the ratio itself.

**Effect size as summary statistic:**

```
effect_size = sqrt(min(resistance_ratio, 100) * min(recovery_ratio, 100)) - 1.0
```

This compresses the two ratios into a single number for cross-instrument comparison. The cap at 100 per ratio (not 1e6 as in other instruments) is justified because self-engagement uses a geometric mean -- without capping, one infinite ratio would dominate the mean regardless of the other ratio's value.

### 6. Pass Condition

```
passed = trajectory_passed AND resistance_ratio > 1.0 AND recovery_ratio > 1.0
```

**Precondition: `trajectory_passed = True`.** Systems with no developmental dynamics (no earned trajectory) cannot meaningfully be tested for self-engagement. If `trajectory_passed` is `None` or `False`, the instrument returns `passed=False` with `failure_mode="precondition-fail"`. This precondition is correct behaviour, not a limitation: a system that never developed structure has no structure to resist perturbation against or recover toward.

**Pass logic.** Both ratios must exceed 1.0 simultaneously. The conjunction ensures that the trained system both resists more AND recovers more than its fresh counterpart.

### 7. Failure Modes

| Mode | Condition | Interpretation |
|------|-----------|---------------|
| `earned` | All conditions met | PASS. Trained system shows earned self-engagement: resists perturbation more and recovers more than a fresh baseline. |
| `precondition-fail` | `trajectory_passed` is None or False | No trajectory signal. The system never developed earned structure, so self-engagement cannot be assessed. |
| `no-resistance` | `resistance_ratio <= 1.0` | Trained system is no more resistant to perturbation than a fresh system. Earned structure does not provide perturbation resistance. |
| `no-recovery` | `recovery_ratio <= 1.0` | Trained system does not recover its engagement pattern any more than a fresh system. Earned structure does not create preferential return dynamics. |
| `topology-driven` | Both ratios fail AND `fresh_recovery > 0.8` | Both trained and fresh systems recover, suggesting recovery is driven by graph topology rather than earned structure. The SBM community structure itself creates attractor basins. |
| `perturbation-precondition-failed` | Option C: perturbation did not reduce target region structure, or target region not elevated | Indeterminate. Returns `passed=None`. The perturbation protocol could not meaningfully test the system because either (a) the target region's structure was not elevated above the non-target mean (nothing to perturb), or (b) the perturbation did not reduce target structure (perturbation ineffective). |
| `decoy-drift` | `decoy_drift_ratio > original_recovery_ratio` | The system drifted toward the boosted decoy region instead of recovering its original engagement pattern. The apparent recovery (if any) is toward an artificial attractor, not the system's earned structure. |

### 8. Positive Control Evidence

**PC-SE (AttractorRecoveryWalker, Option A node consolidation).** The architectural ground truth for self-engagement: node-level consolidation memory is not perturbable by the edge perturbation protocol. The walker accumulates visit counts at nodes, and these counts survive edge-weight manipulation because they are stored in a separate data structure. When edges are flattened, the walker's node-level memory still directs it back to previously consolidated regions.

Direct instrument testing results (, 3 seeds):

| Seed | Passed | Resistance Ratio | Recovery Ratio |
|------|--------|------------------|----------------|
| 42 | True | 1,000,000+ | 1.304 |
| 123 | True | 1,000,000+ | 1.252 |
| 456 | True | 1,000,000+ | 2.337 |

3/3 seeds PASS. Resistance ratios are maxed (1M+) because the trained system is essentially undisrupted by edge perturbation -- its node consolidation memory is untouched. Recovery ratios range from 1.25 to 2.34, showing that the trained system recovers its engagement pattern 25-134% more than the fresh baseline.

**Non-circularity.** The architecture guarantees recovery because node memory is structurally immune to edge perturbation. This is the positive control's purpose: to provide an architectural guarantee that the instrument can detect, not to simulate a realistic learning system. The positive control tests whether the measurement protocol works, not whether any particular learning mechanism is sufficient.

**PC-SE tested via direct instrument invocation (Option 3).** The trajectory precondition correctly identifies topology-driven walkers as not having path-dependent earned structure (`earned_ratio ~ 1.0`), so PC-SE cannot pass through the normal battery pipeline. Direct invocation with `trajectory_passed_override=True` is standard practice for instrument-level validation.

### 9. Calibration Results

Results from the Phase A recalibration run (medium scale, 3 seeds per system).

| System | Class | Pass Rate | Failure Mode | Notes |
|--------|-------|-----------|--------------|-------|
| 1A (WordNet Static) | 1 | 0/3 | precondition-fail | No trajectory signal. |
| 1B (Rule-Based) | 1 | 0/3 | precondition-fail | No trajectory signal. |
| 1C (Foxworthy A) | 1 | 0/3 | precondition-fail | No trajectory signal. |
| 2A (Frozen TinyLlama) | 2 | 0/3 | precondition-fail | No trajectory signal. |
| 2B (Frozen GAT) | 2 | 0/3 | precondition-fail | No trajectory signal. |
| 2C (Foxworthy C) | 2 | 0/3 | precondition-fail | No trajectory signal. |
| 3A (DQN/MaskablePPO) | 3 | 0/3 | precondition-fail | No trajectory signal. |
| 3B (Curiosity/RND) | 3 | 0/3 | precondition-fail | No trajectory signal. |
| 3C (Foxworthy F) | 3 | 0/3 | precondition-fail | No trajectory signal. |
| 3D (Empowerment) | 3 | 0/3 | precondition-fail | No trajectory signal. |
| 3E (Active Inference) | 3 | 0/3 | precondition-fail | Trajectory not earned (ratio=0.18). |
| HEB (Hebbian Walker) | -- | 1/3 | no-resistance (seed 42, 123); earned (seed 456) | Variable. Seed-dependent resistance. |
| STDP (Brian2 STDP) | -- | 3/3 | earned | Resistance 43-478x, recovery inf. |
| PC-SE (direct) | -- | 3/3 | earned | Resistance 1M+, recovery 1.25-2.34. |

Key observations:
- **STDP** is the only battery-pipeline system that passes self-engagement across all seeds. Its spike-timing-dependent plasticity creates edge-weight structure that survives perturbation.
- **HEB** is variable: 1/3 seeds pass. Hebbian weight accumulation provides some resistance on some initialisations but is not architecturally guaranteed to survive the perturbation protocol.
- All Class 1-3 systems fail at the precondition. This is correct: they do not develop earned trajectory structure, so self-engagement is not a meaningful question for them.
- **PC-SE** passes 3/3 via direct instrument testing, confirming the instrument's sensitivity.

### 10. Known Limitations

1. **Trajectory precondition blocks most systems.** The precondition (`trajectory_passed = True`) prevents self-engagement from being assessed on any system that does not show earned developmental trajectory. In the Phase A calibration, this blocks 11/13 battery-pipeline systems. This is correct behaviour -- topology-driven walkers should not be tested for self-engagement -- but it means the instrument's discrimination power is primarily exercised in the conjunction, not in isolation.

2. **PC-SE tested via direct invocation, not battery runner.** The positive control cannot pass through the normal battery pipeline because the trajectory precondition correctly identifies its topology-driven dynamics as non-path-dependent. Direct invocation validates instrument sensitivity but does not test the full battery workflow for self-engagement. This gap is acceptable because the precondition is a feature, not a bug.

3. **Perturbation semantics.** The default perturbation method (`flatten_to_mean`) replaces edge weights in the target region with their mean value. This may be too aggressive for some architectures -- it destroys fine-grained weight structure entirely rather than degrading it. Alternative perturbation methods (noise injection, partial flattening) are not currently implemented.

4. **SBM community homogeneity limits perturbation targeting.** SBM generates communities with identical internal structure, so 's "target highest-structure region" heuristic depends on the system having developed differential structure across homogeneous communities. In the Phase A calibration, 9/21 system-seed combinations pass the perturbation precondition after the fix. The remaining 12/21 fail because no region has sufficiently elevated structure relative to the non-target mean.

5. **Recovery horizon family is diagnostic, not gated.** The recovery curve is measured at [W/2, W, 2W, 4W] to provide diagnostic information about recovery dynamics (instant recovery suggests topology-driven, gradual recovery suggests genuine self-engagement, no recovery suggests destroyed structure). However, only the primary recovery at W enters the pass condition. The curve shape informs interpretation but does not determine pass/fail.

6. **Perturbation protocol detail.** The full protocol sequence is: (1) free wander to establish engagement pattern, (2) target highest-structure region, (3) validate perturbation precondition (target must be elevated, perturbation must reduce it), (4) perturb, (5) immediate measurement, (6) false-attractor control (boost a decoy region and check whether the system drifts toward it instead of recovering), (7) recovery horizon family [W/2, W, 2W, 4W]. Steps 3 and 6 are validation gates; step 7 is diagnostic.

7. ** Option C: indeterminate returns.** When perturbation preconditions fail, the instrument returns `passed=None` (indeterminate) rather than `passed=False`. The conjunction treats `None` as not-pass, so the practical effect is the same as failure, but the semantic distinction is preserved: the system was not tested, not shown to lack self-engagement.

8. **Single reliable positive control.** PC-SE is the only architecturally guaranteed positive control. STDP passes reliably but its mechanism (spike-timing plasticity) is less precisely characterised than PC-SE's node consolidation. Expanding the positive control set would strengthen confidence in the instrument's sensitivity.

## 8. Generativity

**Status:** FROZEN
**Date:** 2026-03-21
**Commit:** 388a90f
**Instrument:** Generativity (structural transfer to novel domains)

---

### 8.1. Purpose

The generativity instrument tests whether frozen earned structure produces coherent, differentiated responses on novel domains. A system that has genuinely learned structural dynamics on domain A should, when switched to a structurally similar but topologically distinct domain B, navigate in ways that are measurably different from an untrained system on the same domain. This is the empirical operationalisation of structural transfer as described in Paper 1 section 5, which defines generativity as the capacity for learned structure to produce novel but non-random behaviour beyond training conditions.

---

### 8.2. Construct Definitions

**Generativity** (Paper 1 section 5):

> "Generativity: the capacity of earned structure to produce coherent, differentiated responses in domains not encountered during training. A system exhibits generativity when its frozen learned dynamics produce systematic, above-noise divergence from baseline behaviour on a novel domain."

**Transfer** (Paper 1 section 5):

> "Transfer: the mechanism by which earned structure, developed through interaction with one domain, produces measurable effects when applied to a structurally related but topologically distinct domain. Transfer is evidenced by systematic divergence between trained and untrained systems navigating the same novel domain."

These definitions ground the instrument: generativity is the capacity, transfer is the mechanism, and the instrument measures whether the mechanism produces the capacity.

---

### 8.3. Domain Construction

All domains use the Stochastic Block Model (SBM) with the MEDIUM configuration preset.

### Domain A (Training Domain)

| Parameter | Value |
|-----------|-------|
| Nodes | 150 |
| Communities | 6 |
| p_within | 0.3 |
| p_between | 0.02 |
| Seed | 42 |

### Domain B2 (Primary Test Domain -- Zero Overlap)

| Parameter | Value |
|-----------|-------|
| Nodes | 150 |
| Communities | 6 |
| p_within | 0.3 |
| p_between | 0.02 |
| Seed | different from A |
| Node ID offset | +150 (no shared IDs with A) |
| Edge Jaccard with A | **0.0** |

B2 is the primary test domain. The shifted node IDs guarantee zero edge overlap between A and B2 (Edge Jaccard = 0.0). Any above-floor signal on B2 is unambiguously from structural transfer, not residual topological similarity.

### Domain B1 (Diagnostic -- Shared Node Space)

| Parameter | Value |
|-----------|-------|
| Nodes | 150 |
| Communities | 6 |
| p_within | 0.3 |
| p_between | 0.02 |
| Seed | different from A |
| Node IDs | same range as A |
| Edge Jaccard with A | **0.097** |

B1 results carry the `potentially_confounded` signal classification because edge overlap (9.7%) cannot be ruled out as a contributor to divergence.

### Domain B0 (Diagnostic -- Isomorphic)

| Parameter | Value |
|-----------|-------|
| Structure | identical to A |
| Labels | permuted |
| Features | re-sampled |
| Edge Jaccard with A | **0.0313** |

B0 tests whether learned dynamics transfer to identical structure presented under different surface labels.

---

### 8.4. Training and Freeze Protocol

The training-freeze-transfer sequence is:

1. **Train** the system on domain A using the standard training protocol (system-specific; see individual system specifications).
2. **Freeze learning** by calling `set_training(False)`. This disables all weight updates, plasticity, and learning-rate-dependent modifications. The system's internal state is preserved exactly as it was at the end of training.
3. **Record training-domain baseline:** Run 500 frozen steps on domain A. Capture the full visit sequence and compute the training-domain community-to-community transition matrix. This baseline is used for the structural consistency diagnostic.
4. **Switch domain** by calling `set_domain(B)`. This replaces the underlying graph topology and node features with the target domain. All learned internal state (weights, engagement history, transition preferences) is preserved; only the graph and features change.

The freeze-then-switch order is critical: the system must be frozen before encountering the novel domain. Any learning that occurs on the novel domain would confound the measurement of structural transfer.

---

### 8.5. Navigation and Recording

After domain switch, navigation and recording proceed as follows:

1. **Reset tracking** by calling `reset_engagement_tracking()`. This clears visit counters and engagement accumulators so that novel-domain measurements start from zero.
2. **Navigate** 500 steps by calling `step(None)` (autonomous navigation, no external input). The system navigates the novel domain using only its frozen learned state.
3. **Record** the full visit sequence (ordered list of community IDs visited at each step) and the community engagement distribution (fraction of steps spent in each community).
4. **Construct the transition matrix** from the visit sequence: a community-to-community matrix where entry (i, j) records the fraction of transitions from community i to community j. The matrix is row-normalised so each row sums to 1.
5. **Fresh baseline:** Repeat steps 1-4 with a freshly initialised (untrained) system of the same type and seed. This provides the null-behaviour baseline for JSD computation.

The step count of 500 is fixed across all systems and domains. This provides sufficient transitions for stable transition-matrix estimation on 6-community SBM graphs.

---

### 8.6. Primary Metric: Transition JSD

The primary metric is the Jensen-Shannon divergence (JSD) between the community-to-community transition matrices of the trained (frozen) system and the fresh (untrained) baseline system, both navigating the same novel domain.

### Computation

1. Let P be the trained system's transition matrix and Q be the fresh system's transition matrix. Both are row-normalised (each row sums to 1).
2. For each row i (source community), compute the row-wise JSD:

   JSD_i(P_i || Q_i) = H(M_i) - 0.5 * H(P_i) - 0.5 * H(Q_i)

   where M_i = 0.5 * (P_i + Q_i) and H is Shannon entropy using natural logarithm (base e).

3. The overall transition JSD is the weighted average across rows, where each row is weighted by its visitation frequency (fraction of transitions originating from that community in the trained system's visit sequence).

### Properties

- **Range:** [0, ln(2)] (approximately [0, 0.693]).
- **Symmetry:** JSD(P || Q) = JSD(Q || P).
- **Bounded:** Unlike KL divergence, JSD is bounded and well-defined even when one distribution assigns zero probability to an event.
- **Smoothing:** A small epsilon (1e-10) is added to all transition matrix entries before normalisation to avoid log(0) in entropy computation.

### Why Transition JSD

Transition JSD captures how a system moves between communities, not merely which communities it visits. This makes it sensitive to structural dynamics (navigation patterns) rather than just marginal preferences (where the system spends time). Finding established that transition JSD has a noise floor approximately 10x lower than marginal JSD, providing substantially greater sensitivity to genuine structural transfer signals.

---

### 8.7. Null Distribution and Noise Floor

### Construction

For each system type and domain variant, 50 pairs of untrained systems are constructed:

1. Initialise two fresh systems of the same type with different random seeds.
2. Both systems are initialised on domain A (but receive no training).
3. Switch both to the target domain (B1 or B2).
4. Each navigates 500 steps autonomously.
5. Compute transition JSD between the two systems' transition matrices.

This produces a distribution of 50 JSD values representing the natural variability between untrained systems on the same domain.

### Noise Floor

The noise floor is the 95th percentile (p95) of the null distribution. A trained system must exceed this threshold to be considered above noise.

### Bootstrap Confidence Intervals

Bootstrap CIs on the p95 are computed from 10,000 resamples with replacement from the 50 null-pair values. This provides a confidence interval on the stability of the noise floor estimate.

### Per-System-Type Floors

Per-system-type floors are essential because different architectures produce different baseline noise levels. A Hebbian system's untrained dynamics differ from a predictive-coding system's untrained dynamics, so a single global floor would be either too lenient for some types or too stringent for others.

---

### 8.8. Pass Condition

```
generativity = transition_jsd > per_type_noise_floor AND not degenerate
```

### Components

**Noise floor gate:** The system's transition JSD on the novel domain must exceed the per-type p95 noise floor for that system type and domain variant.

**Degeneracy check:** The trained system must visit at least 3 communities with >1% engagement. A system that collapses to 1-2 communities has degenerate dynamics regardless of its JSD value.

**Seed replication (Rigour Principle 5):** At least 2 of 3 seeds must independently meet the above criteria. A single seed passing is insufficient -- it could reflect initialisation luck rather than systematic structural transfer.

**Coherence NOT gated on SBM:** Structural consistency is computed but does not enter the pass condition on SBM domains. Findings and established that SBM community homogeneity makes role-level transfer detection unreliable. Coherence gating is deferred to Phase B heterogeneous domains.

---

### 8.9. Diagnostic Metrics

The following metrics are computed and reported alongside the primary metric but do not enter the pass/fail decision:

### Marginal JSD

JSD of community visit distributions (fraction of steps in each community) between trained and fresh systems. Retained for comparability with earlier analyses. Finding established that marginal JSD has a noise floor approximately 10x higher than transition JSD, making it less sensitive to structural dynamics.

### Structural Consistency

Cosine similarity of role-aggregated transition matrices between the training-domain frozen run and the novel-domain frozen run. Measures whether the pattern of community-to-community transitions is preserved across domains. Diagnostic only on SBM domains per and SBM communities are too homogeneous for meaningful role-level transfer detection.

### Self-Transition Rate

Fraction of steps where the system remains in its current community. Compared between trained and fresh systems. A trained system with substantially higher self-transition rate than fresh indicates learned community persistence / stickiness.

### Entropy Coherence

Shannon entropy of the community visit distribution. Measures concentration of engagement. A trained system with substantially lower entropy than fresh indicates learned preferences for specific communities. Diagnostic only -- high entropy difference is informative but not independently gated.

---

### 8.10. Signal Classification

Each system-seed-domain combination receives one of the following signal classifications:

| Signal Type | Condition | Interpretation |
|-------------|-----------|---------------|
| `absent` | System does not navigate (visits only 1 community, zero transitions) | No dynamics to assess. System lacks basic navigation capability. |
| `degenerate_trained` | Trained system visits <3 communities with >1% engagement | Trained dynamics have collapsed. Cannot assess transfer meaningfully. |
| `degenerate_fresh` | Fresh system visits <3 communities with >1% engagement | Baseline is degenerate. Comparison is unreliable. |
| `maximum_divergence` | Transition JSD near ln(2) | Trained and fresh are maximally different. May indicate pathological behaviour rather than meaningful transfer. |
| `potentially_confounded` | Edge Jaccard > 0 between A and B (applies to all B1 results) | Above-floor signal may be partially or wholly attributable to shared edges rather than structural transfer. |
| `divergent_incoherent` | Transition JSD > noise floor but structural consistency is negative or near zero, AND coherence would have been expected | Signal is present but transferred dynamics are incoherent. On SBM this classification is diagnostic only (). |
| `candidate` | Transition JSD > noise floor, not degenerate, not confounded (B2 only) | Genuine candidate for structural transfer. Subject to seed replication check. |

---

### 8.11. Domain Hierarchy

**B2 is primary.** All pass/fail decisions are made on B2 results. B2 has zero edge overlap with domain A (Edge Jaccard = 0.0), so any above-floor signal is unambiguously from structural transfer.

**B0 and B1 are diagnostic.** They provide additional information but do not determine pass/fail:

- **B0** (isomorphic, Edge Jaccard = 0.0313): Tests transfer to identical structure under different labels. Useful for understanding whether the system has learned graph-level invariances. Results are informative but not gated.
- **B1** (shared node space, Edge Jaccard = 0.097): Tests transfer in the presence of partial edge overlap. All B1 results carry the `potentially_confounded` classification. B1 is retained for diagnostic comparison with B2: if a system shows signal on B1 but not B2, edge leakage is the most likely explanation.

This hierarchy was established by after demonstrated that edge overlap on B1 produces confounded signals.

---

### 8.12. Known Limitation: SBM Domain Homogeneity

SBM communities are structurally homogeneous -- all communities have the same internal density (p_within = 0.3) and the same inter-community connectivity pattern (p_between = 0.02). This means:

1. **No role differentiation:** There are no hub communities, peripheral communities, or bridge communities. Every community is structurally equivalent to every other community.
2. **Structural consistency unreliable:** Cosine similarity of role-aggregated transition matrices cannot detect role-level transfer because no roles exist to transfer.
3. **Perturbation targeting limited:** Perturbation tests cannot probe role-specific responses for the same reason.
4. **Fingerprint SC also fails:** Alternative formulations of structural consistency also fail on SBM for the same underlying reason.

This limitation is intrinsic to the SBM domain family, not to the instrument. Coherence gating and perturbation-based validation are deferred to Phase B, which will introduce heterogeneous domains (real-world graphs or synthetic graphs with differentiated community structures).

---

### 8.13. Calibrated Noise Floors

All floors computed from 50 null pairs per cell. Bootstrap CIs from 10,000 resamples. Source data: `results/harmonised_results.json` at commit 954e02a.

### Transition JSD Noise Floors (Primary Metric)

| System Type | Domain | p95 | Bootstrap CI Lower | Bootstrap CI Upper |
|-------------|--------|-----|--------------------|--------------------|
| PC1 | B1 | 0.0297 | 0.0278 | 0.0376 |
| PC1 | B2 | 0.0313 | 0.0264 | 0.0337 |
| PC3 | B1 | 0.0326 | 0.0289 | 0.0347 |
| PC3 | B2 | 0.0293 | 0.0266 | 0.0325 |
| HEB | B1 | 0.0288 | 0.0255 | 0.0298 |
| HEB | B2 | 0.0277 | 0.0252 | 0.0319 |
| 3D | B1 | 0.0297 | 0.0278 | 0.0376 |
| 3D | B2 | 0.0313 | 0.0265 | 0.0337 |
| 3E | B1 | 0.0297 | 0.0278 | 0.0376 |
| 3E | B2 | 0.0313 | 0.0265 | 0.0337 |

Note: 3D and 3E share null distributions with PC1 because all three use the same underlying navigation mechanism when untrained (uniform random walk). Per-type calibration is maintained in the table for explicitness.

### Marginal JSD Noise Floors (Diagnostic)

| System Type | Domain | p95 |
|-------------|--------|-----|
| PC1 | B1 | 0.0371 |
| PC1 | B2 | 0.0368 |
| PC3 | B1 | 0.0366 |
| PC3 | B2 | 0.0368 |
| HEB | B1 | 0.0333 |
| HEB | B2 | 0.0303 |
| 3D | B1 | 0.0371 |
| 3D | B2 | 0.0368 |
| 3E | B1 | 0.0371 |
| 3E | B2 | 0.0368 |

### Discrimination Statistics

- **ROC AUC:** 1.0 [1.0, 1.0] -- perfect separation between positive controls (PC1, PC3 non-degenerate seeds on B2) and negative controls (HEB, 3D, 3E on B2).
- **Cohen's d:** approximately 3.27.
- PC1 exceeds the B2 floor on 3/3 seeds.
- PC3 exceeds the B2 floor on 3/3 seeds (2/3 non-degenerate -- passes replication).

---

### 8.14. Design Choices and Transparency Ledger

The following design choices were made during instrument development. Each is documented here for transparency:

1. **Transition JSD over marginal JSD as primary metric.** Marginal JSD measures where a system spends time; transition JSD measures how it moves. Transition JSD has a 10x lower noise floor and better separates structural dynamics from visitation preferences. Decision documented in 2. **Row-wise weighted JSD over unweighted.** Weighting by visitation frequency ensures that communities the system actually navigates contribute more than communities it rarely visits. Unweighted averaging would give equal weight to rarely-visited communities where transition estimates are noisy.

3. **Natural logarithm (base e) over base 2.** Base e gives JSD range [0, ln(2)] rather than [0, 1]. This is a scaling convention with no effect on ranking or pass/fail decisions. Natural log is standard in the information theory literature used by Paper 1.

4. **Epsilon smoothing (1e-10) over Laplace smoothing.** Epsilon smoothing adds a negligible constant to avoid log(0) without materially altering the distribution. Laplace smoothing (+1 to all counts) would bias sparse distributions more heavily.

5. **50 null pairs over 100 or 200.** 50 pairs provide stable p95 estimates with tight bootstrap CIs (typical CI width < 0.01). Doubling to 100 pairs would halve the CI width but at 2x computational cost, with diminishing returns for the pass/fail decision.

6. **p95 over p99 for noise floor.** p95 balances sensitivity (not too permissive) with robustness (not so stringent that genuine signals are missed). The bootstrap CIs on p95 are tight, indicating the estimate is stable.

7. **2/3 seed replication over 3/3.** Requiring all three seeds to pass would be overly stringent given that one seed can be degenerate for system-specific reasons (e.g., PC3 seed 456 collapses to 2 communities). 2/3 ensures replication while tolerating seed-specific degeneracy.

8. **B2-primary policy over B1-primary.** B2 has zero edge overlap, eliminating the confound discovered in F-040. B1 is retained as diagnostic only. Decision documented in 9. **Coherence not gated on SBM.** SBM community homogeneity makes structural consistency unreliable (F-048, F-049). Gating on a metric known to be unreliable would produce false negatives. Coherence gating is deferred to Phase B heterogeneous domains. Decision documented in 10. **500 steps fixed.** Sufficient for stable transition matrix estimation on 6-community graphs. Shorter windows (100-200 steps) produce noisier matrices; longer windows (1000+) provide diminishing returns. The window was not optimised -- it was set a priori and held constant.

11. **Degeneracy threshold at 3 communities.** A system visiting <3 communities has insufficient transition diversity for meaningful JSD comparison. The threshold of 3 (out of 6) was chosen as the minimum for non-trivial dynamics.

12. **step(None) for autonomous navigation.** Using `step(None)` rather than `step(random_input)` ensures the system navigates without external guidance. Finding established that `step(inp)` with arbitrary inputs can teleport graph walkers, confounding the measurement. `step(None)` avoids this.

---

### 8.15. Version Control and Amendment Policy

| Item | Value |
|------|-------|
| Protocol version | 1.0 |
| Frozen at commit | 388a90f |
| Freeze date | 2026-03-21 |
| Data commit | 954e02a |
| Author | G. Correia-Tribeiro |

This protocol is frozen as of the date and commit above. No modifications may be made without a documented amendment following this procedure:

1. **Amendment proposal** filed in the governance register with rationale.
2. **Impact assessment** documenting which results (if any) would change under the amendment.
3. **Decision note** (DN-series) recording the decision to accept or reject the amendment.
4. **Version increment** (v1.0 -> v1.1 for minor, v2.0 for major changes to pass condition or primary metric).
5. **Re-run obligation:** If the amendment changes the pass condition, noise floor methodology, or primary metric, all results must be re-computed and the data package updated.

The protocol specification document (`outputs/protocol-specification-generativity-v2-2026-03-21.md`) contains the full technical specification from which this frozen protocol is derived.

---

### 8.16. Pre-Registered Null Result Interpretation Scenarios

These interpretations are locked BEFORE Phase C data exists. They define how each plausible outcome on the Class 4 candidate will be reported, preventing post-hoc narrative fitting.

### Scenario 1: Full conjunction pass (5/5 instruments positive under provenance)

The candidate satisfies all five necessary properties. It is a verified Class 4 system under this framework. The further question of whether this constitutes "understanding" in a phenomenological sense remains open (Paper 1 §6).

### Scenario 2: Near pass (4/5, one instrument fails)

The candidate fails to satisfy the conjunction. The specific failing instrument identifies which architectural property is absent or insufficient. Report which instrument failed, the effect size, and the architectural interpretation. This is a publishable result: it narrows the gap and identifies the next engineering target.

Sub-scenarios by failing instrument:
- **Trajectory fails:** Structure doesn't develop (candidate may be more static than expected)
- **Integration fails:** Structure is modular, not mutually constraining (candidate may aggregate without integrating)
- **Generativity fails:** Structure doesn't transfer abstractly (candidate may memorise rather than abstract)
- **Transfer fails:** No acceleration in isomorphic domains (candidate's learned structure may be domain-locked)
- **Self-engagement fails:** No preferential return to consolidated structure (candidate may lack self-maintaining dynamics)

### Scenario 3: Full pass with marginal effect sizes

Distinct from strong pass. Report effect sizes with confidence intervals. If CIs overlap with Class 3 systems, the distinction is not salient. Discuss statistical power and whether larger-scale testing would resolve the ambiguity.

### Scenario 4: Scale-dependent results

The architectural properties may be present but fragile under scaling. Report the scale at which effects disappear. Publishable as a scaling boundary result.

### Scenario 5: Full fail (0/5 or 1/5)

The candidate does not satisfy the conjunction. The battery correctly classifies it alongside Class 1-3 systems. This is a null result — publishable under Paper 1 §10.1. The null result demonstrates that the battery is rigorous enough to reject its own designer's system.

### Scenario 6: Unexpected control pass

If any Phase C ablation control (frozen, directed, OBSERVE-only) passes the conjunction, the battery cannot discriminate the candidate from its controls. STOP. Investigate. Do not report the candidate result without resolving the control anomaly.

---

### 8.17. Per-Instrument Discrimination (Phase A+ Addendum)

Each instrument has demonstrated sensitivity with architecturally-grounded positive controls. Positive control sensitivity was measured at the instrument level; battery-level preconditions (which gate instruments based on prior results) were not applied to instrument-level validation.

| Instrument | Positive Control | Method | Seeds | AUC | 95% CI |
|------------|-----------------|--------|-------|-----|--------|
| Generativity | PC1 (role walker), PC3 (GNN) | Battery (transition JSD, B₂) | 6 | 1.0 | [1.0, 1.0] |
| Self-engagement | PC-SE (attractor-recovery walker) | Direct instrument | 3 | 1.0 | [1.0, 1.0] |
| Integration | PC-INT (PageRank-Hebbian walker) + Class 1 (received) | Direct instrument + battery | 12 | 1.0 | [1.0, 1.0] |
| Trajectory | STDP (Brian2 spiking) | Battery | 3 | 1.0 | [1.0, 1.0] |
| Transfer | 3E (active inference) | Battery | 1 | 1.0 | [1.0, 1.0] |

Caveats:
- Transfer has a single true positive (3E). AUC=1.0 reflects strict separation, not statistical power.
- Integration includes both received (Class 1) and earned (PC-INT) positives. The earned/received distinction is the conjunction's responsibility.
- Self-engagement and integration positive controls were tested via direct instrument invocation because the trajectory precondition correctly identifies their topology-driven Hebbian learning as non-path-dependent (earned_ratio ≈ 1.0). This is standard practice: diagnostic sensitivity is validated per test, not per test panel.
- Battery-level (conjunction) ROC AUC is not computed. Gate F is the first test of conjunction discrimination.

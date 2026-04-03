# Phase A+ Sign-Off

**Date:** 2026-03-21
**Status:** CLOSED
**Commit:** 388a90f
**Signed off by:** G. Correia-Tribeiro

---

## 1. Phase A+ Objective

Phase A+ was created to resolve the generativity instrument after Phase A found 0/13 systems exhibiting structural transfer. The objective was to determine whether the null result reflected genuine absence of generativity in the tested systems or an instrument deficiency, and to produce a calibrated, confound-free protocol capable of detecting structural transfer if it exists.

---

## 2. Evidence Chain

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Root cause of 0/13 identified | PASS | three independent causes (metric insensitivity, confounded domains, placeholder thresholds) |
| 2 | Positive controls constructed | PASS | PC1 and PC3 produce above-floor signal on B2 (3/3 and 2/3 seeds respectively) |
| 3 | Negative controls confirmed | PASS | HEB, 3D, 3E all below floor on B2 (0/3 seeds each) |
| 4 | Noise floors calibrated | PASS | Per-type p95 from 50 null pairs, bootstrap CIs from 10,000 resamples (F-041, F-050) |
| 5 | Confound eliminated | PASS | B2 domain has Edge Jaccard = 0.0 with A; all B1 results classified as potentially_confounded (F-040) |
| 6 | Primary metric validated | PASS | Transition JSD achieves ROC AUC = 1.0 [1.0, 1.0] separating positive from negative controls |
| 7 | Sensitivity validated | PASS | Cohen's d ~ 3.27; transition JSD floor ~10x lower than marginal JSD floor |
| 8 | Instrument bug fixed | PASS | step(inp) teleportation fixed; step(None) used throughout |
| 9 | Protocol frozen | PASS | generativity-instrument-protocol-v1.0.md, commit 388a90f |
| 10 | Reproducibility bundle | PASS | harmonised_results.json + null_distribution_raw_samples.csv + requirements-v3.txt |
| 11 | Coherence limitation documented | PASS | SC unreliable on SBM (F-048, F-049); coherence gating deferred to Phase B |
| 12 | 0/13 reconfirmed | PASS | 0/13 confirmed under harmonised transition protocol on B2 |

---

## 3. What Phase A+ Establishes

1. **The generativity instrument works.** It can detect structural transfer when it exists (PC1, PC3) and correctly reports absence when it does not (HEB, 3D, 3E, and all 13 original systems). ROC AUC = 1.0 with stable bootstrap CIs.

2. **The 0/13 result is real.** None of the 13 systems in the current battery exhibit structural transfer on SBM domains under the calibrated transition JSD protocol. This is not an instrument artefact -- it is a genuine empirical finding.

3. **Transition JSD is the correct primary metric.** It provides approximately 10x better sensitivity than marginal JSD and achieves perfect discrimination between positive and negative controls.

4. **The B2 domain eliminates the edge-leakage confound.** Zero edge overlap between training and test domains means any above-floor signal is unambiguously from structural transfer.

5. **Per-type noise floors are stable.** Bootstrap CIs on the p95 thresholds are tight (typical width < 0.01), indicating that 50 null pairs provide reliable calibration.

6. **The protocol is reproducible.** All results are traceable to specific commits, raw data files, and computational parameters.

---

## 4. What Phase A+ Does NOT Establish

1. **Structural consistency as a gating metric.** Coherence (structural consistency / ) is computed but not gated on SBM domains. SBM community homogeneity makes role-level transfer detection unreliable (F-048, F-049). Coherence gating requires heterogeneous domains and is deferred to Phase B.

2. **Generativity on non-SBM domains.** All results are on synthetic SBM graphs. Whether the instrument and its calibration transfer to real-world networks, scale-free graphs, or heterogeneous synthetic graphs is unknown. Phase B will address this.

3. **Perturbation-based causal validation.** SBM homogeneity limits perturbation targeting. The perturbation audit found 19/21 fail before the step(inp) fix, and 9/21 after. Perturbation-based validation of structural transfer requires differentiated community roles, deferred to Phase B.

4. **Why 0/13 systems lack generativity.** The phase establishes the empirical fact but does not explain the architectural reasons. Understanding why specific system families fail to produce structural transfer is a Phase B research question.

5. **Sufficiency of the metric bundle.** The current diagnostic metrics (marginal JSD, structural consistency, self-transition, entropy) may not capture all aspects of structural transfer. Additional metrics may be needed for heterogeneous domains.

---

## 5. Known Limitations

1. **SBM domain homogeneity.** All communities are structurally equivalent (same p_within, same p_between). This prevents coherence gating, limits perturbation targeting, and may mask transfer signals that depend on role differentiation. Addressed by Phase B heterogeneous domains.

2. **PC3 seed 456 degeneracy.** PC3 collapses to 2 communities on seed 456 across all domain variants. This is a known property of the PC3 architecture under specific initialisation conditions. It does not invalidate the 2/3 replication result but limits the robustness claim for PC3.

3. **3D and 3E produce identical navigation.** Both systems reset to uniform engagement on domain switch, producing indistinguishable results. This is correct null behaviour but means the instrument cannot differentiate between "no transfer because the system lacks it" and "no transfer because the system's transfer mechanism is not engaged by SBM structure."

4. **Transition JSD is sensitive to sample size.** With 500 steps on 6 communities, some community-to-community transitions may be observed rarely, producing noisy matrix entries. The epsilon smoothing (1e-10) mitigates log(0) issues but does not eliminate estimation noise. Larger step counts or different domain sizes may require recalibration.

5. **Bootstrap CIs assume i.i.d. null pairs.** The 50 null pairs use different random seeds, which should be approximately independent. However, any systematic bias in the initialisation procedure would propagate to all pairs, making the CI overly optimistic.

---

## 6. Findings Register (through F-051)

| Finding | Summary |
|---------|---------|
| | Three independent causes of 0/13 generativity: metric insensitivity, confounded domains, placeholder thresholds |
| | Empowerment 0/5 is genuine (topological) -- empowerment systems lack internal dynamics that survive domain switch |
| | Instrument measured structural metric, not behavioural JSD -- original generativity used wrong comparison basis |
| | HEB self-engagement seed-dependent (1/3) -- only one seed shows any signal, insufficient for replication |
| | Placeholder threshold produced false positive (HEB conjunction) -- uncalibrated threshold flagged noise as signal |
| | HEB generativity signal is seed noise -- B1 edge overlap confounds apparent signal |
| | Null JSD distributions established per-type p95 floors -- 50 pairs per cell, bootstrap CIs confirm stability |
| | step(inp) teleported graph walkers -- external input bypassed graph topology, producing artefactual transitions |
| | Positive control signal below marginal JSD noise floor -- marginal JSD too insensitive to detect known transfer |
| | Complete positive control results compiled -- PC1 and PC3 fully characterised across B0/B1/B2 |
| | Transition JSD resolves sensitivity (10x lower floor) -- transition matrices capture dynamics marginal distributions miss |
| | Perturbation audit: 19/21 fail before fix, 9/21 after fix -- SBM homogeneity limits perturbation targeting |
| | Structural consistency appears to validate PC1 -- high SC values on PC1 suggest role-level transfer |
| | SC temperature-dependent and fragile on SBM -- small parameter changes flip SC sign, unreliable for gating |
| | Fingerprint SC also fails -- SBM too homogeneous for any structural consistency formulation to be reliable |
| | ROC AUC = 1.0, bootstrap CIs stable -- perfect discrimination between positive and negative controls |
| | 0/13 confirmed under harmonised transition protocol -- original null result holds with calibrated instrument |

---

## 7. Decision Register (through )

| Decision | Summary |
|----------|---------|
| | Phase A+ scope: investigate 0/13, build positive controls, recalibrate instrument |
| | Positive control strategy: use PC1 (predictive coding with transfer mechanism) and PC3 (enhanced PC with stronger transfer) |
| | step(None) policy: all generativity navigation uses autonomous stepping to avoid teleportation artefact |
| | Hebbian walker publishable: HEB is publishable for self-engagement but does not exhibit generativity |
| | Generativity test design: transition JSD primary, per-type noise floors, 2/3 seed replication, degeneracy check |
| | Structural consistency definition and SBM addendum: SC is diagnostic only on SBM, coherence gating deferred to Phase B |
| | Domain difficulty ladder: B0 (isomorphic), B1 (shared node space), B2 (zero overlap) -- establishes confound gradient |
| | Phase B gating decisions: B2-primary policy, coherence deferral, heterogeneous domains required for SC validation |

---

## 8. Phase B Entry Conditions

Phase A+ is complete. The following conditions are met for Phase B entry:

1. Calibrated generativity protocol frozen (v1.0).
2. Positive controls validated (PC1 3/3, PC3 2/3 on B2).
3. Negative controls confirmed (HEB, 3D, 3E 0/3 on B2).
4. ROC AUC = 1.0 with stable bootstrap CIs.
5. All findings and decisions documented.
6. Reproducibility bundle complete.

Phase B will:
- Introduce heterogeneous domains (real-world graphs, synthetic graphs with differentiated community roles).
- Validate structural consistency as a gating metric on heterogeneous domains.
- Investigate perturbation-based causal validation with role-differentiated targets.
- Explore why 0/13 systems lack generativity at the architectural level.

---

## 9. Artefact Index

| Artefact | Location |
|----------|----------|
| Protocol freeze | `protocol/generativity-instrument-protocol-v1.0.md` |
| Protocol specification | `outputs/protocol-specification-generativity-v2-2026-03-21.md` |
| Harmonised results | `results/harmonised_results.json` |
| Null distribution samples | `results/null_distribution_raw_samples.csv` |
| Reproducibility bundle | `reproducibility/README.md` + `reproducibility/requirements-v3.txt` |
| This sign-off | `governance/phase-a-plus-sign-off-2026-03-21.md` |

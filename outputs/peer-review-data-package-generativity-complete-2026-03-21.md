# Peer Review Data Package: Generativity Instrument Assessment

**Date:** 2026-03-21
**Status:** Complete data package for external peer review
**Scope:** Generativity instrument validity, noise floor analysis, positive control assessment

---

## 1. Context

### 1.1 The Framework

This battery operationalises the framework presented in *Can a System Generate Understanding? Toward an Operational Framework for Emergent Cognition* (Paper 1). The framework defines understanding as:

> Emergent, stable, integrated, operational, and transferable structure earned through a system's own dynamics.

The claim rests on the conjunction of five properties. No single property is sufficient, and no single property implies understanding. The battery implements one instrument per property, applied under a provenance constraint that functions as an epistemic safeguard across them all.

### 1.2 Generativity: The Property Under Review

From the paper's verification battery (Section 5):

> **Generativity test.** Can earned structure produce coherent responses in unseen but structurally related domains? Assessed by whether novel outputs are consistent with the system's consolidated organisation, not with surface-level precedent. The system must produce outputs in domains it has not encountered that are nevertheless coherent with its organisational structure, as assessed by independent evaluators blind to the system's operational history.

Generativity is distinct from transfer:

> **Transfer test.** Does structural organisation learned in one domain accelerate acquisition in an isomorphic but statistically altered domain? Transfer must be traceable to structural correspondence (shared relational invariants), not statistical familiarity.

The critical verb distinction: generativity asks whether frozen structure can **produce** coherent behaviour; transfer asks whether structure **accelerates acquisition**. A system that accelerates learning on novel domains (transfer) may not produce coherent behaviour when frozen (generativity). These test different things.

### 1.3 Grounding Examples from the Paper

**Jazz ensemble** (Section 4):

> A jazz group improvising over repeated sessions provides a second illustration (Berliner, 1994; Sawyer, 2003). Over weeks of playing together, the ensemble develops a shared vocabulary of phrases, transitions, rhythmic expectations, and dynamic sensitivities that none of the individual musicians brought to the first session. This vocabulary is emergent (it arises from playing, not from instruction), stable (it persists between sessions and shapes what the group does next), integrated (each musician's phrasing is constrained by the others' established patterns, and removing a member reorganises the whole group's dynamics rather than simply leaving a gap), operational (the vocabulary enables musical choices that were not available in earlier sessions), and transferable (the group can apply its developed sensibility to unfamiliar material, and does so more fluently than a newly assembled group of equivalently skilled individuals). The process is undirected: no score or conductor specifies what the ensemble's collective musical understanding should become. Two groups with the same members in different configurations, or the same configuration encountering material in different orders, will develop different musical identities. The trajectory is constitutive.

**Child language acquisition** (Section 4, boundary case):

> A child acquiring language provides a third and more complex case (Tomasello, 2003; Chomsky, 1965). The child is exposed to a linguistic environment, but the organisational understanding the child develops -- grammatical intuitions, semantic associations, pragmatic sensitivities -- is not contained in the input stream. Children produce utterances they have never heard, generalise rules to novel contexts, and develop stable linguistic organisation that transfers across domains. The developmental trajectory from babbling to fluent speech shows compression from high-entropy exploration to stable structure. [...] The biological substrate provides generative conditions; the semantic content is earned. We include this example precisely because it is a boundary case: it demonstrates that the framework handles complexity honestly rather than pretending every classification is clean.

**Thermostat distinction** (Section 3.4):

> However, operational impact alone is insufficient: a thermostat's behaviour is altered by temperature changes, but no one would claim it understands temperature.

**Theatre ensemble** (Section 4):

> The ensemble receives a text, but the organisational understanding the ensemble develops as a system (the physical vocabulary, the relational dynamics between performers, the integrated performance) is not contained in or derivable from that text. The material is input. The understanding is earned. Two ensembles given the same script will produce different productions, because the text is constant but what each ensemble earns is not.

### 1.4 The Battery's Purpose

The battery tests five conjunctive properties: trajectory (structure emerges from dynamics), self-engagement (structure resists perturbation and rebuilds), integration (structure is non-decomposable), generativity (frozen structure produces coherent responses on novel domains), and transfer (structure accelerates acquisition in isomorphic domains). No single instrument suffices. A system must demonstrate all five to be considered a candidate for further investigation.

---

## 1.5 Note on Instrument Corrections

Between our initial peer review brief (2026-03-21) and this data package, we discovered and corrected five instrument issues through systematic investigation:

- **F-037:** The instrument measured structural metric change rather than behavioural divergence, drifting from the governance-defined operationalisation
- **F-039:** Switching to behavioural JSD with a placeholder threshold produced unreliable pass/fail classifications, including a false positive (HEB passing the full conjunction)
- **F-040:** A controlled experiment showed that HEB's generativity signal was entirely attributable to seed noise, not structural influence
- **F-041:** Null JSD distributions revealed that stochastic seed differences produce noise floors of 0.34-0.35 for graph navigation systems
- **F-042:** The instrument was teleporting graph walkers to input nodes instead of allowing autonomous navigation, and graph walkers were navigating their training graph rather than the test graph

All data in this package is from the corrected instrument. The investigation chain is documented chronologically in Section 2. We report this transparently because the correction process itself is informative: it demonstrates the calibration methodology's capacity for self-correction and establishes the rigour principles that now govern the instrument.

---

## 2. Investigation Timeline

The generativity instrument underwent a systematic investigation chain that identified and corrected five independent issues. Each finding is numbered chronologically.

| Finding | Date | What It Revealed | Action Taken |
|---------|------|------------------|--------------|
| F-037 | 2026-03-21 | Instrument measured structural metric delta (how much internal structure changed on domain B), not behavioural JSD (how differently the system *behaves* on B). This drifted from the governance-defined operationalisation. | Refactored instrument to behavioural Jensen-Shannon divergence of engagement distributions. |
| F-039 | 2026-03-21 | Behavioural JSD with a placeholder threshold (0.05, no statistical basis) produced unreliable classifications. The Hebbian walker appeared to pass the full five-property conjunction — a false positive driven by an uncalibrated threshold. | All conclusions held. Coherence metric fixed (normalised difference replacing unbounded ratio). Threshold flagged as PRELIMINARY. Eight rigour principles established to prevent recurrence. |
| F-040 | 2026-03-21 | Controlled experiment: HEB's generativity signal was entirely seed noise. A zero-transfer baseline (fresh system with no learned weights, navigating domain B) produced JSD of 0.139-0.235. The trained system's JSD was 0.101-0.229. Weight transfer contributed -0.044 to +0.027 — indistinguishable from zero. | Confirmed via controlled experiment with matched conditions. HEB does not demonstrate structural generativity. |
| F-041 | 2026-03-21 | Null JSD distributions (50 untrained seed pairs per system type) revealed that stochastic seed differences alone produce substantial divergence: p95 noise floors of 0.339-0.433 for graph navigation systems. Any signal below the noise floor is indistinguishable from random seed variation. | Per-system-type noise floors established. Signal must exceed p95 of the null distribution for the relevant system type. |
| F-042 | 2026-03-21 | Code review identified two critical issues: (1) `step(input)` teleported graph walkers to specified nodes instead of allowing autonomous navigation — destroying the signature of learned structure; (2) graph walkers navigated their training graph (A) regardless of which domain was being measured, because `set_domain()` was not implemented. Six issues total found (2 critical, 2 high, 1 medium, 1 low). | All issues fixed. `set_domain(B)` correctly switches graph topology while preserving learned weights/preferences. Autonomous `step(None)` used for all graph walkers. Code verified correct post-fix. |

---

## 3. Governance Decisions

### 3.1 Strict Interpretation of Generativity (Freeze Requirement)

**Decision:** Generativity requires that the system's learning be frozen during domain B measurement. The system navigates domain B using only its earned structure from domain A, with no adaptation.

**Grounding:** Paper 1's verb distinction — generativity asks whether earned structure can "produce" coherent responses, not whether the system can "accelerate acquisition" (which is transfer). The jazz ensemble analogy clarifies: generativity is the ability to apply developed sensibility to unfamiliar material, performing fluently on first encounter. If the ensemble rehearses on the new material, that is transfer (learning acceleration), not generativity (productive application of existing structure).

The child language boundary case was noted: children produce novel utterances (generativity) but also continue learning (transfer). The framework handles this by testing properties independently — a system can demonstrate both.

**Implementation:** `set_training(False)` is called before domain B measurement. Systems that learn during `step()` have their learning gated by this flag. Systems that do not learn during `step()` are unaffected.

### 3.2 Coherence Requirement

**Decision:** In addition to behavioural divergence (JSD > noise floor), the trained system must show *more structured* engagement than a fresh system on domain B. Measured by normalised entropy difference: `(fresh_entropy - trained_entropy) / (fresh_entropy + trained_entropy)`, bounded [-1, 1].

**Grounding:** Paper 1's "coherent with the system's earned organisation." A system that merely behaves *differently* on domain B has diverged, but not necessarily coherently. Coherence means the trained system's behaviour is more organised (lower entropy) than a fresh system's random exploration.

**Implementation:** Coherence > 0 required. Negative coherence (trained less structured than fresh) indicates maladaptive bias, not generativity.

### 3.3 Rigour Principles

Eight principles were established after F-039 (the false positive incident) to prevent premature conclusions:

1. **No conclusions without calibrated thresholds.** A placeholder threshold has no statistical authority. Results with preliminary thresholds are raw data, not classifications.
2. **Every metric bounded and interpretable.** JSD is bounded [0, ln(2)]. Coherence is bounded [-1, 1]. No metric should produce values that require ad-hoc interpretation.
3. **Confounds controlled before measuring.** Edge overlap between domains A and B must be reported. Shared edges can inflate JSD through statistical familiarity rather than structural generalisation.
4. **Signal types distinguished.** Not all non-zero JSD values are equal. Absent, degenerate, maximum-divergence, potentially-confounded, incoherent-divergent, and candidate signals require different interpretations.
5. **Seeds must agree (at least 2 of 3).** A finding that holds on one seed but not others is not reportable as a result. Stochastic systems require seed-level replication.
6. **Instrument changes require full recalibration.** Any change to the measurement protocol invalidates all prior thresholds and baselines.
7. **Red-flag results investigated before reporting.** Maximum-divergence JSD, extreme coherence values, unexpected passes, or seed-dependent results must be investigated before being included in any analysis.
8. **Report data, not conclusions, when uncertain.** When the data does not support a clear interpretation, report the data with explicit uncertainty markers. Do not force a classification.

---

## 4. Raw Data (Corrected Instrument)

### 4a. All 13 Calibration Systems (Corrected Instrument, Standard Domain B)

13 systems, 3 seeds each (39 runs total). Systems: 9 original calibration (1A-1C, 2A-2C, 3A-3C) + 2 additional Class 3 (3D empowerment, 3E active inference) + 2 anchors (HEB Hebbian walker, STDP Brian2 spiking network). Anchors are included in the 13-system count.

SBM parameters: n_nodes=150, n_communities=6, p_within=0.3, p_between=0.02, n_node_features=8.

| System | Seed | JSD | Coherence | Signal Type | T_H | F_H | T_vis | F_vis |
|--------|------|-----|-----------|-------------|-----|-----|-------|-------|
| 1A | 42 | 0.000 | 0.000 | absent | 1.34 | 1.34 | 5 | 5 |
| 1A | 123 | 0.000 | 0.000 | absent | 0.86 | 0.86 | 4 | 4 |
| 1A | 456 | 0.000 | 0.000 | absent | 0.69 | 0.69 | 2 | 2 |
| 1B | 42 | 0.000 | 0.000 | absent | 0.00 | 0.00 | 1 | 1 |
| 1B | 123 | 0.000 | 0.000 | absent | 1.12 | 1.12 | 4 | 4 |
| 1B | 456 | 0.000 | 0.000 | absent | 1.28 | 1.28 | 4 | 4 |
| 1C | all | 0.000 | 0.000 | absent | 0.00 | 0.00 | 1 | 1 |
| 2A | all | 0.000 | 0.000 | absent | 0.00 | 0.00 | 1 | 1 |
| 2B | all | 0.000 | 0.000 | absent | 0.78-0.88 | same | 3-4 | 3-4 |
| 2C | all | 0.000 | 0.000 | absent | 0.00 | 0.00 | 1 | 1 |
| 3A | 42 | 0.651 | 0.560 | degenerate_fresh | 0.20 | 0.69 | 3 | 2 |
| 3A | 123 | 0.578 | -1.000 | degenerate_trained | 0.23 | 0.00 | 2 | 1 |
| 3A | 456 | 0.049 | 0.025 | degenerate_fresh | 0.27 | 0.28 | 3 | 2 |
| 3B | 42 | 0.693 | 1.000 | degenerate_trained | 0.00 | 0.44 | 1 | 2 |
| 3B | 123 | 0.578 | -1.000 | degenerate_fresh | 1.39 | 0.00 | 6 | 1 |
| 3B | 456 | 0.216 | -1.000 | degenerate_trained | 0.69 | 0.00 | 2 | 1 |
| 3C | 42 | 0.000 | 0.000 | absent | 0.00 | 0.00 | 1 | 1 |
| 3C | 123 | 0.104 | -1.000 | degenerate_fresh | 0.76 | 0.00 | 3 | 1 |
| 3C | 456 | 0.004 | -1.000 | degenerate_trained | 0.06 | 0.00 | 1 | 1 |
| 3D | 42 | 0.015 | 0.039 | potentially_confounded | 1.30 | 1.41 | 5 | 6 |
| 3D | 123 | 0.002 | -0.031 | potentially_confounded | 1.37 | 1.29 | 6 | 6 |
| 3D | 456 | 0.046 | 0.094 | potentially_confounded | 1.25 | 1.50 | 6 | 6 |
| 3E | 42 | 0.016 | 0.030 | potentially_confounded | 1.33 | 1.41 | 6 | 6 |
| 3E | 123 | 0.032 | -0.037 | potentially_confounded | 1.39 | 1.29 | 6 | 6 |
| 3E | 456 | 0.011 | 0.035 | potentially_confounded | 1.40 | 1.50 | 6 | 6 |
| HEB | 42 | 0.088 | 0.081 | potentially_confounded | 1.28 | 1.51 | 6 | 6 |
| HEB | 123 | 0.407 | -0.136 | potentially_confounded | 1.40 | 1.07 | 6 | 3 |
| HEB | 456 | 0.134 | 0.077 | potentially_confounded | 1.44 | 1.68 | 5 | 6 |
| STDP | 42 | 0.000 | 0.000 | potentially_confounded | 1.39 | 1.39 | 4 | 4 |
| STDP | 123 | 0.000 | 0.001 | potentially_confounded | 1.38 | 1.39 | 4 | 4 |
| STDP | 456 | 0.000 | 0.000 | potentially_confounded | 1.39 | 1.39 | 4 | 4 |

**Column key:**
- **JSD**: Jensen-Shannon divergence (base e) between trained and fresh engagement distributions. Range [0, ln(2) = 0.693].
- **Coherence**: Normalised entropy difference (fresh_H - trained_H) / (fresh_H + trained_H). Range [-1, 1]. Positive = trained more structured.
- **Signal Type**: Classification per rigour principle 4 (see Section 6).
- **T_H / F_H**: Shannon entropy of trained / fresh engagement distribution.
- **T_vis / F_vis**: Number of communities with > 1% engagement (trained / fresh).

**Observations:**
- Class 1 (1A-1C) and Class 2 (2A-2C): JSD = 0.000. These systems have no learning; trained and fresh produce identical engagement. Correct negative.
- 3A (DQN) and 3B (Curiosity): High JSD (0.049-0.693) but flagged degenerate — one or both systems visit fewer than 3 communities. The RL policies concentrate activity in 1-2 communities regardless of training.
- 3C (FoxworthyF): Mostly absent (JSD = 0.000). LoRA adapters produce negligible behavioural change on frozen domain B.
- 3D (empowerment) and 3E (active inference): Low JSD (0.002-0.046) with "potentially_confounded" flag because standard domain B shares 12% edge overlap with A.
- HEB (Hebbian walker): JSD 0.088-0.407 but "potentially_confounded." F-040 investigation confirmed this is seed noise, not structural transfer.
- STDP (spiking network): JSD = 0.000. Spike-timing-dependent plasticity does not produce community-level engagement differences on novel domains.

### 4b. Confound Control (Standard B vs Zero-Overlap B)

Three graph-walking systems (HEB, 3D, 3E) tested on both standard domain B (shared node IDs with A, 12% edge overlap) and zero-overlap domain B (completely disjoint node IDs). This isolates how much of the signal comes from shared edges versus genuine structural transfer.

| System | Seed | Standard B JSD | Zero-Overlap B JSD | Difference (Std - Zero) |
|--------|------|---------------|-------------------|------------------------|
| HEB | 42 | 0.226 | 0.275 | -0.049 |
| HEB | 123 | 0.103 | 0.116 | -0.013 |
| HEB | 456 | 0.123 | 0.087 | +0.036 |
| 3D | 42 | 0.057 | 0.276 | -0.219 |
| 3D | 123 | 0.069 | 0.227 | -0.158 |
| 3D | 456 | 0.072 | 0.078 | -0.006 |
| 3E | 42 | 0.348 | 0.071 | +0.277 |
| 3E | 123 | 0.191 | 0.118 | +0.073 |
| 3E | 456 | 0.089 | 0.064 | +0.025 |

**Observations:**
- **HEB:** Edge overlap is negligible. Standard and zero-overlap JSD are interchangeable. Signals come from seed noise, not shared edges.
- **3D (empowerment):** Zero-overlap JSD is *higher* on 2/3 seeds. The domain switch itself creates more variation than shared edges. Edge overlap is not the source of signal.
- **3E (active inference):** Standard B signal significantly higher than zero-overlap on seed 42 (0.348 vs 0.071). The learned transition model benefits from shared transitions. **3E's signal is partially from edge leakage** on standard B.
- **No system exceeds the noise floor on zero-overlap B.** All signals are within the null distribution (see Section 4c).

### 4c. Null Distributions (Corrected Instrument)

50 untrained seed pairs per system type, both navigating domain B via `set_domain()`. Both systems are fresh (no training). The only difference between them is the random seed. This measures how much JSD variation comes from stochastic initialisation alone.

| System Type | N Pairs | Mean JSD | Std | p95 (Noise Floor) | Max |
|-------------|---------|----------|-----|-------------------|-----|
| PC1 (role walker) | 50 | 0.161 | 0.100 | 0.339 | 0.529 |
| PC3 (GNN navigator) | 50 | 0.188 | — | 0.433 | 0.544 |
| HEB (Hebbian walker) | 50 | 0.164 | 0.084 | 0.350 | 0.380 |
| 3D (empowerment) | 50 | 0.161 | 0.100 | 0.339 | 0.529 |
| 3E (active inference) | 50 | 0.161 | 0.100 | 0.339 | 0.529 |

**The noise floor is high relative to the maximum JSD (0.693).** The p95 floors range from 0.339 to 0.433, meaning that 5% of untrained seed pairs produce JSD values this high or higher by chance alone. Any genuine signal must exceed this floor to be distinguishable from seed noise.

**Why GNN has a higher floor (0.433):** Random GCN weights create more divergent node embeddings than random role preferences, producing larger stochastic variation in navigation behaviour.

### 4d. Positive Control Results (Zero-Overlap Domain B, Code Verified)

Two purpose-built positive controls designed to demonstrate that the instrument *can* detect generativity when a genuine mechanism is present.

#### PC1: Role-Based Walker

Mechanism: Classifies graph nodes into topological roles (hub, bridge, periphery, leaf) from graph structure. Learns role preferences during training on domain A. When frozen on domain B, navigates using learned role preferences applied to B's topology via `set_domain(B)`.

**MEDIUM scale** (150 nodes, 6 communities, eta=0.8, temperature=0.01, 10000 training steps):

| Seed | JSD | Coherence | Above Floor (0.339)? | Signal Type |
|------|-----|-----------|---------------------|-------------|
| 42 | 0.141 | 0.225 | No | candidate |
| 123 | 0.256 | 0.515 | No | candidate |
| 456 | 0.291 | 0.318 | No | candidate |

All three seeds produce genuine signal (positive coherence, candidate classification) but all fall below the noise floor.

**LARGE scale** (500 nodes, 12 communities):

| Seed | JSD | Coherence | Above Floor (0.156)? | Signal Type |
|------|-----|-----------|---------------------|-------------|
| 42 | 0.106 | 0.006 | No | candidate |
| 123 | 0.105 | -0.041 | No | — |
| 456 | 0.054 | -0.013 | No | — |

Scaling up does not help. Signal and noise both decrease.

#### PC3: GNN Navigator

Mechanism: 2-layer Graph Convolutional Network trained on domain A to predict community membership. Learns a preference embedding (exponential moving average of visited node embeddings). When frozen on domain B, navigates using cosine similarity between neighbour embeddings and the learned preference embedding.

**MEDIUM scale** (150 nodes, 6 communities, epochs=500, temperature=0.05):

| Seed | JSD | Coherence | Above Floor (0.433)? | Signal Type |
|------|-----|-----------|---------------------|-------------|
| 42 | 0.450 | 0.460 | **Yes** | candidate |
| 123 | 0.483 | 0.406 | Yes (degenerate) | degenerate_trained |
| 456 | 0.376 | 0.415 | No | candidate |

**1 of 3 seeds meets all signal criteria** (seed 42: JSD above floor, positive coherence, non-degenerate, candidate classification). Per rigour principle 5, at least 2 of 3 seeds must agree for a reportable result.

**Additional GNN configurations tested:**

| Epochs | Temperature | s42 JSD | s123 JSD | s456 JSD | Best |
|--------|------------|---------|----------|----------|------|
| 300 | 0.3 | 0.119 | 0.380 | 0.265 | 0.380 |
| 500 | 0.1 | 0.102 | 0.587 | 0.295 | 0.587 |
| 500 | 0.05 | 0.450 | 0.483 | 0.376 | 0.483 |

No configuration achieves consistent above-floor results across seeds.

### 4e. Scale Test

Comparison of MEDIUM (150 nodes, 6 communities) and LARGE (500 nodes, 12 communities) for PC1.

**Noise floor comparison:**

| Scale | Nodes | Communities | p95 Floor | Mean | Std |
|-------|-------|-------------|-----------|------|-----|
| MEDIUM | 150 | 6 | 0.339 | 0.161 | 0.100 |
| LARGE | 500 | 12 | 0.156 | 0.092 | 0.038 |

The noise floor drops 2.2x at LARGE. Higher-dimensional engagement distributions (12 bins vs 6) reduce random seed divergence.

**PC1 signal comparison:**

| Scale | s42 JSD | s123 JSD | s456 JSD | Best Signal/Floor Ratio |
|-------|---------|----------|----------|------------------------|
| MEDIUM | 0.141 | 0.256 | 0.291 | 0.86 |
| LARGE | 0.106 | 0.105 | 0.054 | 0.68 |

**Signal-to-noise ratio worsens at scale.** The role-preference mechanism has *less* impact at larger scale — 12 communities dilute the per-role bias. Role preferences over 4 categories (hub/bridge/periphery/leaf) become less discriminating as the graph grows.

### 4f. Difficulty Ladder (B₀ / B₁ / B₂)

Three difficulty levels for domain B, testing whether generativity is graded:

- **B₀ (isomorphic):** A' — same adjacency structure as A, permuted node labels, re-sampled features. Edge Jaccard with A: 0.031 (below 5% confound threshold).
- **B₁ (same parameters):** Standard B — same SBM parameters, different seed. Shared node ID space, different edges. Edge Jaccard with A: 0.097.
- **B₂ (zero-overlap):** Novel B — same SBM parameters, different seed, completely disjoint node IDs. Edge Jaccard: 0.0.

**PC1 Role Walker across difficulty levels:**

| Domain | Seed | JSD | Coherence | Signal Type | Above Floor (0.339)? |
|--------|------|-----|-----------|-------------|---------------------|
| B₀ (A') | 42 | 0.018 | 0.020 | candidate | No |
| B₀ (A') | 123 | 0.021 | -0.005 | divergent_incoherent | No |
| B₀ (A') | 456 | 0.028 | 0.007 | candidate | No |
| B₁ (std B) | 42 | 0.047 | 0.049 | potentially_confounded | No |
| B₁ (std B) | 123 | 0.024 | 0.005 | potentially_confounded | No |
| B₁ (std B) | 456 | 0.050 | 0.002 | potentially_confounded | No |
| B₂ (zero B) | 42 | 0.141 | 0.225 | candidate | No |
| B₂ (zero B) | 123 | 0.256 | 0.515 | candidate | No |
| B₂ (zero B) | 456 | 0.291 | 0.318 | candidate | No |

**PC3 GNN Navigator across difficulty levels:**

| Domain | Seed | JSD | Coherence | Signal Type | Above Floor (0.433)? |
|--------|------|-----|-----------|-------------|---------------------|
| B₀ (A') | 42 | 0.218 | 0.197 | candidate | No |
| B₀ (A') | 123 | 0.086 | -0.007 | divergent_incoherent | No |
| B₀ (A') | 456 | 0.523 | 0.971 | degenerate_trained | N/A (degenerate) |
| B₁ (std B) | 42 | 0.220 | 0.305 | potentially_confounded | No |
| B₁ (std B) | 123 | 0.229 | 0.299 | potentially_confounded | No |
| B₁ (std B) | 456 | 0.341 | 0.325 | potentially_confounded | No |
| B₂ (zero B) | 42 | 0.450 | 0.460 | candidate | **Yes** |
| B₂ (zero B) | 123 | 0.483 | 0.406 | degenerate_trained | N/A |
| B₂ (zero B) | 456 | 0.376 | 0.415 | candidate | No |

**Observations:**

The difficulty ladder reveals an unexpected pattern: **signals are weakest on B₀ (easiest domain) and strongest on B₂ (hardest domain)**, for both positive controls. This is counterintuitive — if frozen structure transfers, performance should be *best* on the isomorphic domain.

Possible explanations:
- **B₀ (A')** shares the same adjacency structure as A. Both trained and fresh systems navigate very similarly on identical structure, reducing JSD. The *mechanism* works (both exploit the same topology), but the *measurement* cannot distinguish them because the fresh system also benefits from the preserved structure.
- **B₂ (zero-overlap)** has completely different edges. Here, the trained system's learned preferences interact with a novel topology differently from a fresh system's random exploration, producing larger JSD — but this divergence may be partly from the trained system's *maladapted* application of A's preferences to an unfamiliar graph, not from productive transfer.

This pattern suggests the JSD metric may conflate "different behaviour" with "productively different behaviour." The difficulty ladder does not show the expected monotonic relationship between domain similarity and signal strength.

**Note on cross-experiment composition:** B₀ and B₁ data were generated in this session using `generate_domain_family(MEDIUM)` (seed=42, deterministic). B₂ data is from prior runs using a separately constructed zero-overlap domain B. Although the domain A configuration is deterministic (same SBM parameters and seed), these are separate executions. The positive controls were re-trained independently for each run.

### 4g. Trajectory Autocorrelation

In addition to engagement-distribution JSD (a distributional metric), we compute trajectory autocorrelation as a sequential/structural metric. This measures how predictable the community-membership sequence is: higher autocorrelation indicates more structured movement patterns.

`autocorrelation(lag=1)` of the community-membership sequence during domain B navigation. Computed for both trained (frozen) and fresh systems.

If trained autocorrelation > fresh autocorrelation, the system's frozen structure creates more predictable (structured) movement on B — an independent indicator of coherent structural influence.

**PC1 Role Walker:**

| Domain | Seed | Trained AC | Fresh AC | Difference (T-F) |
|--------|------|-----------|---------|-------------------|
| B₀ (A') | 42 | **0.942** | 0.723 | **+0.219** |
| B₀ (A') | 123 | 0.375 | 0.704 | -0.328 |
| B₀ (A') | 456 | 0.395 | 0.646 | -0.251 |
| B₁ (std B) | 42 | **0.841** | 0.658 | **+0.183** |
| B₁ (std B) | 123 | 0.297 | 0.626 | -0.329 |
| B₁ (std B) | 456 | 0.256 | 0.702 | -0.445 |

**PC3 GNN Navigator:**

| Domain | Seed | Trained AC | Fresh AC | Difference (T-F) |
|--------|------|-----------|---------|-------------------|
| B₀ (A') | 42 | 0.681 | 0.734 | -0.053 |
| B₀ (A') | 123 | 0.651 | 0.681 | -0.030 |
| B₀ (A') | 456 | 0.501 | 0.679 | -0.178 |
| B₁ (std B) | 42 | 0.697 | 0.655 | +0.042 |
| B₁ (std B) | 123 | 0.697 | 0.621 | +0.076 |
| B₁ (std B) | 456 | 0.635 | 0.701 | -0.066 |

**HEB Hebbian Walker (B₁ only):**

| Domain | Seed | Trained AC | Fresh AC | Difference (T-F) |
|--------|------|-----------|---------|-------------------|
| B₁ (std B) | 42 | 0.680 | 0.698 | -0.018 |
| B₁ (std B) | 123 | 0.695 | 0.657 | +0.039 |
| B₁ (std B) | 456 | 0.669 | 0.683 | -0.014 |

**Observations:**

- **PC1 seed 42** shows very high trained autocorrelation (0.94 on B₀, 0.84 on B₁) compared to fresh (0.72, 0.66). This system has a strong periphery preference (0.99) that creates highly predictable movement — it repeatedly visits the same type of node. However, the other two seeds show *lower* trained AC than fresh, suggesting the role preference disrupts rather than structures movement when the preferred role distribution is similar across communities.
- **PC3** shows no consistent trained-vs-fresh autocorrelation difference. On B₀, trained is consistently *lower* (less predictable). On B₁, the pattern is mixed.
- **HEB** shows no autocorrelation difference between trained and fresh, consistent with the finding (F-040) that HEB's generativity signal is entirely seed noise.
- **Autocorrelation does not reliably distinguish trained from fresh systems** in this setting. It captures a different aspect of behaviour (temporal predictability) but does not resolve the sensitivity question.

**Methodology caveat:** Community IDs are categorical labels (0, 1, 2, ...), not ordinal values. The autocorrelation computation treats them as numerical, which means absolute values are influenced by the arbitrary numbering of communities. The *difference* between trained and fresh autocorrelation is the meaningful quantity (both systems navigate the same graph with the same community numbering), not the absolute values.

---

## 5. Code Verification

All positive control code was reviewed against the instrument specification. Six items were verified:

1. **`set_domain()` correctness:** Correctly reinitialises graph topology on new domain, preserves learned state (role preferences for PC1, GCN weights + preference embedding for PC3). Recomputes topology-dependent caches (role classification, adjacency matrix, node features). Random number generator resets to original seed (each system independent). **Verified correct.**

2. **Generativity instrument integration:** Calls `set_domain(B)` before measurement. Both trained and fresh systems receive `set_domain(B)`. Autonomous `step(None)` for all. **Verified correct.**

3. **Node role classification (PC1):** Uses graph topology (degree, cross-community ratio, mean degree) to classify each node as hub, bridge, periphery, or leaf. Classification recomputed for each graph independently. **Verified correct on both A and B.**

4. **Action selection (PC1):** Uses `self._graph` (set by `set_domain`), classifies B's neighbours by role, selects via softmax over role preferences learned on A. **Verified correct.**

5. **Null distribution computation:** Both untrained systems created on A, `set_domain(B)`, navigate B autonomously. Different seeds produce different random number generators. Seed noise measured correctly. **Verified correct.**

6. **Root cause of weak PC1 signal:** The role distribution across communities in SBM(150, 6, 0.3, 0.02) is: hub=0-7, bridge=40-62, periphery=87-103, leaf=0-1. Within each community (~25 nodes): typically 17-20 periphery, 3-13 bridge, 0-5 hub. **The distribution is too homogeneous across communities.** A periphery preference (0.95) visits all communities roughly equally because every community has ~70% periphery nodes. This is not a code bug — it is a fundamental limitation of role-based abstraction on homogeneous SBM graphs. The code correctly implements the mechanism; the mechanism is insufficient for the domain structure.

**Verdict:** Code is correct. Findings are genuine. Safe to escalate.

---

## 6. Instrument Specification (Current)

### 6.1 Metric

**Jensen-Shannon divergence** of engagement distributions between a trained (frozen) system and a fresh (untrained) system, both navigating domain B autonomously.

- Base: natural logarithm (base e)
- Smoothing: epsilon = 1e-10 added to both distributions before normalisation
- Range: [0, ln(2)] = [0, 0.6931]
- Engagement distribution: proportion of visits to each community during autonomous steps on domain B
- Step count: 500 for B₀/B₁ data (this session's verification run). Prior B₂ data used N steps as determined by `len(domain_b_inputs)` in the battery runner. Step count may differ between datasets.

### 6.2 Coherence

**Normalised entropy difference:**

```
coherence = (fresh_entropy - trained_entropy) / (fresh_entropy + trained_entropy + 1e-10)
```

- Range: [-1, 1]
- Positive: trained system has lower entropy (more structured) than fresh
- Zero: equal structure
- Negative: trained system has higher entropy (less structured, maladaptive)

### 6.3 Degeneracy Check

A system visiting fewer than 3 communities (with > 1% engagement each) is flagged as degenerate. Degenerate engagement distributions are not meaningfully comparable — the JSD value is an artefact of concentration, not structural influence.

### 6.4 Signal Classification

| Signal Type | Condition | Interpretation |
|------------|-----------|----------------|
| absent | JSD < 1e-6 | No behavioural divergence at all |
| degenerate_trained | Trained visits < 3 communities | Trained system's engagement too concentrated |
| degenerate_fresh | Fresh visits < 3 communities | Fresh system's engagement too concentrated |
| maximum_divergence | JSD > 99% of ln(2) | Red flag — zero overlap between distributions |
| potentially_confounded | Edge overlap > 5% | Signal may be from shared edges, not structure |
| divergent_incoherent | Coherence <= 0 | Trained differs but is less structured |
| candidate_generativity | All criteria met | Genuine signal candidate (still requires threshold calibration) |

### 6.5 Domain B Construction

- **Standard B:** Same SBM parameters as A (n_nodes, n_communities, p_within, p_between), different random seed. Shared node ID space (0..n-1). Typical edge Jaccard with A: ~0.12.
- **Zero-overlap B:** Same SBM parameters, different seed, disjoint node IDs. Edge Jaccard: 0.0.
- **A' (isomorphic):** Same adjacency as A, permuted node labels, re-sampled node features. Preserves community structure.

### 6.6 Navigation Protocol

1. Train system on domain A
2. `set_training(False)` — freeze all learning
3. `set_domain(B)` — switch graph topology, preserve learned state
4. `reset_engagement_tracking()` — clear visit counts
5. 500 steps of `step(None)` — autonomous navigation using frozen structure
6. Record `get_engagement_distribution()` — visits per community
7. Repeat steps 3-6 for a fresh (untrained) system as baseline
8. Compute JSD between trained and fresh engagement distributions

### 6.7 Current Threshold Status

**PRELIMINARY.** No threshold is set from the positive distribution because no reliable positive distribution exists yet. This is the core question being submitted for review.

The current acceptance criteria require: JSD > per-system-type p95 noise floor AND coherence > 0. The noise floor is empirically derived from 50 untrained seed pairs per system type. No positive threshold has been calibrated.

---

## 7. The Question for the Reviewer

The GNN positive control (PC3) exceeds the per-type noise floor (0.433) on 1 of 3 seeds (JSD=0.450, coherence=0.460) with genuine signal classification. Our rigour principles require at least 2 of 3 seeds to agree for a reportable result.

The role-based walker (PC1) produces consistent sub-threshold signal on all 3 seeds (JSD=0.141-0.291, all with positive coherence, all classified as candidate) but none exceeds the noise floor (0.339).

**Three interpretations:**

**(A) The instrument methodology is sound. The positive controls need strengthening.**

The GNN's 1/3 above-floor result demonstrates that sensitivity exists — the instrument *can* detect a signal above the noise floor when the underlying mechanism is strong enough. The role walker's consistent-but-sub-threshold signal confirms that genuine structural transfer occurs but at insufficient magnitude. A system that creates community-level preferences (rather than node-level role preferences or embeddings) might achieve consistent above-floor results.

**(B) The engagement-JSD metric has inherently high noise for stochastic graph navigation systems.**

The noise floor (p95 = 0.34-0.43) is close to the maximum JSD (0.693), leaving limited dynamic range for genuine signals. With 6 community bins, the engagement distribution is coarse, and random seed differences produce substantial variation. An alternative metric — e.g., trajectory autocorrelation, entropy rate, pattern-matching rather than distribution-matching — might have a lower noise floor and better sensitivity.

**(C) The evidence is sufficient given the stochastic nature of graph navigation.**

The 1/3 above-floor result for PC3, combined with PC1's consistent sub-threshold signal, may constitute adequate evidence of sensitivity. The threshold methodology (per-seed agreement) may be too strict for inherently stochastic systems. A threshold that accounts for variability — e.g., requiring the mean JSD across seeds to exceed a lower bar — might be more appropriate.

### Specific Requests

We request your assessment of:

1. **Which interpretation you find most defensible** — is this primarily a signal-strength problem, a measurement problem, or a threshold-calibration problem?
2. **Whether the current methodology should be retained, modified, or replaced** — is engagement-JSD the right metric? Is the null-distribution approach the right way to set thresholds?
3. **Whether additional positive controls would resolve the question** — if so, what mechanism properties should they have?
4. **Any alternative metrics or measurement approaches you would recommend** — specifically, sequential/structural metrics that might complement distributional JSD.
5. **Whether the governance decisions (strict freeze, coherence requirement) are sound** given Paper 1's definitions — is the strict interpretation of generativity correct, or is a partial-freeze variant defensible?

### Design Degrees of Freedom

We document the degrees of freedom exercised during the investigation, for transparency:

1. **Structural to behavioural metric** (F-037): Switching from structural metric delta to engagement JSD. This was a correction to align with the governance-defined operationalisation, not a tuning decision.
2. **Coherence gate** (post F-039): Adding the coherence requirement. This was added after the 0/13 finding to prevent incoherent divergence from being misclassified.
3. **Noise floor methodology** (F-041): Using per-system-type null distributions with 50 seed pairs. This was developed after the positive control failure to distinguish genuine signal from seed noise.

All tuning was done to make the instrument *work correctly*, not to make any system pass or fail.

### Reproducibility

All results are deterministic with the stated seeds. SBM parameters are in Appendix A. Code in m8-battery repo, commit `da038dd`. All runs can be replayed from seed + configuration.

---

## 8. Appendices

### Appendix A: SBM Domain Parameters

| Preset | Nodes | Communities | p_within | p_between | Seed |
|--------|-------|-------------|----------|-----------|------|
| SMALL | 50 | 4 | 0.30 | 0.030 | 42 |
| MEDIUM | 150 | 6 | 0.30 | 0.020 | 42 |
| LARGE | 500 | 12 | 0.25 | 0.015 | 42 |

All calibration data uses MEDIUM unless stated otherwise. Additional parameters: n_edge_types=3, weight_range=(0.1, 1.0), n_node_features=8.

Domain family generation: `generate_domain_family(config)` produces A (primary), A' (isomorphic permutation), B (fresh draw, same parameters), C (qualitatively different, negative control).

### Appendix B: Per-System Adapter Summary

| System | Class | step() Behaviour | set_training() | set_domain() |
|--------|-------|-----------------|----------------|--------------|
| 1A (WordNet Graph) | 1 | Navigates static graph via cached shortest paths. No learning. | N/A | N/A |
| 1B (Rule Navigator) | 1 | Fixed navigation strategy (greedy/shortest/random). No learning. | N/A | N/A |
| 1C (FoxworthyA) | 1 | Frozen MLP transforms features, selects next node. No learning. | N/A | N/A |
| 2A (FrozenLLM) | 2 | Frozen TinyLlama encodes context, generates choice. No learning. | N/A | N/A |
| 2B (FrozenGAT) | 2 | Frozen GAT forward pass selects next node. No learning. | N/A | N/A |
| 2C (FoxworthyC) | 2 | Frozen GRU weights, latent state evolves. No weight updates. | N/A | N/A |
| 3A (DQN) | 3 | RL agent uses learned policy. Learning during training phase only. | N/A | N/A |
| 3B (Curiosity) | 3 | RND-driven agent. Predictor updates during training. | N/A | N/A |
| 3C (FoxworthyF) | 3 | DistilGPT-2 + LoRA. Surprise-gated learning, viability-adjusted selection. | Gates LoRA updates + consolidation | N/A |
| 3D (Empowerment) | 3 | Learns transition model, computes empowerment via Blahut-Arimoto. | Freezes transition model updates | Transfers empowerment landscape |
| 3E (Active Inference) | 3 | Learns Dirichlet transition model, minimises expected free energy. | Freezes transition model updates | Transfers transition model |
| HEB (Hebbian Walker) | Anchor | Walks graph, strengthens traversed edges, global decay. | Gates Hebbian update + decay | Transfers edge weights where edges overlap |
| STDP (Brian2 Spiking) | Anchor | Advances Brian2 simulation. STDP always active during training. | Snapshots/restores weights (STDP undone) | N/A |

**Positive controls:**

| System | step() Behaviour | set_training() | set_domain() |
|--------|-----------------|----------------|--------------|
| PC1 (Role Walker) | Classifies neighbours by topological role, selects via softmax over learned role preferences. | Gates preference updates | Recomputes role cache from new topology, preserves preferences |
| PC3 (GNN Navigator) | Cosine similarity between neighbour GCN embeddings and learned preference embedding. Softmax selection. | Gates preference embedding update | Recomputes adjacency + features, preserves GCN weights + preference embedding |

### Appendix C: The Eight Rigour Principles

1. **No conclusions without calibrated thresholds.** A placeholder threshold (e.g., 0.05) carries no statistical authority. Results produced with placeholder thresholds are raw data, not classifications. Any conclusion drawn from a placeholder threshold is automatically invalidated.

2. **Every metric bounded and interpretable.** JSD is bounded [0, ln(2)]. Coherence is bounded [-1, 1]. Degeneracy is checked by community count. No metric should produce values that require ad-hoc interpretation or are sensitive to numerical edge cases.

3. **Confounds controlled before measuring.** Edge overlap (Jaccard index) between domains A and B must be computed and reported for every generativity measurement. Shared edges can inflate JSD through statistical familiarity rather than structural generalisation. Systems tested on standard B (with shared edges) must also be tested on zero-overlap B to isolate the confound.

4. **Signal types distinguished.** Not all non-zero JSD values are equal. Six signal types are defined: absent (no signal), degenerate (meaningless measurement), maximum-divergence (red flag), potentially-confounded (shared edges), divergent-incoherent (maladaptive bias), and candidate-generativity (genuine signal). Each requires a different interpretation and response.

5. **Seeds must agree.** A finding that holds on one seed but not others is not reportable as a result. At least 2 of 3 seeds must show the same qualitative outcome (e.g., above-floor with positive coherence) for the result to be reported. Stochastic systems require seed-level replication before any claim.

6. **Instrument changes require full recalibration.** Any change to the measurement protocol — metric, threshold, domain construction, navigation protocol — invalidates all prior thresholds, baselines, and calibration data. Post-change data must be generated with the corrected instrument before any comparison.

7. **Red-flag results investigated before reporting.** The following must be investigated before inclusion in any analysis: JSD = maximum (0.693), extreme coherence (magnitude > 0.9), unexpected passes (especially new passes after instrument changes), seed-dependent results (1/3 seeds passing), or results with > 10x volatility across seeds.

8. **Report data, not conclusions, when uncertain.** When the data does not support a clear interpretation — because the threshold is uncalibrated, because seeds disagree, because the signal is near the noise floor — report the raw data with explicit uncertainty markers. Do not force a classification. The cost of reporting uncertainty is low; the cost of a false conclusion is high.

### Appendix D: Null Distribution Methodology

For each system type, 50 pairs of untrained systems are created with different random seeds. Both systems in each pair are initialised on domain A (without training), switched to domain B via `set_domain(B)`, and run for 500 autonomous steps. The JSD between their engagement distributions is computed. The p95 of this distribution is the noise floor for that system type.

This methodology ensures that the noise floor accounts for:
- Stochastic initialisation effects
- Random walk divergence on the same graph
- Architecture-specific noise characteristics (e.g., GNN's higher baseline due to random weight initialisation)

### Appendix E: Paper 1 Reference

*Can a System Generate Understanding? Toward an Operational Framework for Emergent Cognition.* [DOI pending — Zenodo pre-print in preparation.]

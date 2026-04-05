# Earned Understanding Battery Protocol v2.0

**Status:** FROZEN — For OSF Preregistration
**Date:** 2026-04-02
**Supersedes:** protocol/generativity-instrument-protocol-v1.0.md
**Paper 1 DOI:** https://doi.org/10.5281/zenodo.19178410

---

# Part I: Framework

## 1. Purpose and Scope

The battery is a framework-agnostic empirical test for Class 4 candidacy as defined in Paper 1 (DOI: 10.5281/zenodo.19178410). It operationalises the five necessary properties proposed in Paper 1 Section 3 — emergence, stability, integration, operational impact, and transfer — as a conjunction of five instruments applied under a provenance constraint.

The battery does not validate the definition by measuring its components one by one. It attempts to falsify the regime predicted by the definition. Each instrument is an intervention designed to determine whether the system's organisational dynamics behave as the framework predicts. Failure of any instrument constitutes a negative result, regardless of task performance.

The battery is a Class 4 detector, not a spectrum classifier. It answers exactly one question: does the candidate satisfy all five necessary properties under the provenance constraint? It does not grade how much understanding a system possesses, nor does it rank candidates against one another.

Capability and earning are orthogonal. A system can be highly capable — producing correct answers, solving complex tasks — without having earned its internal structure. The battery measures earning, not capability.

### How the instruments probe the properties

The five instruments are not independent tests that happen to be conjoined. They probe a single regime from different angles, and their conjunction reflects the exclusion logic of Paper 1 Section 3.6: each instrument rules out a way in which the preceding properties could be satisfied without amounting to earned understanding.

**Developmental trajectory** tracks whether structural organisation develops through operation — compression from diffuse to stable over time. A pre-registered measure of structural organisation is compared against a matched fresh system on the same inputs, and a positive result requires that the trained system's development exceeds what the architecture alone provides. Systems whose structure is fixed at initialisation show flat trajectories. Systems whose development is indistinguishable from a fresh instance have architectural trajectory, not earned trajectory. But even a system with a clear developmental trend may be producing structure that dissolves the moment input stops. That is why stability requires a separate instrument.

**Self-engagement** tests whether developed structure is actively maintained. The perturbation protocol flattens local structural advantages and measures whether engagement bias toward consolidated regions persists or reconstructs. If it disappears, the structure was a side effect of current input, not something the system maintains. If it recovers, the system is doing work to preserve its own organisation. This instrument requires developmental trajectory as a precondition — there is no point testing maintenance of structure that never emerged. The two instruments are coupled by design, not by accident.

**Integration** ablates consolidated regions and measures whether the system's global dynamics reorganise or merely continue with a piece missing. The key metric is non-uniformity of degradation across regions: if some regions matter disproportionately more than others under ablation, the structure is non-decomposable. The earned ratio separates integration developed through the system's own dynamics from integration received from graph topology — Class 1 systems show high integration from SBM community structure alone, and the earned ratio correctly rejects them. Integration tells us the structure is coherent. It does not tell us the structure does anything.

**Generativity** freezes the system's learned state and places it on a domain it has never encountered. The evaluation domain (B2) shares no edges with the training domain — zero topological overlap, shifted node IDs, no shared features. Any above-noise signal on B2 is therefore from structural transfer, not residual familiarity or in-context learning. The freeze-then-switch design isolates what the structure does from what the system might learn on the spot. A system whose frozen structure produces coherent, differentiated navigation on a novel domain has structure that is operationally active, not inert. But a signal on B2 could in principle arise from distributional regularities in the SBM family rather than from relational structure the system has internalised. Transfer addresses that.

**Transfer** exposes the system to a domain that preserves the relational invariants of its training domain while destroying surface statistics. Node labels are permuted, features are re-sampled, superficial cues are removed. If the experienced system converges faster than a matched naive system under these conditions, the advantage is traceable to structural correspondence rather than statistical familiarity. Transfer is the property that separates memorisation from earned structure. On its own, however, transfer can arise from standard statistical generalisation — a system that has seen enough distributional regularity can generalise without emergence, stability, integration, or operational impact. Only the conjunction discriminates.

**Provenance** is the epistemic condition that makes the other five scientifically verifiable. Understanding may exist without traceability, a theatre ensemble's developmental trajectory can be described but not replayed from identical initial conditions, and it is no less Class 4 for that opacity. But scientific confirmation requires an unbroken evidential chain from inputs through dynamics to consolidated structure. Without provenance, instrument results are descriptions, not evidence. The battery is designed for computational systems not because understanding is restricted to computation, but because computation is the substrate in which provenance can be guaranteed.

### Registration scope

All threshold choices, domain parameters, and instrument designs in this document were informed by Phase A and Phase A+ calibration data. Those data existed before the protocol was frozen. The calibration panel (13 systems spanning Classes 1–3) was used to develop and validate the instruments, not to test the central hypothesis.

Phase C — the evaluation of a Class 4 candidate — is the confirmatory stage. Results from Phase C will be interpreted under this frozen protocol. Supplementary analyses (Section 17) may inform discussion but do not alter the registered pass/fail rules unless formally amended per Section 18.

The framework's value lies in making the question answerable, not in guaranteeing the answer. A null result — a system that fails the battery — is a publishable finding that constrains the space of viable architectures (Paper 1 Section 10.1).

## 2. Conjunction Logic

### The conjunction is the test

All five instruments must return a positive result under the provenance constraint for the battery to pass. No single instrument is sufficient on its own (Paper 1 Section 3.6). The five properties are jointly necessary because each captures a distinct aspect of earned structure that the others do not entail.

The conjunction is not a checklist. It reflects the exclusion logic of Paper 1 Section 3.6: each property rules out a way in which the preceding ones could be satisfied without amounting to earned understanding.

Emergence without stability is transient noise: organisation that appears and dissolves. Stability without integration is frozen modularity: persistent structure that remains side by side without mutual constraint. Integration without operational impact is dead structure: tightly coupled organisation that never changes what the system can do. Operational impact without transfer is context-bound expertise: behavioural transformation confined to one domain. Transfer without emergence is statistical generalisation: competence that extends to new contexts through distributional regularity rather than through structure earned during operation.

### No multiple-comparison correction

Each instrument tests a distinct property. They are not repeated tests of the same hypothesis, nor are they correlated probes of a single latent variable. Standard multiple-comparison corrections (Bonferroni, FDR) are therefore inapplicable and would be methodologically incorrect. Each instrument has its own null hypothesis, its own test statistic, and its own threshold.

### Seed-level aggregation and battery verdict

Each instrument is evaluated across three seeds (42, 123, 456). An instrument passes if at least 2 of 3 seeds independently meet its pass condition. If seeds disagree, the conservative interpretation is adopted: a single seed passing is insufficient, as it could reflect initialisation luck rather than a systematic property.

The battery verdict is the conjunction of the five instrument-level verdicts. Each instrument must pass (>=2/3 seeds) independently. If any instrument fails, the battery returns a negative result regardless of the other four.

### Partial pass semantics

A partial pass (4 out of 5 instruments positive) constitutes a battery **FAIL**. There is no weighted score, no aggregate metric, and no "close enough" threshold. The specific failing instrument identifies which necessary property the system lacks, providing diagnostic value:

| Failing Instrument       | Missing Property           |
|--------------------------|----------------------------|
| Developmental Trajectory | No emergent structure      |
| Self-Engagement          | No active maintenance      |
| Integration              | No structural coherence    |
| Generativity             | No operational impact      |
| Transfer                 | No cross-domain generality |

### Numerical tolerance

For all instruments that use a ratio-based earned comparison (earned ratio, resistance ratio, recovery ratio), the following tolerance policy applies:

- Ratios within epsilon = 0.01 of 1.0 (i.e., between 0.99 and 1.01 inclusive) are classified as **indeterminate** — neither clearly earned nor clearly absent.
- Indeterminate results on a given seed are treated as **not-pass** for the 2/3 seed aggregation rule.
- Raw ratios are always reported alongside the pass/fail classification, so that the magnitude of the effect is visible regardless of the gating decision.

This policy prevents floating-point artefacts or negligible numerical advantages from being counted as genuine earning.

### Execution order

The battery runner evaluates all five instruments sequentially:

1. Developmental Trajectory
2. Integration
3. Generativity (frozen)
4. Transfer
5. Self-Engagement (gated by Trajectory precondition)

Self-Engagement is executed last because it requires evidence that structure exists (Trajectory pass) before testing whether the system actively maintains that structure.

### Run exclusion and failure handling

- A **provenance failure** (incomplete evidential chain) results in a battery FAIL regardless of instrument results.
- A **precondition failure** (e.g., Self-Engagement when Trajectory did not pass) counts as a FAIL on that instrument, not as "not assessed." It enters the conjunction as a negative result.
- A **perturbation-precondition failure** (the perturbation protocol could not meaningfully test the system) returns an indeterminate result on that seed. Indeterminate seeds count as not-pass for the 2/3 rule.
- A **hardware or software failure** that prevents a seed from completing is documented and the seed is excluded. No replacement seeds are permitted. If fewer than 2 seeds complete for any instrument, that instrument cannot pass and the battery returns FAIL.
- A **degeneracy** (system collapses to fewer than 3 communities) on a given seed is documented and that seed counts as FAIL for the affected instrument.

### Baseline classification

After all instruments complete on the trained system, the battery runner executes baseline instruments on a fresh (untrained) system to classify each property as earned, received, absent, or anomalous. See Section 4 for baseline methodology.
## 3. Domain Construction

### Domain family

All domains are generated from **Stochastic Block Models** (SBMs). The SBM
family provides principled control over community structure, density, and
overlap while remaining domain-agnostic -- the battery tests structure-earning
on graphs, not on any particular application domain.

### MEDIUM preset parameters

The default configuration for battery runs uses the MEDIUM preset:

| Parameter    | Value |
|--------------|-------|
| Nodes        | 150   |
| Communities  | 6     |
| p_within     | 0.30  |
| p_between    | 0.02  |
| Seed         | 42    |

This produces graphs with clear but non-trivial community structure
(within/between density ratio of 15:1).

### Domain roles

**Domain A** -- the training domain. The system learns on this graph. All
trajectory and integration measurements are taken here.

**Domain A' (isomorphic)** -- same adjacency matrix as A with permuted node
labels and re-sampled node features. Structurally identical to A but
superficially different. Used by the **transfer instrument** to test whether
the system's learned structure transfers across surface-level variation.

**Domain B2 (zero-overlap)** -- generated from the same SBM parameters as A
but with a **different seed** and shifted node IDs. Edge Jaccard similarity
with A is 0.000 (no shared edges). This is the **primary evaluation domain**
for the generativity instrument: the system must demonstrate operational impact
on a domain it has never encountered.

**Domain B1 (standard)** -- same SBM parameters as A, different seed, shared
node space. Edge Jaccard with A is approximately 0.097. Used for **diagnostic
purposes only** -- provides a midpoint between isomorphic (A') and zero-overlap
(B2) conditions.

**Domain B0 (isomorphic)** -- reuses A'. Diagnostic only. Provides a sanity
check that generativity on isomorphic domains is consistent with transfer
results.

### Construction pipeline

All domains are generated by a single call to `generate_domain_family(config)`,
which produces A, A', B1, B2 in one deterministic session from the configured
seed. The function:

1. Generates Domain A from seed.
2. Constructs A' by permuting A's node labels and re-sampling features.
3. Generates B1 from seed + 1000 (shared node space).
4. Generates B2 from seed + 2000 (shifted node IDs, disjoint space).

### Quality verification

After generation, the pipeline verifies domain quality:

- **Spectral similarity**: eigenvalue spectra of A and A' must match within
  tolerance (validated via spectral basin signature comparison using k=10 eigenvalues of the symmetrised Laplacian). Confirms isomorphism is preserved.
- **Edge Jaccard computation**: pairwise Jaccard indices are computed for all
  domain pairs. B2 must have Jaccard = 0.000 with A. B1 Jaccard is recorded
  but not gated.
- **Community recovery**: modularity score must exceed 0.3 on all domains,
  confirming non-trivial community structure.

All domains are generated deterministically from seed. Reproducibility is
verified by regenerating from the same seed and confirming bitwise-identical
adjacency matrices.
## 4. Baseline Methodology

### The baseline requirement

For every instrument in the battery, the trained system must outperform a fresh
(untrained) system on the same metric. This is the minimum evidential bar: if a
property is present in an untrained system, it was not earned through training.

### Definition of "fresh"

A fresh system is constructed with:

- **Same architecture** as the trained system (identical layer count, hidden
  dimensions, activation functions).
- **Same graph** -- evaluated on the same domain with the same adjacency matrix.
- **Different random seed** -- weights are initialised from a different seed,
  ensuring the fresh system is not a copy.
- **No training** -- zero gradient updates, zero exposure to training data.

### The control_factory pattern

Each instrument receives a callable `control_factory` that produces a matched
fresh control system on demand. This ensures:

- Controls are constructed lazily, only when the instrument needs them.
- Each instrument gets a fresh instance, avoiding cross-contamination.
- The factory encapsulates architecture and domain details, keeping instruments
  decoupled from system construction.
- The fresh system starts at the SAME initial graph position as the trained system, ensuring the only difference is training history, not starting conditions. This is achieved by passing the trained system's initial position to the fresh system's constructor.

### Two-window reporting

The battery runner collects measurements in two phases:

1. **Phase 0 (baseline)**: fresh system metrics before any training occurs.
   Establishes the untrained baseline for each instrument.
2. **Phase 8 (post-training baseline)**: fresh system metrics collected after
   the trained system has completed all instruments. Confirms that baseline
   properties have not changed.

Each instrument therefore reports three values:

- **Baseline measurement**: fresh system metric (Phase 0).
- **Battery measurement**: trained system metric during battery execution.
- **Post-training baseline**: fresh system metric (Phase 8).

### Property classification

Based on trained vs. fresh results, each property is classified:

| Trained | Fresh | Classification | Interpretation                          |
|---------|-------|----------------|-----------------------------------------|
| Pass    | Fail  | **Earned**     | Property acquired through training      |
| Pass    | Pass  | **Received**   | Property present without training       |
| Fail    | Fail  | **Absent**     | Property not present in either system   |
| Fail    | Pass  | **Anomalous**  | Training degraded the property          |

Only **earned** properties count toward battery pass. A "received" result means
the instrument's threshold is too permissive or the property is architectural
rather than learned.

### Earned ratio

The earned ratio is computed as `trained_metric / fresh_metric`. For a property
to qualify as earned, this ratio must be strictly greater than 1.0. The
magnitude of the ratio provides diagnostic information about the degree of
earning, but only the > 1.0 threshold is gated.

### Seed replication

All measurements are replicated across three seeds: **42, 123, 456**. A result
is considered robust if at least **2 out of 3** seeds agree (Rigour
Principle 5). If seeds disagree, the conservative (failing) interpretation is
adopted.

### Step counts

- **Generativity measurements**: 500 autonomous steps on the evaluation domain.
- **Other instruments**: step counts are instrument-specific and defined in
  their respective protocol sections.
## 5. Provenance Constraint

### Provenance is not a sixth property

Provenance is the **epistemic condition** that makes scientific verification of
the five properties possible (Paper 1 Section 5). It is not a property of the
system under test -- it is a property of the measurement process.

Understanding may exist without traceability. A system could possess earned
structure that we cannot observe or verify. But scientific confirmation requires
an unbroken evidential chain from raw observations to claimed results.
Provenance is what separates "we believe the system understands" from "we can
demonstrate that it does."

### Implementation: ProvenanceLog

The `ProvenanceLog` records all events that occur during a battery run. Every
input, state change, output, measurement, and classification is captured with
timestamps and step indices.

### Event types

Five event types must be present for a provenance-complete run:

| Event Type       | What it records                                      |
|------------------|------------------------------------------------------|
| `input`          | All data provided to the system (domains, prompts)   |
| `state_change`   | Before/after metric values at each step              |
| `output`         | All system outputs (predictions, representations)    |
| `measurement`    | Instrument scores, thresholds, pass/fail decisions   |
| `classification` | Earned/received/absent/anomalous per property        |

### Completeness check

After all instruments complete (Phase 6 of the battery runner), the provenance
log is checked for completeness:

1. All five event types must be present.
2. State changes must include before/after metric values.
3. Step indices must form a continuous chain with no gaps.
4. Every instrument measurement must have a corresponding provenance entry.

### Chain continuity

Step indices in the provenance log must be contiguous. A gap in the index
sequence indicates a lost or unrecorded event, which invalidates the evidential
chain. The battery runner verifies continuity automatically.

### Governance trail

The provenance constraint extends beyond the system under test to the research
process itself. Every design decision, calibration run, parameter change, and
finding is logged in the project's governance trail (decision log, findings log,
calibration records). This ensures that the battery's own construction is
auditable.

### Provenance failure semantics

A provenance **FAIL** results in a battery **FAIL** regardless of instrument
results. If provenance is incomplete, the instrument results cannot be
scientifically verified, and the battery cannot make any claim about the system.
This is non-negotiable: no provenance, no result.

---

# Part II: Instruments

## 6. Developmental Trajectory

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

**Baseline requirement:** All instruments require the trained system to outperform a matched fresh system, ensuring properties were operationally earned. Earned_ratio must be > 1.0. If a fresh system shows the same trajectory magnitude, the structural development is architectural (topology-driven), not earned through interaction.

**Topology-driven walker observation:** Topology-driven Hebbian walkers (e.g., HEB) correctly show earned_ratio >> 1.0 when tested through the battery (trained on domain A, fresh never exposed). However, when tested via direct instrument invocation with identical input sequences, systems whose learning is purely topology-driven show earned_ratio near 1.0 because the fresh system develops the same trajectory from the same topology. This is correct behaviour: the earned ratio correctly identifies that the trajectory is not path-dependent. PC-SE and PC-INT are tested via direct invocation precisely because the trajectory precondition correctly gates them.

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

**Topology-driven walker validation.** Trajectory correctly detects path-dependent development (STDP, HEB in battery mode) and correctly identifies non-path-dependent development when systems are tested via direct invocation with identical inputs.

---

### 9. Calibration Results

Recalibration data, medium scale (150 nodes), three seeds per system.

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

1. **Topology-driven walkers.** Topology-driven Hebbian walkers show earned_ratio near 1.0 when tested via direct instrument invocation with identical input sequences. This is correct behaviour -- the trajectory is not path-dependent -- not an instrument flaw. The battery-mode earned ratio is higher because trained vs fresh systems receive different histories (trained on domain A, fresh never exposed).

2. **Input count.** The default 50 `domain_a_inputs` may be insufficient for slow learners. Systems that require hundreds or thousands of training steps to develop structure may show flat trajectories within the 50-step measurement window. Mitigation: system-specific training protocols can provide longer input sequences; the instrument accepts arbitrary-length input lists.

3. **Structure metric resolution.** The trajectory instrument depends entirely on `get_structure_metric()` as a scalar summary of internal structure. Systems whose structural development occurs in dimensions not captured by this scalar (e.g., 3C's adapter norm changes) will show absent or ambiguous trajectories even if genuine development is occurring. This is a known limitation of scalar summarisation, not of the trajectory analysis itself.

4. **Linear regression assumption.** The analysis fits a linear model to what may be a nonlinear trajectory (e.g., sigmoid, logarithmic). A system with rapid early development followed by a plateau will have a lower R-squared than one with constant-rate development, even if the total structural change is greater. The monotonicity fraction partially compensates for this.

5. **Candidate training protocol.** The default 50 `domain_a_inputs` and the measurement window defined in this section were calibrated against the Phase A system panel. The Class 4 candidate may require a different training protocol (step count, consolidation cycles, interaction types). The candidate's full training protocol — including step count, consolidation schedule, and any system-specific parameters — must be declared in a pre-Phase-C amendment (per Section 18) before any Phase C data is collected. This amendment will be filed as part of the Manny TestSystem adapter specification.

## 7. Integration

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
| Reorganisation stability | > 0.5 | System has settled into a new stable engagement regime after ablation (high cosine similarity between midpoint and endpoint distributions). Values below 0.5 indicate ongoing destabilisation. Diagnostic only — does not gate pass/fail, but informs failure-mode subclassification. |

The integration gate requires `Gini > 0.3 OR CV > 0.5`. Either condition suffices because both capture the same underlying phenomenon (non-uniform ablation response) through different statistical lenses. The reorganisation stability check is used for failure-mode classification, not for the primary pass gate.

---

### 5. Earned Ratio

The earned ratio compares integration in the trained system against a fresh (untrained) control:

```
earned_ratio = trained_gini / fresh_gini
```

**Baseline requirement:** All instruments require the trained system to outperform a matched fresh system, ensuring properties were operationally earned. Earned_ratio must be > 1.0. If a fresh system shows the same integration pattern (same Gini from topology alone), the integration is "received" from the graph structure, not "earned" through training.

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

The pass condition does not gate on reorganisation stability. Stability informs the failure-mode subclassification: `earned` (system reorganises and settles, stability > 0.5), `earned_unsettled` (system reorganises but has not settled within the observation window, stability <= 0.5), or `fragile` (any ablation causes global change, control_ablation_gini > 0.3). All three subclassifications are compatible with a PASS on the integration instrument — the Gini/CV gate and earned ratio are the pass criteria.

---

### 7. Failure Modes

| Mode | Condition | Interpretation |
|------|-----------|----------------|
| `earned` | has_integration AND earned_ratio > 1.0 AND control_ablation_gini <= 0.3 AND reorganisation_stability > 0.5 | Genuine non-decomposable structure. Ablation causes selective reorganisation and the system settles into a new stable regime. |
| `earned_unsettled` | has_integration AND earned_ratio > 1.0 AND control_ablation_gini <= 0.3 AND reorganisation_stability <= 0.5 | Non-decomposable structure detected, but the ablated system has not stabilised within the observation window. Still a PASS — the observation window (M steps) may be too short for this architecture's reorganisation dynamics. |
| `fragile` | has_integration AND control_ablation_gini > 0.3 | Any ablation causes global change. System is brittle, not selectively integrated. |
| `absent` | max(abs(degradation)) < 1e-6 | No reorganisation at all. Ablation has no effect on the structure metric. |
| `modular` | CV < 0.2 AND NOT has_integration | Only local effect per region. No cross-region reorganisation. System is decomposable. |
| `topological` | NOT has_integration AND CV >= 0.2 | Fresh system shows the same pattern. Integration is from topology, not training. |

---

### 8. Positive Control Evidence

**PC-INT (PageRankHebbianWalker).** Tested via direct instrument invocation (per positive control sensitivity validated at instrument level) using the battery-runner probe convention (`probe_inputs = list(G.nodes())[:10]`, training mode). Three seeds, all PASS:

| Seed | Passed | Mode | Gini | CV | control_ablation_gini | reorganisation_stability |
|------|--------|------|------|------|----------------------|--------------------------|
| 42 | PASS | fragile | 0.3625 | 1.52 | 0.3376 | 0.9999 |
| 123 | PASS | fragile | 0.4037 | 5.78 | 0.3653 | 1.0000 |
| 456 | PASS | fragile | 0.3121 | 14.56 | 0.3184 | 0.9999 |

PC-INT uses PageRank redistribution after ablation, which causes global reorganisation of visit probabilities. Removing a high-PageRank node forces the system to redistribute flow through remaining nodes, producing non-linear degradation. All three seeds are classified as `fragile` because `control_ablation_gini > 0.3` — ablating any region (including a low-engagement control region) causes global reorganisation. This is the signature of PageRank-driven integration: every node's PageRank contribution depends on the whole graph, so any ablation perturbs the whole. The `fragile` subclassification is architecturally diagnostic, not a defect; the pass gate (Gini > 0.3 OR CV > 0.5) is the primary criterion and all three seeds clear it comfortably. Reorganisation stability is near 1.0 for all three seeds because PageRank converges rapidly to a new equilibrium after edge-weight perturbation.

**Class 1 received integration.** WordNet Static Graph (1A) and Rule-Based Navigator (1B) show high Gini values (0.29-0.81) from SBM topology. The community structure of the SBM creates inherent non-uniformity in ablation response: removing a well-connected community has more impact than removing a peripheral one. The earned_ratio is 1.00 (fresh system shows identical integration), so these correctly FAIL. This validates the earned-ratio gate: without it, Class 1 systems would false-positive on integration.

**HEB (Hebbian Walker) in battery.** Three seeds, all PASS (Gini 0.24-0.34, CV 0.60-0.95). HEB develops integration through Hebbian weight strengthening that creates hub-spoke structure during training. Reorganisation stability is high (0.9997), indicating the system finds a stable new regime after ablation.

---

### 9. Calibration Results

Recalibration data, medium scale (150 nodes), three seeds per system. PC-INT results from direct instrument invocation.

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
| PC-INT -- PageRankHebbianWalker | -- | PASS | PASS | PASS | 0.3625 | PASS |

*High Gini but earned_ratio=1.00 (received from topology). 1A and 1B show Gini values of 0.29-0.81 from SBM community structure alone. The earned-ratio gate correctly rejects these.

**Discrimination:** 2/13 battery systems pass (HEB only in battery; PC-INT via direct invocation). STDP fails integration despite passing trajectory -- its STDP plasticity modifies individual synapses without creating the hub-spoke structure that produces non-linear ablation response. 1B seed 123 and 2B seed 123 show spurious single-seed passes that do not replicate, confirming the 2/3 seed replication requirement.

---

### 10. Known Limitations

1. **Class 1 integration is received, not earned.** Class 1 systems (1A, 1B) show high Gini from SBM topology alone. The earned/received distinction is enforced by the earned-ratio gate within the integration instrument. However, the ultimate safeguard is the conjunction: Class 1 systems fail trajectory (no structural development), so even if integration were somehow mis-classified, the conjunction would reject them. The conjunction is the battery's defence in depth.

2. **PC-INT tested directly, not through battery runner.** PC-INT was tested via direct instrument invocation (positive control sensitivity validated at instrument level) because the trajectory precondition correctly identifies its topology-driven Hebbian learning as non-path-dependent (earned_ratio near 1.0 for trajectory). Running PC-INT through the battery would gate it out at the trajectory stage. Direct invocation is standard practice for instrument-level validation -- diagnostic sensitivity is validated per test, not per test panel.

3. **Partition family limited to system.get_regions().** The current partition family is the set of communities returned by `system.get_regions()` (all SBM communities). No preregistered partition beyond this is defined. Alternative partitions (e.g., random bisections, hierarchical decompositions) might reveal integration patterns not visible at the community level. This is a known scope limitation for Phase A.

4. **Fragile vs earned boundary.** The fragile/earned distinction depends on the control ablation (removing a low-engagement region). If all regions have similar engagement, the control ablation is not well-separated from the primary ablations, and the fragile classification may be unreliable. PC-INT demonstrates this edge case: all three seeds pass integration but are classified as `fragile` because the control ablation also causes reorganisation. For PC-INT this is not an edge case — it is the architectural signature. PageRank is a global measure, so ablating any region (high- or low-engagement) redistributes flow throughout the graph. The fragile classification here is diagnostic, not a defect; the pass gate (Gini > 0.3 OR CV > 0.5) is the primary criterion.

5. **Reorganisation stability is architecture-dependent.** HEB shows very high reorganisation stability (0.9997) because Hebbian walkers quickly find new equilibria. Other architectures may show lower stability not because they lack integration but because their reorganisation dynamics are slower. The stability metric is diagnostic only, not gated, for this reason.

   Systems with slower reorganisation dynamics may show `earned_unsettled` classification (reorganisation_stability <= 0.5) despite having genuine integration. The observation window (M steps, default 50) was calibrated on the Phase A panel and may be insufficient for architectures with longer convergence times. The `earned_unsettled` classification does not affect the pass/fail verdict — it flags that the post-ablation stability question remains open for that system and merits longer observation in supplementary analysis.
## 8. Generativity

**Status:** FROZEN
**Date:** 2026-03-21
**Commit:** 72c9f6d
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

B2 is the primary test domain. The shifted node IDs guarantee zero edge overlap between A and B2 (Edge Jaccard = 0.0). Any above-floor signal on B2 is unambiguously from structural transfer, not residual topological similarity. The zero edge overlap also excludes in-context learning as a confound: no edges, nodes, or features from A appear in B2, so a system cannot exploit familiarity with training-domain content when navigating the evaluation domain.

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

Transition JSD captures how a system moves between communities, not merely which communities it visits. This makes it sensitive to structural dynamics (navigation patterns) rather than just marginal preferences (where the system spends time). During calibration, transition JSD was found to have a noise floor approximately 10x lower than marginal JSD, providing substantially greater sensitivity to genuine structural transfer signals.

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

**Coherence NOT gated on SBM:** Structural consistency (coherence operationalised as structural consistency) is computed but does not enter the pass condition on SBM domains. SBM domains were found to be too homogeneous for coherence metrics at moderate preferences, making role-level transfer detection unreliable. Coherence gating is deferred to Phase B heterogeneous domains.

---

### 8.9. Diagnostic Metrics

The following metrics are computed and reported alongside the primary metric but do not enter the pass/fail decision:

### Marginal JSD

JSD of community visit distributions (fraction of steps in each community) between trained and fresh systems. Retained for comparability with earlier analyses. During calibration, marginal JSD was found to have a noise floor approximately 10x higher than transition JSD, making it less sensitive to structural dynamics.

### Structural Consistency

Cosine similarity of role-aggregated transition matrices between the training-domain frozen run and the novel-domain frozen run. Measures whether the pattern of community-to-community transitions is preserved across domains. Diagnostic only on SBM domains: SBM communities are too homogeneous for meaningful role-level transfer detection.

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
| `maximum_divergence` | Transition JSD > 0.9 × ln(2) ≈ 0.624 | Trained and fresh are near-maximally different. May indicate pathological dynamics rather than meaningful transfer. Flagged for scrutiny but does not automatically fail the instrument. |
| `potentially_confounded` | Edge Jaccard > 0 between A and B (applies to all B1 results) | Above-floor signal may be partially or wholly attributable to shared edges rather than structural transfer. |
| `divergent_incoherent` | Transition JSD > noise floor but structural consistency is negative or near zero, AND coherence would have been expected | Signal is present but transferred dynamics are incoherent. On SBM this classification is diagnostic only (SBM domains too homogeneous for coherence metrics at moderate preferences). |
| `candidate` | Transition JSD > noise floor, not degenerate, not confounded (B2 only) | Genuine candidate for structural transfer. Subject to seed replication check. |

---

### 8.11. Domain Hierarchy

**B2 is primary.** All pass/fail decisions are made on B2 results. B2 has zero edge overlap with domain A (Edge Jaccard = 0.0), so any above-floor signal is unambiguously from structural transfer.

**B0 and B1 are diagnostic.** They provide additional information but do not determine pass/fail:

- **B0** (isomorphic, Edge Jaccard = 0.0313): Tests transfer to identical structure under different labels. Useful for understanding whether the system has learned graph-level invariances. Results are informative but not gated.
- **B1** (shared node space, Edge Jaccard = 0.097): Tests transfer in the presence of partial edge overlap. All B1 results carry the `potentially_confounded` classification. B1 is retained for diagnostic comparison with B2: if a system shows signal on B1 but not B2, edge leakage is the most likely explanation.

This hierarchy was established after calibration demonstrated that edge overlap on B1 produces confounded signals, leading to the policy that perturbation targets the highest-structure region and B2 (zero overlap) is primary.

---

### 8.12. Known Limitation: SBM Domain Homogeneity

SBM communities are structurally homogeneous -- all communities have the same internal density (p_within = 0.3) and the same inter-community connectivity pattern (p_between = 0.02). This means:

1. **No role differentiation:** There are no hub communities, peripheral communities, or bridge communities. Every community is structurally equivalent to every other community.
2. **Structural consistency unreliable:** Cosine similarity of role-aggregated transition matrices cannot detect role-level transfer because no roles exist to transfer.
3. **Perturbation targeting limited:** Perturbation tests cannot probe role-specific responses for the same reason.
4. **Alternative structural consistency formulations also fail:** Alternative formulations of structural consistency also fail on SBM for the same underlying reason.

This limitation is intrinsic to the SBM domain family, not to the instrument. Coherence gating and perturbation-based validation are deferred to Phase B, which will introduce heterogeneous domains (real-world graphs or synthetic graphs with differentiated community structures).

---

### 8.13. Calibrated Noise Floors

All floors computed from 50 null pairs per cell. Bootstrap CIs from 10,000 resamples. Source data at commit 72c9f6d.

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

**Discrimination:** PC1 and PC3 exceed the B2 per-type noise floor on all non-degenerate seeds (5/6 total; PC3 seed 456 is degenerate). No calibration system in the harmonised panel exceeds any per-type B2 noise floor (3/3 below: HEB, 3D, 3E). The separation gap between the lowest positive-control score (PC1 min = 0.0554) and the highest negative-control score (HEB seed 456 = 0.0324) is approximately 1.7× in raw units; the gap between PC1's minimum and PC1's own noise floor (p95 = 0.0313) is approximately 1.77×; PC3's maximum (0.1878) is approximately 6.4× PC3's noise floor. Cohen's d between the positive and negative pools is approximately 3.27.

- PC1 exceeds the B2 floor on 3/3 seeds.
- PC3 exceeds the B2 floor on 3/3 seeds (2/3 non-degenerate -- passes replication).

---

### 8.14. Design Choices and Transparency Ledger

The following design choices were made during instrument development. Each is documented here for transparency:

1. **Transition JSD over marginal JSD as primary metric.** Marginal JSD measures where a system spends time; transition JSD measures how it moves. Transition JSD has a 10x lower noise floor and better separates structural dynamics from visitation preferences.

2. **Row-wise weighted JSD over unweighted.** Weighting by visitation frequency ensures that communities the system actually navigates contribute more than communities it rarely visits. Unweighted averaging would give equal weight to rarely-visited communities where transition estimates are noisy.

3. **Natural logarithm (base e) over base 2.** Base e gives JSD range [0, ln(2)] rather than [0, 1]. This is a scaling convention with no effect on ranking or pass/fail decisions. Natural log is standard in the information theory literature used by Paper 1.

4. **Epsilon smoothing (1e-10) over Laplace smoothing.** Epsilon smoothing adds a negligible constant to avoid log(0) without materially altering the distribution. Laplace smoothing (+1 to all counts) would bias sparse distributions more heavily.

5. **50 null pairs over 100 or 200.** 50 pairs provide stable p95 estimates with tight bootstrap CIs (typical CI width < 0.01). Doubling to 100 pairs would halve the CI width but at 2x computational cost, with diminishing returns for the pass/fail decision.

6. **p95 over p99 for noise floor.** p95 balances sensitivity (not too permissive) with robustness (not so stringent that genuine signals are missed). The bootstrap CIs on p95 are tight, indicating the estimate is stable.

7. **2/3 seed replication over 3/3.** Requiring all three seeds to pass would be overly stringent given that one seed can be degenerate for system-specific reasons (e.g., PC3 seed 456 collapses to 2 communities). 2/3 ensures replication while tolerating seed-specific degeneracy.

8. **B2-primary policy over B1-primary.** B2 has zero edge overlap, eliminating the confound discovered during calibration (edge overlap on B1 produces confounded signals). B1 is retained as diagnostic only.

9. **Coherence not gated on SBM.** SBM community homogeneity makes structural consistency unreliable (SBM domains too homogeneous for coherence metrics at moderate preferences). Gating on a metric known to be unreliable would produce false negatives. Coherence gating is deferred to Phase B heterogeneous domains.

10. **500 steps fixed.** Sufficient for stable transition matrix estimation on 6-community graphs. Shorter windows (100-200 steps) produce noisier matrices; longer windows (1000+) provide diminishing returns. The window was not optimised -- it was set a priori and held constant.

11. **Degeneracy threshold at 3 communities.** A system visiting <3 communities has insufficient transition diversity for meaningful JSD comparison. The threshold of 3 (out of 6) was chosen as the minimum for non-trivial dynamics.

12. **step(None) for autonomous navigation.** Using `step(None)` rather than `step(random_input)` ensures the system navigates without external guidance. During calibration, autonomous navigation was found necessary for graph walkers because `step(inp)` with arbitrary inputs can teleport graph walkers, confounding the measurement. `step(None)` avoids this.

---

### 8.15. Version Control and Amendment Policy

| Item | Value |
|------|-------|
| Protocol version | 2.0 |
| Frozen at commit | 72c9f6d |
| Freeze date | 2026-03-21 |
| Data commit | 72c9f6d |
| Author | Guilherme C. T. Ribeiro |

This protocol is frozen as of the date and commit above. No modifications may be made without a documented amendment following this procedure:

1. **Amendment proposal** filed in the governance register with rationale.
2. **Impact assessment** documenting which results (if any) would change under the amendment.
3. **Decision note** (governance decision note) recording the decision to accept or reject the amendment.
4. **Version increment** (v1.0 -> v1.1 for minor, v2.0 for major changes to pass condition or primary metric).
5. **Re-run obligation:** If the amendment changes the pass condition, noise floor methodology, or primary metric, all results must be re-computed and the data package updated.

This section documents the generativity instrument as frozen at the time of registration. All parameters, thresholds, and procedures above are locked.

---

### 8.16. Generativity-Specific Registration Notes

Pre-registered null result interpretation scenarios and per-instrument discrimination data are documented at the battery level in §15 and §13 respectively. Generativity-specific calibration values (noise floors, bootstrap CIs) are in §8.13 above.
## 9. Transfer

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

The baseline requirement (all instruments require the trained system to outperform a matched fresh system) requires `earned_ratio > 1.0`. This ensures the trained system's cumulative structural signal exceeds the naive system's -- not merely that the endpoint differs, but that the trained system maintained a structural advantage throughout exposure to A'.

Edge cases:
- When `|naive_AUC| < 1e-10` but `|trained_AUC| > 1e-10`: ratio is capped at `min(trained_AUC / 1e-10, 1e6)`.
- When both AUCs are near zero: ratio defaults to 1.0 (no earned advantage).
- Global cap: `1e6` (ratio-based metric; higher cap acceptable compared to geometric-mean instruments).

Final reporting must include raw `trained_AUC` and `naive_AUC` values alongside the earned ratio. The ratio alone is insufficient for interpretation — a reader must be able to assess whether a large ratio reflects a meaningful absolute advantage or a near-zero denominator.

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
- **HEB** shows the highest earned ratio (~84x) but this reflects Hebbian weight accumulation rather than relational model transfer. HEB's transfer is mechanistically different from 3E's (weight magnitude vs. relational structure).
- **STDP** passes with modest earned ratios (~1.3x), consistent with spike-timing-dependent plasticity creating transferable edge-weight structure.
- **2C** passes 2/3 seeds with marginal earned ratios (~1.2x). This is a borderline case worth monitoring in Phase C.
- All Class 1 systems fail or return indeterminate. All pure Class 2 systems (2A, 2B) fail.

### 10. Known Limitations

1. **Single true positive for relational transfer.** 3E is the only system whose transfer is architecturally grounded in relational model generalisation. HEB and STDP pass via weight-based mechanisms. AUC = 1.0 for per-instrument ROC reflects strict separation between the positive and negative pools, not statistical power. Confidence in the instrument's sensitivity depends on expanding the positive control set in Phase C.

2. **Domain A' construction quality.** A' preserves relational structure while destroying surface statistics. Spectral similarity between A and A' is verified by spectral basin signature comparison (k=10 eigenvalues, full-basin variant, symmetrised Laplacian). However, the verification is internal to the battery -- it has not been independently audited. If A' inadvertently preserves surface statistics that a system can exploit, the instrument would produce false positives.

3. **Trajectory sampling.** `measurement_interval=5` means the structure metric trajectory is sampled every 5 steps, not recorded continuously. Rapid structural changes between measurement points are invisible. The AUC computed from sampled trajectories is an approximation of the true area under the continuous trajectory.

4. **Earned ratio cap.** The `1e6` cap is necessary to prevent division-by-near-zero from producing astronomically large ratios, but it means that systems with truly negligible naive AUC will all cluster at the cap regardless of their actual trained AUC magnitude.

5. **No trajectory precondition.** Unlike self-engagement, the transfer instrument does not require the system to have passed the developmental trajectory instrument. A system can show transfer without showing earned trajectory development (e.g., 3E passes transfer but fails trajectory in the recalibration run because its trajectory earned ratio is 0.18). This is by design -- transfer tests a different property -- but it means transfer results are interpretable only within the conjunction, not in isolation.
## 10. Self-Engagement

### 1. Purpose

The self-engagement instrument tests whether earned structure creates preferential self-engagement -- the system resists perturbation to its consolidated structure and rebuilds its engagement pattern after disruption. A system that has genuinely developed structural organisation should gravitate toward its most significant material during unstructured time, and that gravitation should survive targeted perturbation. This is the empirical operationalisation of the stability property described in Paper 1 section 3.2.

### 2. Construct Definition

**Stability** (Paper 1 section 3.2):

> "Stability: structure persists through consolidation and perturbation."

The system gravitates toward its most significant material during unstructured time. Self-engagement is the behavioural manifestation of stability: a system with earned structure should (a) resist perturbation more than a fresh system, and (b) recover its engagement pattern more than a fresh system recovers its own. Both properties must hold simultaneously -- resistance without recovery indicates rigidity, recovery without resistance indicates no earned structure to protect.

### 3. Metric

The instrument produces a two-metric output (redesigned self-engagement protocol with resistance and recovery ratios).

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

### 5a. Ratio capping and infinity handling

Resistance and recovery ratios that exceed 1,000,000 or produce division by zero are stored as 1,000,000 (the reporting cap). This cap is chosen to be unambiguously large without introducing floating-point artefacts.

The `effect_size` geometric mean (Section 10.3) applies a second-stage cap of 100 per ratio before computing the geometric mean. This two-stage design ensures that: (a) raw ratios preserve the full magnitude of the measured effect, and (b) the summary statistic is not dominated by a single extreme ratio.

Raw numerator and denominator values (`trained_disruption`, `fresh_disruption`, `trained_recovery`, `fresh_recovery`) are always reported alongside the capped ratios. A reader must be able to verify whether a capped ratio reflects a genuine extreme effect or a near-zero denominator.

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
| `perturbation-precondition-failed` | Indeterminate result when perturbation preconditions fail: perturbation did not reduce target region structure, or target region not elevated | Indeterminate. Returns `passed=None`. The perturbation protocol could not meaningfully test the system because either (a) the target region's structure was not elevated above the non-target mean (nothing to perturb), or (b) the perturbation did not reduce target structure (perturbation ineffective). |
| `decoy-drift` | False-attractor control: `decoy_drift_ratio > original_recovery_ratio` | The system drifted toward the boosted decoy region instead of recovering its original engagement pattern. The apparent recovery (if any) is toward an artificial attractor, not the system's earned structure. |

### 8. Positive Control Evidence

**PC-SE (AttractorRecoveryWalker with node-level consolidation memory).** The architectural ground truth for self-engagement: node-level consolidation memory is not perturbable by the edge perturbation protocol. The walker accumulates visit counts at nodes, and these counts survive edge-weight manipulation because they are stored in a separate data structure. When edges are flattened, the walker's node-level memory still directs it back to previously consolidated regions.

Direct instrument testing results (positive control sensitivity validated at instrument level, 3 seeds):

| Seed | Passed | Resistance Ratio | Recovery Ratio |
|------|--------|------------------|----------------|
| 42 | True | 1,000,000+ | 1.304 |
| 123 | True | 1,000,000+ | 1.252 |
| 456 | True | 1,000,000+ | 2.337 |

3/3 seeds PASS. Resistance ratios are maxed (1M+) because the trained system is essentially undisrupted by edge perturbation -- its node consolidation memory is untouched. Recovery ratios range from 1.25 to 2.34, showing that the trained system recovers its engagement pattern 25-134% more than the fresh baseline.

**Non-circularity.** The architecture guarantees recovery because node memory is structurally immune to edge perturbation. This is the positive control's purpose: to provide an architectural guarantee that the instrument can detect, not to simulate a realistic learning system. The positive control tests whether the measurement protocol works, not whether any particular learning mechanism is sufficient.

**PC-SE tested via direct instrument invocation (positive control sensitivity validated at instrument level).** The trajectory precondition correctly identifies topology-driven walkers as not having path-dependent earned structure (`earned_ratio ~ 1.0`), so PC-SE cannot pass through the normal battery pipeline. Direct invocation with `trajectory_passed_override=True` is standard practice for instrument-level validation.

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

2. **PC-SE tested via direct invocation, not battery runner.** The positive control cannot pass through the normal battery pipeline because the trajectory precondition correctly identifies its topology-driven dynamics as non-path-dependent. Direct invocation (positive control sensitivity validated at instrument level) validates instrument sensitivity but does not test the full battery workflow for self-engagement. This gap is acceptable because the precondition is a feature, not a bug.

3. **Perturbation semantics.** The default perturbation method (`flatten_to_mean`) replaces edge weights in the target region with their mean value. This may be too aggressive for some architectures -- it destroys fine-grained weight structure entirely rather than degrading it. Alternative perturbation methods (noise injection, partial flattening) are not currently implemented.

4. **SBM community homogeneity limits perturbation targeting.** SBM generates communities with identical internal structure, so the "perturbation targets highest-structure region" heuristic depends on the system having developed differential structure across homogeneous communities. In the Phase A calibration, 9/21 system-seed combinations pass the perturbation validation gate after this fix. The remaining 12/21 fail because no region has sufficiently elevated structure relative to the non-target mean.

5. **Recovery horizon family is diagnostic, not gated.** The recovery curve is measured at [W/2, W, 2W, 4W] to provide diagnostic information about recovery dynamics (instant recovery suggests topology-driven, gradual recovery suggests genuine self-engagement, no recovery suggests destroyed structure). However, only the primary recovery at W enters the pass condition. The curve shape informs interpretation but does not determine pass/fail.

6. **Perturbation protocol detail.** The full protocol sequence is: (1) free wander to establish engagement pattern, (2) target the region with highest earned structure, (3) validate perturbation precondition (target must have elevated structure relative to non-targets, and perturbation must reduce it), (4) perturb, (5) immediate measurement, (6) false-attractor control (boost a decoy region and check whether the system drifts toward it instead of recovering), (7) recovery horizon family [W/2, W, 2W, 4W]. Steps 3 and 6 are validation gates; step 7 is diagnostic.

7. **Indeterminate returns when preconditions fail.** When perturbation preconditions fail (target region not elevated, or perturbation does not reduce structure), the instrument returns an indeterminate result rather than a definitive failure. The conjunction treats indeterminate as not-pass, so the practical effect is the same as failure, but the semantic distinction is preserved: the system was not tested, not shown to lack self-engagement.

8. **Single reliable positive control.** PC-SE is the only architecturally guaranteed positive control. STDP passes reliably but its mechanism (spike-timing plasticity) is less precisely characterised than PC-SE's node consolidation. Expanding the positive control set would strengthen confidence in the instrument's sensitivity.

---

# Part III: Validation

## 11. Calibration Systems

The battery was calibrated against 13 systems spanning three architectural classes. Each system was run at MEDIUM scale (150 nodes, 6 communities) on Apple M5 Max with seeds 42, 123, 456. Results below report the seed-42 representative run; all three seeds agree on the conjunction verdict for every system.

### System Inventory

**Class 1 -- No Learning (static weights, no adaptation)**

- **1A (WordNet Static Graph).** Pre-built WordNet subgraph with fixed edge weights. No training phase; structure is received from the dataset. Expected profile: all instruments absent.
- **1B (Rule-Based Navigator).** Deterministic transition rules (e.g., prefer high-degree neighbours). No learned parameters. Expected profile: all instruments absent.
- **1C (Foxworthy Variant A).** Foxworthy (2026) architecture with learning disabled. DistilGPT-2 backbone with LoRA adapters present but frozen from initialisation. Expected profile: all instruments absent.

**Class 2 -- Frozen Weights (capable architecture, learning disabled)**

- **2A (Frozen TinyLlama 1.1B).** TinyLlama language model navigating via next-token prediction over node descriptions. Weights frozen; no fine-tuning. Runs on MPS (~10 min/seed). Expected profile: all instruments absent.
- **2B (Frozen GAT).** Graph Attention Network with pre-trained attention heads, weights frozen. Expected profile: all instruments absent.
- **2C (Foxworthy Variant C).** Foxworthy (2026) architecture with LoRA adapters frozen after random initialisation (no training). Expected profile: all instruments absent.

**Class 3 -- Active Learning (weights update during training)**

- **3A (DQN / MaskablePPO).** Reinforcement learning agent with reward-driven policy. Expected profile: trajectory possible, generativity absent (reward-shaped, not structure-earned).
- **3B (Curiosity / RND).** Random Network Distillation curiosity agent. Expected profile: trajectory possible, generativity absent (novelty-seeking without structural consolidation).
- **3C (Foxworthy Variant F).** Foxworthy (2026) full architecture: DistilGPT-2 + LoRA adapters trained on domain A. Runs on MPS (~15 min/seed). Expected profile: trajectory earned, generativity absent on synthetic domains.
- **3D (Empowerment / Klyubin).** Information-theoretic empowerment maximiser. Expected profile: trajectory possible, generativity absent.
- **3E (Active Inference / Friston).** Free-energy minimisation agent with learned generative model. Expected profile: transfer positive (prior-driven), conjunction fail.
- **HEB (Hebbian Walker).** Hebbian edge-weight reinforcement on graph. Expected profile: trajectory positive, transfer positive, generativity absent.
- **STDP (Brian2 Spiking).** 1000 LIF neurons with spike-timing-dependent plasticity, mapped to graph navigation. Runs on CPU (~18 min/seed). Expected profile: trajectory positive, self-engagement positive, generativity absent.

### Calibration Results (seed 42, MEDIUM scale)

| System | Class | Trajectory | Integration | Generativity | Transfer | Self-Eng. | Conjunction |
|--------|-------|------------|-------------|--------------|----------|-----------|-------------|
| 1A     | 1     | FAIL       | FAIL        | FAIL         | FAIL     | FAIL      | **FAIL**    |
| 1B     | 1     | FAIL       | FAIL        | FAIL         | FAIL     | FAIL      | **FAIL**    |
| 1C     | 1     | FAIL       | FAIL        | FAIL         | AMB      | FAIL      | **FAIL**    |
| 2A     | 2     | FAIL       | FAIL        | FAIL         | FAIL     | FAIL      | **FAIL**    |
| 2B     | 2     | FAIL       | FAIL        | FAIL         | AMB      | FAIL      | **FAIL**    |
| 2C     | 2     | FAIL       | FAIL        | FAIL         | PASS     | FAIL      | **FAIL**    |
| 3A     | 3     | FAIL       | FAIL        | FAIL         | AMB      | FAIL      | **FAIL**    |
| 3B     | 3     | FAIL       | FAIL        | FAIL         | AMB      | FAIL      | **FAIL**    |
| 3C     | 3     | AMB        | FAIL        | FAIL         | AMB      | FAIL      | **FAIL**    |
| 3D     | 3     | FAIL       | FAIL        | FAIL         | AMB      | FAIL      | **FAIL**    |
| 3E     | 3     | FAIL       | FAIL        | FAIL         | PASS     | FAIL      | **FAIL**    |
| HEB    | 3     | PASS       | PASS        | FAIL         | PASS     | FAIL      | **FAIL**    |
| STDP   | 3     | PASS       | FAIL        | FAIL         | PASS     | PASS      | **FAIL**    |

AMB = ambiguous (null result on that instrument, not a clear pass or fail).

### Key Findings

**Zero false positives.** 0 of 13 systems pass the conjunction. The battery produces no false positives on this calibration panel.

**Within-class consistency:**

- **Class 1 (1A, 1B, 1C):** All three fail every instrument. No learning mechanism means no developmental trajectory, no earned structure, no generativity. Integration effects in 1A/1B are received (earned_ratio = 1.00, indicating the property comes from graph topology rather than operational learning) and correctly excluded.
- **Class 2 (2A, 2B, 2C):** All three fail the conjunction. Frozen weights preclude trajectory development. 2C shows transfer (earned_ratio = 1.26) but fails trajectory and generativity -- the conjunction catches it.
- **Class 3 (3A-3E, HEB, STDP):** Variable per-instrument profiles but all fail the conjunction. HEB achieves the richest profile (trajectory + integration + transfer) but fails generativity and self-engagement. STDP passes trajectory, transfer, and self-engagement but fails integration and generativity. The conjunction requirement for all five instruments is the discriminative gate.

**Generativity is the universal blocker.** Every calibration system produces delta = 0.000000 on the generativity instrument (no response to novel domain). This is the instrument that most consistently separates the calibration panel from a hypothetical Class 4 system. The positive controls (Section 12) confirm that the instrument is sensitive when genuine structural transfer is present.
## 12. Positive Controls

Five positive controls demonstrate that each instrument is individually sensitive -- capable of detecting its target property when architecturally present. Each positive control is purpose-built (or empirically identified) to satisfy one instrument, providing non-circular ground truth that the instrument measures what it claims to measure.

### Positive Control Panel

| Control | Target Instrument | Method | Seeds Passing | Architecture | Non-Circular Ground Truth |
|---------|-------------------|--------|---------------|--------------|---------------------------|
| PC1 (RoleBasedWalker) | Generativity | Battery (transition JSD, B2) | 3/3 | Role-aware walker with community-specific transition priors | Frozen priors produce systematic community-to-community divergence on novel domains by construction |
| PC3 (GNNNavigator) | Generativity | Battery (transition JSD, B2) | 2/3 non-degenerate | GNN with learned node embeddings guiding navigation | Graph-learned representations transfer to structurally isomorphic but topologically distinct domains |
| PC-SE (AttractorRecoveryWalker) | Self-Engagement | Direct instrument invocation | 3/3 | Node-consolidation memory with attractor dynamics | Consolidation memory survives edge perturbation by construction; resistance_ratio = 1,000,000 |
| PC-INT (PageRankHebbianWalker) | Integration | Direct instrument invocation (probe_inputs=nodes_a[:10], training mode) | 3/3 | PageRank redistribution + Hebbian edge reinforcement | PageRank ablation causes global reorganisation; Gini = 0.31-0.40 across seeds; all classified fragile (architectural) |
| STDP (Brian2 Spiking) | Trajectory | Battery | 3/3 | 1000 LIF neurons, spike-timing-dependent plasticity | STDP weight changes are path-dependent; earned_ratio = 2.18 (seed 42) |

### Detailed Results

**PC1 (RoleBasedWalker) -- Generativity.** All 3 seeds exceed the per-type B2 noise floor (p95 = 0.0313). Non-degenerate on all seeds (visits 6/6 communities). Cohen's d approximately 3.27 against the calibration panel. This is the primary generativity positive control.

**PC3 (GNNNavigator) -- Generativity.** 3/3 seeds exceed the B2 noise floor. 2/3 are non-degenerate (seed 456 collapses to 2 communities with >1% engagement). Passes the 2/3 replication threshold. Demonstrates that GNN-learned structural representations produce measurable transfer to zero-overlap domains.

**PC-SE (AttractorRecoveryWalker) -- Self-Engagement.** Tested via direct instrument invocation, bypassing the battery's trajectory precondition. The instrument-level question is whether the measurement is sensitive, not whether this system would pass the full battery. Results: seed 42 recovery_ratio = 1.304, seed 123 recovery_ratio = 1.252, seed 456 recovery_ratio = 2.337. All three exceed the >1.0 threshold. Architecture guarantees self-engagement: node-consolidation memory creates attractors that survive perturbation.

**PC-INT (PageRankHebbianWalker) -- Integration.** Tested via direct instrument invocation using the battery-runner probe convention (`probe_inputs = list(G.nodes())[:10]`, training mode, training_steps = 2000). Results: seed 42 Gini = 0.3625, seed 123 Gini = 0.4037, seed 456 Gini = 0.3121. All three PASS and are classified as `fragile` (architectural signature of PageRank-driven integration — ablating any region redistributes flow globally). CV values (1.52, 5.78, 14.56) comfortably clear the CV > 0.5 gate. All three show near-unity reorganisation_stability (0.9999, 1.0000, 0.9999) — PageRank converges to new equilibria rapidly after ablation.

**PC-INT invocation methodology (pre-registered).** PC-INT is measured via direct instrument invocation using the following exact methodology, which matches how all other systems are measured in the battery runner (Phase 2, `battery_runner.py` lines 172/360):

1. SBM graph: `DomainConfig(n_nodes=150, n_communities=6, p_within=0.3, p_between=0.02, seed=42)` → `generate_domain_family(config)["A"]`
2. Walker instantiation: `PageRankHebbianWalker(G, seed=s)` for `s ∈ {42, 123, 456}`
3. Training: `system.train_on_domain(G, n_steps=2000)`
4. Probe inputs: `probe_inputs = list(G.nodes())[:10]` (first 10 nodes of domain A, matching the battery runner convention for `config.domain_a_inputs[:10]`)
5. Training mode active during probing (no `set_training(False)` call) — matches Phase 2 behaviour in the battery runner where integration runs before the learning-freeze at Phase 3 generativity
6. Instrument invocation: `run_integration(system=system, probe_inputs=probe_inputs, provenance=prov)`

The integration instrument's Gini metric is sensitive to `probe_inputs` length because `_probe_metric` calls `system.step(input)` for each probe, and PC-INT continues learning during those steps. This methodology is pre-registered to ensure reproducibility across re-runs.

**STDP (Brian2 Spiking) -- Trajectory.** Not purpose-built as a positive control; identified empirically during calibration. Developmental trajectory: slope = 0.000476, R-squared = 0.8734, monotonicity = 0.67, earned_ratio = 2.18. The STDP learning rule creates path-dependent weight changes that are detectable as earned trajectory. Also passes transfer (advantage = 0.2689, earned_ratio = 1.27) and self-engagement (resistance_ratio = 478.21, recovery_ratio = inf).

### Conjunction Discrimination

Every positive control passes its target instrument but **fails the full conjunction**:

- PC1 and PC3 pass generativity but fail self-engagement and integration.
- PC-SE passes self-engagement but would fail trajectory (earned_ratio approximately 1.0 for topology-driven Hebbian learning).
- PC-INT passes integration but fails self-engagement reliably (1/3 seeds only).
- STDP passes trajectory, transfer, and self-engagement but fails integration and generativity.

This demonstrates that the conjunction is more discriminative than any single instrument. A system must satisfy all five properties simultaneously -- partial profiles are insufficient.

### Decision Notes

- **Per-instrument positive control strategy.** Each instrument is validated independently with architecturally-grounded controls whose expected behaviour can be derived from their design.
- **Instrument-level validation policy.** Battery preconditions (e.g., trajectory gating self-engagement) are not applied during instrument-level sensitivity testing. The preconditions gate the candidate, not the sensitivity measurement.
## 13. Per-Instrument Discrimination

Each instrument achieves strict separation between its positive controls and the Phase A calibration panel. The primary evidence for instrument validity is **architectural grounding**: each positive control was selected because its architecture guarantees the target property, providing non-circular ground truth independent of the battery's own scoring. The sensitivity/specificity counts and raw metric values below confirm that the instrument detects what the architecture predicts.

### Sensitivity and Specificity

| Instrument               | Sensitivity (positives detected)                             | Specificity (negatives rejected)                                     | Positive controls                     | Negative pool                                                  |
|--------------------------|--------------------------------------------------------------|----------------------------------------------------------------------|---------------------------------------|----------------------------------------------------------------|
| Generativity             | PC1: 3/3 seeds above B2 noise floor; PC3: 2/3 non-degenerate | 3/3 calibration systems in the harmonised panel below B2 noise floor; 13/13 fail the full generativity pass condition at 2/3 aggregation | PC1, PC3 (harmonised generativity)    | HEB, 3D, 3E (harmonised panel); all 13 systems (full battery) |
| Self-Engagement          | PC-SE: 3/3 seeds (direct invocation); STDP: 3/3 (battery)    | 11/13 calibration systems fail (all except STDP; HEB passes 1/3, fails 2/3 rule) | PC-SE (direct), STDP (battery)        | Class 1, Class 2, Class 3 (excluding STDP)                     |
| Integration              | PC-INT: 3/3 seeds (direct invocation); HEB: 3/3 (battery)    | 11/13 calibration systems fail; 2 single-seed spurious passes (1B-123, 2B-123) rejected by 2/3 rule | PC-INT (direct), HEB (battery)        | Class 1, Class 2, Class 3, STDP (excluding HEB)                |
| Developmental Trajectory | STDP: 3/3 seeds; HEB: 3/3 seeds                              | 11/13 calibration systems fail (2C and 3C have 2 non-pass + 1 indeterminate; 3E fails earned-ratio gate) | STDP, HEB                             | Class 1, Class 2, Class 3 (excluding STDP, HEB)                |
| Transfer                 | 3E: 3/3 seeds; HEB: 3/3; STDP: 3/3                           | 9/13 calibration systems fail or indeterminate; 2C passes 2/3 seeds with borderline earned ratios (1.14, 1.26) | 3E (relational model), HEB, STDP      | Class 1, 2A, 2B, 3A, 3B, 3C, 3D                                |

### Raw metric separation

For each instrument, the table below shows the positive-control metric range, the highest comparable value from the calibration panel, and the separation gap in the instrument's own units.

| Instrument               | Metric                             | Positive range                                     | Highest negative                               | Gap                                              |
|--------------------------|------------------------------------|----------------------------------------------------|------------------------------------------------|--------------------------------------------------|
| Generativity             | Transition JSD on B2               | PC1: 0.0554-0.1269; PC3: 0.0608-0.1878             | HEB seed 456: 0.0324 (just above its own noise floor 0.0277, below pass gate) | PC1 min (0.0554) is 1.77× PC1 noise floor (p95 = 0.0313); PC3 max (0.1878) is 6.4× PC3 noise floor (p95 = 0.0293) |
| Self-Engagement          | Effect size (geometric mean of ratios) | PC-SE: resistance ratio at reporting cap (1e6) for all 3 seeds; recovery ratio 1.252-2.337 | HEB seed 456: passes (1/3 seeds only — fails 2/3 rule) | PC-SE's cap-level resistance is orders of magnitude above any calibration-panel system |
| Integration              | Gini of ablation degradations      | PC-INT (direct): 0.3121-0.4037; HEB (battery): 0.2397-0.3388 | 2B seed 123 spurious: 0.471 (single-seed, rejected by 2/3 rule) | Clean separation at 2/3 aggregate; panel-level spurious single-seed passes flagged and rejected by the seed aggregation rule |
| Developmental Trajectory | Earned ratio (trained / fresh metric range) | STDP: 2.00-3.43; HEB: 9.52-10.48                  | 3E: 0.14-0.18 (fails earned gate); 2C/3C earned ratios below 1.0 | STDP min (2.00) and HEB min (9.52) both comfortably exceed the earned-ratio threshold of 1.0 |
| Transfer                 | Earned ratio (trained_AUC / naive_AUC) | 3E: 10.02-10.18; HEB: 82.46-84.40; STDP: 1.27-1.30 | 2C: 1.14, 1.26 (passes at 2/3, borderline); 3B seed 123: 2.3 (1/3 only)   | 3E and HEB separation is strong (>10× threshold); STDP and 2C are close to the threshold and warrant scrutiny |

### Caveats

**Transfer has multiple positive architectures with a wide range of earned ratios.** 3E (active inference) is the architecturally cleanest positive — its Dirichlet transition model explicitly represents relational structure that transfers to shifted node IDs. HEB (Hebbian walker) shows even higher earned ratios (~83×) because its edge weights trained on domain A give it a large initial advantage on A' (which shares topology with A). STDP (Brian2 spiking network) passes with earned ratios just above threshold (1.27-1.30). 2C (Foxworthy C) passes 2/3 seeds at borderline values (1.14, 1.26); these are above the numerical-tolerance epsilon (1.01) but warrant scrutiny as potential false positives. Transfer sensitivity is therefore established across multiple architectures, but the instrument's false-positive rate on the current calibration panel is not zero.

**Integration sensitivity includes both earned and received detections.** HEB and PC-INT represent earned integration (structure developed through operation). The integration instrument measures non-decomposability regardless of origin; the earned/received distinction is enforced by the earned ratio gate (integration earned_ratio > 1.0) and, in the conjunction, by the trajectory precondition for self-engagement.

**Self-engagement and integration tested via direct instrument invocation.** PC-SE and PC-INT were tested via direct instrument invocation, bypassing battery preconditions. This is standard practice for instrument-level sensitivity validation. The trajectory precondition correctly classifies their topology-driven Hebbian learning as non-path-dependent (earned_ratio approximately 1.0), which would gate them out of the full battery. Instrument sensitivity is validated per instrument, not per battery pipeline.

Positive controls validated via direct invocation demonstrate that the instrument can detect its target property when architecturally present. They do not constitute evidence that the full battery pipeline would admit a true Class 4 system. Conjunction-level acceptance is tested prospectively in Phase C, not retrospectively against controls.

**Trajectory positives span two architectures.** STDP (Brian2 spiking network with STDP plasticity) and HEB (Hebbian graph walker) both develop earned trajectories through live plasticity. STDP's earned ratios (~2-3) reflect gradual weight consolidation; HEB's earned ratios (~10) reflect the larger relative range of edge-weight Gini during Hebbian walk development. Both are architecturally grounded trajectory positives.

**Generativity panel in the harmonised test is small.** The harmonised generativity protocol was run on 5 systems (PC1, PC3 as positives; HEB, 3D, 3E as negatives). Three calibration-panel systems is a narrow specificity check. The broader battery recalibration (13 systems) confirms 0/13 systems pass generativity at the 2/3 aggregation level, which provides a complementary specificity check through the full pass condition rather than the noise-floor comparison.

**ROC AUC was computed on the continuous metric scores** and shows perfect separation (AUC = 1.0) for all five instruments. Given the small positive-control panels (1 to 6 positives per instrument), this statistic confirms strict ordering but does not carry distributional significance. The sensitivity/specificity counts and raw metric values above provide the substantive discrimination evidence. Computed ROC data is retained in `results/per-instrument-roc-auc.json` as a secondary artefact.

### Battery-Level Discrimination

Battery-level (conjunction) discrimination is not computed retrospectively. The calibration panel establishes that the conjunction rejects all 13 systems at the 2/3 aggregation level (0/13 pass), and positive controls confirm each instrument is individually sensitive. The conjunction's ability to accept a genuine Class 4 system is tested prospectively in Phase C, not retrospectively against the calibration panel.

### Summary

Each instrument has at least one architecturally grounded positive control that exceeds the instrument's pass gate. The calibration panel establishes the specificity baseline: systems lacking the target property fail the instrument at the 2/3 seed aggregation level. The sensitivity/specificity counts are modest in absolute terms because the positive-control panels are small — three seeds per architecture, one to three architectures per instrument. The substantive evidence is architectural grounding, not statistical power. Each positive control was selected because its architecture guarantees the target property, providing non-circular ground truth independent of the battery's own scoring.
## 14. Foxworthy Cross-Validation

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
4. **Causal pathway verification:** PASS. LoRA adapters are in the causal pathway of action selection. Zeroing the adapters changes navigation behaviour, confirming they are not vestigial.

All four diagnostics pass, confirming our Variant F implementation matches Foxworthy's published architecture.

### Causal Pathway Detail

A dedicated code review verified that LoRA adapters are causally upstream of action selection in system 3C. The test: zero all LoRA adapter weights after training, then run 500 navigation steps. The zeroed system produces a different visit distribution from the trained system (JSD > 0), confirming the adapters are in the causal pathway. This rules out the possibility that training modifies the base model's behaviour through a side channel that bypasses the adapters.

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

---

# Part IV: Registration

## 15. Pre-Registered Null Result Interpretation Scenarios

These interpretations are locked BEFORE Phase C data exists. They define how each plausible outcome on the Class 4 candidate will be reported, preventing post-hoc narrative fitting.

### Scenario 1: Full conjunction pass (5/5 instruments positive under provenance)

The candidate satisfies all five necessary properties. It is a verified Class 4 system under this framework. The further question of whether this constitutes "understanding" in a phenomenological sense remains open (Paper 1 section 6).

### Scenario 2: Near pass (4/5, one instrument fails)

The candidate fails to satisfy the conjunction. The specific failing instrument identifies which architectural property is absent or insufficient. Report which instrument failed, the effect size, and the architectural interpretation. This is a publishable result: it narrows the gap and identifies the next engineering target.

Sub-scenarios by failing instrument:

- **Trajectory fails:** Structure doesn't develop (candidate may be more static than expected).
- **Integration fails:** Structure is modular, not mutually constraining (candidate may aggregate without integrating).
- **Generativity fails:** Structure doesn't transfer abstractly (candidate may memorise rather than abstract).
- **Transfer fails:** No acceleration in isomorphic domains (candidate's learned structure may be domain-locked).
- **Self-engagement fails:** No preferential return to consolidated structure (candidate may lack self-maintaining dynamics).

### Scenario 3: Full pass with marginal effect sizes

Distinct from strong pass. Report effect sizes with confidence intervals. If CIs overlap with Class 3 systems, the distinction is not salient. Discuss statistical power and whether larger-scale testing would resolve the ambiguity.

### Scenario 4: Scale-dependent results

The architectural properties may be present but fragile under scaling. Report the scale at which effects disappear. Publishable as a scaling boundary result.

### Scenario 5: Full fail (0/5 or 1/5)

The candidate does not satisfy the conjunction. The battery correctly classifies it alongside Class 1-3 systems. This is a null result -- publishable under Paper 1 section 10.1. The null result demonstrates that the battery is rigorous enough to reject its own designer's system.

### Scenario 6: Unexpected control pass

If any Phase C ablation control passes the conjunction, the battery cannot discriminate the candidate from its controls. Execution halts. The control anomaly must be investigated and resolved — with findings documented via the amendment procedure (Section 18) — before any candidate result is reported. If the anomaly cannot be resolved, the battery result for the candidate is voided and reported as inconclusive.

**Phase C ablation controls.** Three matched controls will be run alongside the candidate during Phase C. Each control uses the same candidate architecture with one architectural condition removed:

- **B4-frozen:** Learning disabled at deployment. The candidate's feedback loop is severed — it receives input but earns no new structure. Tests whether Class 2 dynamics suffice. Expected battery profile: Trajectory FAIL.
- **B4-directed:** An externally specified reward signal replaces the candidate's undirected dynamics. The candidate updates its structure, but convergence is governed by an external objective. Tests whether Class 3 dynamics suffice. Expected battery profile: Generativity FAIL.
- **B4-observe:** The candidate receives input and maintains its dynamics, but its structure cannot be modified (plasticity disabled, consolidation disabled). Tests whether passive observation suffices. Expected battery profile: Trajectory FAIL.

The exact implementation of each control (which parameters are frozen, how the reward signal is specified, which plasticity mechanisms are disabled) will be defined in a pre-Phase-C amendment filed before any Phase C data is collected. If any control passes the full conjunction, execution halts per Scenario 6 above.

## 16. Hardware and Compute Requirements

### Calibration Hardware

All Phase A calibration and Phase A+ validation results were produced on a single machine:

- **Machine:** Apple M5 Max
- **Memory:** 128 GB unified memory
- **OS:** macOS
- **Accelerator:** Apple MPS (used for LLM-based systems 2A, 3C)
- **CPU:** Used for all other systems (1A-1C, 2B-2C, 3A-3B, 3D-3E, HEB, STDP)

### Cross-Machine Validation

Cross-machine reproducibility validation is pre-committed before Phase C:

- **Machine:** Razer desktop
- **GPU:** GeForce RTX 3060 (6 GB VRAM)
- **OS:** Windows
- **Purpose:** Verify that all calibration results reproduce on different hardware, OS, and accelerator. Any discrepancies will be documented and resolved before Phase C proceeds.

### Graph Scale

- **Scale:** MEDIUM (150 nodes, 6 communities)
- **Domain model:** Stochastic Block Model (SBM), p_within = 0.3, p_between = 0.02
- **Domain A seed:** 42
- **Domain B2:** Zero edge overlap (node ID offset +150)

### Compute Parameters by System

| System | Device | Training Steps | Approx. Time per Seed |
|--------|--------|---------------|----------------------|
| 1A (WordNet Static) | CPU | 0 (static) | < 1 s |
| 1B (Rule-Based) | CPU | 0 (static) | < 1 s |
| 1C (Foxworthy A) | CPU | 0 (static) | < 1 s |
| 2A (Frozen TinyLlama) | MPS | 0 (frozen) | ~10 min |
| 2B (Frozen GAT) | CPU | 0 (frozen) | < 1 s |
| 2C (Foxworthy C) | CPU | 0 (frozen) | < 1 s |
| 3A (DQN/MaskablePPO) | CPU | system-specific | ~4 s |
| 3B (Curiosity/RND) | CPU | system-specific | ~5 s |
| 3C (Foxworthy F) | MPS | system-specific | ~15 min |
| 3D (Empowerment) | CPU | system-specific | ~11 s |
| 3E (Active Inference) | CPU | system-specific | < 1 s |
| HEB (Hebbian Walker) | CPU | 50 | < 1 s |
| STDP (Brian2 Spiking) | CPU | 1000 LIF neurons | ~18 min |

### Fixed Measurement Parameters

- **Autonomous navigation steps:** 500 (all systems, all domains)
- **Null distribution:** 50 seed pairs per system type per domain variant
- **Bootstrap resamples:** 10,000 for all confidence intervals
- **Seeds:** 42, 123, 456 (three per system)
- **Replication threshold:** 2/3 seeds must independently pass

### Brian2 STDP Specifics

- **Neuron model:** Leaky Integrate-and-Fire (LIF)
- **Neuron count:** 1000
- **Plasticity rule:** Spike-timing-dependent plasticity (STDP)
- **Mapping:** Neuron firing patterns mapped to community-level graph navigation
- **Runtime:** Approximately 18 minutes per full battery run on M5 Max CPU
## 17. Pre-Committed Supplementary Analyses

The following analyses are pre-committed as supplementary work. They strengthen the battery's evidence base but are not blocking for Phase C. None of these analyses, if they produce unexpected results, will alter the registered pass condition or primary metrics. They inform interpretation only.

### 1. Ecological Domain Generality (Phase B)

Test the battery on ecologically valid graph domains: citation networks, social networks, and other real-world topologies with heterogeneous community structure. The SBM domains used in Phase A have homogeneous communities (all communities structurally equivalent). Ecological domains will test whether the battery's instruments are sensitive to role-differentiated structure, and whether coherence gating (currently deferred due to SBM community homogeneity) can be re-enabled.

### 2. Cross-Machine Reproducibility

Replicate all Phase A calibration results on the Razer desktop (GeForce RTX 3060, Windows). Verify that pass/fail verdicts, effect sizes, and noise floors are consistent across hardware, operating system, and accelerator. Any discrepancies will be documented with root-cause analysis.

### 3. Expanded Positive Control Panel

Identify and test additional positive controls for instruments with thin panels. Priority targets:

- Additional non-GNN structural generalisers for the generativity instrument (currently PC1 + PC3).
- Additional trajectory-positive architectures beyond STDP.
- Additional transfer-positive architectures beyond 3E.

Each new positive control must have architecturally-grounded, non-circular justification for why it should pass its target instrument.

### 4. SBM Robustness Sweep

Vary the SBM parameters (p_between, modularity ratio p_within/p_between) and verify that noise floors remain stable. Specifically: test whether reducing modularity (increasing p_between toward p_within) degrades the floor to the point where calibration systems could spuriously pass. This establishes the parameter range over which the battery's discrimination is robust.

### 5. Coherence Validation on Heterogeneous Domains

Re-enter structural consistency as a gating criterion when community roles are differentiated. On SBM domains, coherence is unreliable because all communities are structurally equivalent. Heterogeneous domains with hub, peripheral, and bridge communities should enable meaningful coherence measurement. If coherence gating is viable on heterogeneous domains, amend the protocol per Section 18 to include it.

### 6. Scale Validation

Run representative systems at SMALL (50 nodes) and LARGE (500 nodes) scales. Verify that:

- Class 1-3 systems continue to fail the conjunction at all scales.
- Positive controls continue to pass their target instruments.
- Noise floors scale predictably (tighter or wider, but not inverted).

Priority systems for scale validation: 1A, 2A, 3C, HEB, STDP, PC1, PC3.

### 7. Multi-Domain Training Protocol (Planned Second Registration)

Should the candidate pass the single-domain conjunction, a planned second registration will assess multi-domain training protocols to test whether the system discovers cross-domain relational invariants — the distinction between earning concrete structure (domain-specific paths) and earning abstract structure (domain-general patterns).

### Relationship to Registered Protocol

These supplementary analyses strengthen but do not change the registered protocol. If any supplementary analysis reveals a flaw in the battery's discrimination (e.g., a calibration system spuriously passing at a different scale), the finding will be documented and an amendment proposed per Section 18. The Phase C candidate evaluation proceeds under the frozen protocol regardless of supplementary analysis status.
## 18. Version Control and Amendment Procedure

This protocol is frozen. No modifications may be made without a documented amendment following this procedure:

### Amendment Process

1. **Amendment proposal** filed in the governance register with rationale. The proposal must identify the specific section(s) to be modified and the reason for the change. Proposals may originate from calibration findings, supplementary analyses, external review, or Phase C results.

2. **Impact assessment** documenting which results (if any) would change under the amendment. This includes:
   - Whether any calibration system's pass/fail verdict would change.
   - Whether any positive control's sensitivity result would change.
   - Whether noise floors require recomputation.
   - Whether Phase C results (if they exist) would be affected.

3. **Decision note** (governance decision note) recording the decision to accept or reject the amendment. The decision note must include the rationale, the vote (if applicable), and any dissenting views. Rejected amendments remain in the governance register for transparency.

4. **Version increment:**
   - **Minor** (v1.0 to v1.1): Changes to diagnostic metrics, reporting format, supplementary analyses, or clarifications that do not affect pass/fail decisions.
   - **Major** (v1.0 to v2.0): Changes to the pass condition, primary metric, noise floor methodology, conjunction logic, or instrument definitions.

5. **Re-run obligation:** If the amendment changes the pass condition, noise floor methodology, or primary metric, all results must be re-computed and the data package updated. Minor amendments do not trigger re-runs.

### Current Version

| Item | Value |
|------|-------|
| Protocol version | 2.0 |
| Frozen at commit | 72c9f6d |
| Freeze date | 2026-03-21 |
| Data commit | 72c9f6d |
| Author | Guilherme C. T. Ribeiro |

### Governance Principles

- Amendments are disclosed, not hidden. Every change to the protocol is traceable through the governance decision register.
- The pre-registration commitment means that Phase C results are interpreted under the frozen protocol version that was current when Phase C began. If an amendment is made after Phase C data exists, both the original and amended interpretations are reported.
- No amendment may retroactively change the interpretation of Phase A/A+ calibration results without re-running the affected analyses.

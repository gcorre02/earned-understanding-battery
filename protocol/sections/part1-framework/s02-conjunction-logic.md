## 2. Conjunction Logic

### The conjunction is the test

All five instruments must return a positive result under the provenance
constraint for the battery to pass. No single instrument is sufficient on its
own (Paper 1 Section 3.6). The five properties are jointly necessary because
each captures a distinct aspect of earned structure that the others do not
entail.

### No multiple-comparison correction

Each instrument tests a **distinct property**. They are not repeated tests of
the same hypothesis, nor are they correlated probes of a single latent
variable. Standard multiple-comparison corrections (Bonferroni, FDR) are
therefore inapplicable and would be methodologically incorrect. Each instrument
has its own null hypothesis, its own test statistic, and its own threshold.

### Partial pass semantics

A partial pass (4 out of 5 instruments positive) constitutes a battery
**FAIL**. There is no weighted score, no aggregate metric, and no "close
enough" threshold. The specific failing instrument identifies which necessary
property the system lacks, providing diagnostic value:

| Failing Instrument | Missing Property          |
|--------------------|---------------------------|
| Trajectory         | No emergent structure      |
| Self-Engagement    | No active maintenance      |
| Integration        | No structural coherence    |
| Generativity       | No operational impact      |
| Transfer           | No cross-domain generality |

### Execution order

The battery runner (`battery_runner.py`) evaluates all five instruments
sequentially in the following order:

1. **Trajectory** -- must pass before Self-Engagement is meaningful
2. **Integration** -- structural coherence of learned representations
3. **Generativity (frozen)** -- operational impact with frozen weights
4. **Transfer** -- generalisation to novel domain A'
5. **Self-Engagement** -- gated by Trajectory precondition

Self-Engagement is executed last because it requires evidence that structure
exists (Trajectory pass) before testing whether the system actively maintains
that structure.

### Baseline classification (Phase 8)

After all instruments complete on the trained system, the battery runner
executes baseline instruments on a fresh (untrained) system to classify each
property as earned, received, absent, or anomalous. See Section 4 for baseline
methodology.

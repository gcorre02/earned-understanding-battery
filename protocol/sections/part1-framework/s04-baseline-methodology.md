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

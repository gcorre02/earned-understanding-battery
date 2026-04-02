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

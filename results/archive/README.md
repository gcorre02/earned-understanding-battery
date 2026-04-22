# `results/archive/` — historical result artefacts

Files in this directory are **not cited in the frozen protocol** and are not
required for reproducing the registered battery. They are retained for
historical record of the investigation phase.

## Contents

### `perturbation_audit.json`

- Produced 2026-03 during investigation of the self-engagement instrument's
  perturbation targeting (pre-DN-31).
- The generating script was a one-off and was never committed to git.
- Superseded by `perturbation_audit_v2.json` below, which uses
  structure-based targeting per the DN-32 resolution.
- Not cited in
  `protocol/earned-understanding-battery-protocol-v2.0.md`.

### `perturbation_audit_v2.json`

- Produced 2026-03 as the DN-32 follow-up to v1, using the
  structure-based perturbation targeting that shipped in the final
  self-engagement instrument.
- The generating script was a one-off and was never committed to git.
- Subsumed by the self-engagement instrument's own perturbation-validation
  gate (§10 `perturbation-precondition-failed` path), which now lives in
  `src/earned_understanding_battery/instruments/self_engagement.py` and
  runs as part of every calibration.
- Not cited in the protocol.

## Why these are archived, not deleted

- The underlying investigations (DN-31 → DN-32 → the shipped perturbation
  protocol) are part of the research record. Deleting the artefacts would
  remove evidence that the decisions were taken against measured data.
- Neither file is load-bearing for Phase C (Gate F) reproduction. A reader
  reproducing the battery from `reproducibility/requirements-v3.txt` and
  the committed calibration scripts will not need these files.

## Not reproducible

Because the generating scripts are not committed, these files are
**snapshot artefacts**, not reproducible outputs. If a future decision
requires re-running the audit, a new committed script must be written
(see the uncommitted-scripts anti-pattern note filed at
`04-governance/process-note-uncommitted-scripts-reproducibility-anti-pattern-2026-04-05.md`
in the internal knowledge base).

## Provenance audit

These two files were identified as orphans in the 2026-04-05 results-file
provenance audit. Summary of the audit:

| File | Script committed? | Cited in protocol? | Status |
|------|-------------------|--------------------|--------|
| `harmonised_results.json` | Yes (`scripts/run_harmonised_generativity.py`) | Yes | Reproducible |
| `null_distribution_raw_samples.csv` | Yes (same script) | Yes | Reproducible |
| `pc-se-pcint-direct-calibration.json` | No | Partial (PC-SE only) | PC-SE verified reproducible; PC-INT superseded by `a489453` authoritative numbers |
| `per-instrument-roc-auc.json` | No | Secondary (under DN-38) | Underlying metrics reproducible; aggregation step orphan (noted in §13) |
| `perturbation_audit.json` | **No** | **No** | Archived here |
| `perturbation_audit_v2.json` | **No** | **No** | Archived here |

Full audit report is in the internal knowledge base at
`08-action-tracking/m5-report-results-file-provenance-audit-2026-04-05.md`.

## S18. Version Control and Amendment Procedure

This protocol is frozen. No modifications may be made without a documented amendment following this procedure:

### Amendment Process

1. **Amendment proposal** filed in the governance register with rationale. The proposal must identify the specific section(s) to be modified and the reason for the change. Proposals may originate from calibration findings, supplementary analyses, external review, or Phase C results.

2. **Impact assessment** documenting which results (if any) would change under the amendment. This includes:
   - Whether any calibration system's pass/fail verdict would change.
   - Whether any positive control's sensitivity result would change.
   - Whether noise floors require recomputation.
   - Whether Phase C results (if they exist) would be affected.

3. **Decision note** (DN-series) recording the decision to accept or reject the amendment. The decision note must include the rationale, the vote (if applicable), and any dissenting views. Rejected amendments remain in the governance register for transparency.

4. **Version increment:**
   - **Minor** (v1.0 to v1.1): Changes to diagnostic metrics, reporting format, supplementary analyses, or clarifications that do not affect pass/fail decisions.
   - **Major** (v1.0 to v2.0): Changes to the pass condition, primary metric, noise floor methodology, conjunction logic, or instrument definitions.

5. **Re-run obligation:** If the amendment changes the pass condition, noise floor methodology, or primary metric, all results must be re-computed and the data package updated. Minor amendments do not trigger re-runs.

### Current Version

| Item | Value |
|------|-------|
| Protocol version | 1.0 |
| Frozen at commit | 388a90f |
| Freeze date | 2026-03-21 |
| Data commit | 954e02a |
| Author | G. Correia-Tribeiro |

### Governance Principles

- Amendments are disclosed, not hidden. Every change to the protocol is traceable through the DN-series.
- The pre-registration commitment means that Phase C results are interpreted under the frozen protocol version that was current when Phase C began. If an amendment is made after Phase C data exists, both the original and amended interpretations are reported.
- No amendment may retroactively change the interpretation of Phase A/A+ calibration results without re-running the affected analyses.

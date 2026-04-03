## S17. Pre-Committed Supplementary Analyses

The following analyses are pre-committed as supplementary work. They strengthen the battery's evidence base but are not blocking for Phase C. None of these analyses, if they produce unexpected results, will alter the registered pass condition or primary metrics. They inform interpretation only.

### 1. Ecological Domain Generality (Phase B)

Test the battery on ecologically valid graph domains: citation networks, social networks, and other real-world topologies with heterogeneous community structure. The SBM domains used in Phase A have homogeneous communities (all communities structurally equivalent). Ecological domains will test whether the battery's instruments are sensitive to role-differentiated structure, and whether coherence gating (currently deferred per ) can be re-enabled.

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

Re-enter structural consistency as a gating criterion when community roles are differentiated. On SBM domains, coherence is unreliable because all communities are structurally equivalent (F-048, F-049). Heterogeneous domains with hub, peripheral, and bridge communities should enable meaningful coherence measurement. If coherence gating is viable on heterogeneous domains, amend the protocol per S18 to include it.

### 6. Scale Validation

Run representative systems at SMALL (50 nodes) and LARGE (500 nodes) scales. Verify that:

- Class 1-3 systems continue to fail the conjunction at all scales.
- Positive controls continue to pass their target instruments.
- Noise floors scale predictably (tighter or wider, but not inverted).

Priority systems for scale validation: 1A, 2A, 3C, HEB, STDP, PC1, PC3.

### Relationship to Registered Protocol

These supplementary analyses strengthen but do not change the registered protocol. If any supplementary analysis reveals a flaw in the battery's discrimination (e.g., a calibration system spuriously passing at a different scale), the finding will be documented and an amendment proposed per S18. The Phase C candidate evaluation proceeds under the frozen protocol regardless of supplementary analysis status.

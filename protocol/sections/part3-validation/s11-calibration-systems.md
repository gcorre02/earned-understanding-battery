## S11. Calibration Systems

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

- **Class 1 (1A, 1B, 1C):** All three fail every instrument. No learning mechanism means no developmental trajectory, no earned structure, no generativity. Integration effects in 1A/1B are received (earned_ratio = 1.00) and correctly excluded.
- **Class 2 (2A, 2B, 2C):** All three fail the conjunction. Frozen weights preclude trajectory development. 2C shows transfer (earned_ratio = 1.26) but fails trajectory and generativity -- the conjunction catches it.
- **Class 3 (3A-3E, HEB, STDP):** Variable per-instrument profiles but all fail the conjunction. HEB achieves the richest profile (trajectory + integration + transfer) but fails generativity and self-engagement. STDP passes trajectory, transfer, and self-engagement but fails integration and generativity. The conjunction requirement for all five instruments is the discriminative gate.

**Generativity is the universal blocker.** Every calibration system produces delta = 0.000000 on the generativity instrument (no response to novel domain). This is the instrument that most consistently separates the calibration panel from a hypothetical Class 4 system. The positive controls (S12) confirm that the instrument is sensitive when genuine structural transfer is present.

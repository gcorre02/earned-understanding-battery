## S16. Hardware and Compute Requirements

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

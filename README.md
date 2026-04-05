# Earned Understanding Battery

A framework-agnostic empirical test suite for Class 4 candidacy, implementing the five-instrument battery proposed in Paper 1 (Ribeiro, 2026).

**Paper 1:** [DOI 10.5281/zenodo.19178410](https://doi.org/10.5281/zenodo.19178410)

**Pre-registered protocol:** [`protocol/earned-understanding-battery-protocol-v2.0.md`](protocol/earned-understanding-battery-protocol-v2.0.md)

## What this is

The battery operationalises the five necessary properties proposed in Paper 1 Section 3 — emergence, stability, integration, operational impact, and transfer — as a conjunction of five instruments applied under a provenance constraint. It is a Class 4 detector, not a spectrum classifier. It answers exactly one question: does a candidate system satisfy all five necessary properties under the provenance constraint?

The five instruments:

1. **Developmental Trajectory** — does structural organisation develop through operation?
2. **Integration** — is the structure non-decomposable under regional ablation?
3. **Generativity** — does frozen structure produce coherent behaviour on novel domains?
4. **Transfer** — does the structure preserve relational invariants across surface changes?
5. **Self-Engagement** — is the developed structure actively maintained?

Plus **Provenance** as the epistemic condition that makes the other five scientifically verifiable.

A system passes only if all five instruments return a positive result under the provenance constraint. See the protocol document for the conjunction logic, pass/fail rules, and per-instrument specifications.

## Installation

```bash
git clone https://github.com/<owner>/earned-understanding-battery.git
cd earned-understanding-battery
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras for running the full calibration panel:

```bash
pip install -e ".[all]"        # all system classes + analysis tools
pip install -e ".[class1]"     # Class 1 systems only (lightweight)
pip install -e ".[class2]"     # Class 2 systems (requires PyTorch)
pip install -e ".[class3]"     # Class 3 systems (requires PyTorch + transformers)
```

Python 3.11 or later is required.

## Running the tests

```bash
pytest tests/
```

The test suite contains 158 tests covering the core library, instruments, domain generation, and cross-validation of the positive controls. One test (`test_known_received`) exercises the full battery on a DistilGPT-2 Class 2 system and takes approximately 12 minutes on CPU.

## Running the battery on a candidate

```python
from earned_understanding_battery.instruments.battery_runner import run_battery, BatteryConfig
from earned_understanding_battery.core.types import SystemClass
from earned_understanding_battery.domains.sbm_generator import generate_domain_family
from earned_understanding_battery.domains.presets import MEDIUM

family = generate_domain_family(MEDIUM)
config = BatteryConfig(
    domain_a_inputs=list(family["A"].nodes())[:50],
    domain_a_prime_inputs=list(family["A_prime"].nodes())[:50],
    domain_b_inputs=list(family["B"].nodes())[:50],
)

result = run_battery(
    system=your_system,
    system_name="MySystem",
    system_class=SystemClass.CLASS_3,
    config=config,
    control_factory=your_fresh_control_factory,
)

print(f"Battery verdict: {'PASS' if result.overall_pass else 'FAIL'}")
```

See `scripts/run_calibration.py` for a complete example running the battery across all calibration systems.

## Repository structure

```
protocol/                       Pre-registered OSF protocol (v2.0)
src/earned_understanding_battery/
    core/                       Base types, provenance log, test system ABC
    domains/                    SBM domain generation, spectral verification
    instruments/                Five instruments + battery runner
    analysis/                   CKA, behavioural generativity
    systems/                    Calibration panel (Class 1/2/3 + anchor + positive controls)
tests/                          Pytest suite (158 tests)
scripts/                        Calibration and analysis scripts
results/                        Calibration data (JSON + CSV artefacts)
reproducibility/                Pinned dependencies and reproduction instructions
governance/                     Phase A+ sign-off document
```

## Citation

If you use this battery in academic work, please cite Paper 1:

```
Ribeiro, G. C. T. (2026). Understanding as earned structure: a
framework-agnostic empirical test.
https://doi.org/10.5281/zenodo.19178410
```

## Status

This repository accompanies the OSF pre-registration of the battery protocol. Phase C (candidate evaluation) is forthcoming. Calibration data for the 13-system panel is included in `results/`.

## License

MIT License — see [`LICENSE`](LICENSE). Copyright (c) 2026 Guilherme C. T. Ribeiro.

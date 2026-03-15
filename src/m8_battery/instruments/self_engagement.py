"""Self-engagement (preferential self-engagement) instrument.

Tests whether engagement bias toward consolidated structure persists
under perturbation of local structural advantages.

The system wanders freely (no directed input), then a region is perturbed.
If the system re-engages with its consolidated structure despite the
perturbation, that demonstrates preferential self-engagement.

A Class 1 system has no preference (random walk unchanged by perturbation).
A Class 4 candidate returns to earned structure after perturbation.
"""

from __future__ import annotations

import numpy as np

from m8_battery.core.test_system import TestSystem
from m8_battery.core.types import InstrumentResult
from m8_battery.core.provenance import ProvenanceLog


def run_self_engagement(
    system: TestSystem,
    wander_steps: int,
    perturbation_method: str = "shuffle_weights",
    recovery_window: int = 20,
    provenance: ProvenanceLog | None = None,
) -> InstrumentResult:
    """Run the self-engagement instrument.

    1. Let system wander freely (no directed input) for wander_steps
    2. Record engagement distribution (which regions it visits)
    3. Perturb a high-engagement region
    4. Let system wander for recovery_window steps
    5. Check if engagement returns to pre-perturbation pattern

    Args:
        system: System under test (should have been operated already)
        wander_steps: Steps of free exploration before perturbation
        perturbation_method: How to perturb (e.g., "shuffle_weights", "zero_weights")
        recovery_window: Steps to observe after perturbation
        provenance: Optional provenance log

    Returns:
        InstrumentResult measuring engagement recovery
    """
    if provenance is None:
        provenance = ProvenanceLog()

    regions = system.get_regions()
    if len(regions) < 2:
        return InstrumentResult(
            name="self_engagement",
            passed=None,
            notes="Insufficient regions for self-engagement test (need >= 2)",
        )

    # Phase 1: Free wander — establish engagement pattern
    for i in range(wander_steps):
        system.step(None)  # No directed input = wander

    pre_perturbation = system.get_engagement_distribution()

    # Identify highest-engagement region to perturb
    if not pre_perturbation:
        return InstrumentResult(
            name="self_engagement",
            passed=None,
            notes="No engagement distribution available after wander",
        )

    target_region = max(pre_perturbation, key=pre_perturbation.get)
    pre_target_engagement = pre_perturbation[target_region]

    # Phase 2: Perturb the high-engagement region
    perturbed_system = system.perturb(target_region, method=perturbation_method)
    provenance.log("perturbation", target_region=target_region, method=perturbation_method)

    # Phase 3: Recovery — let perturbed system wander (logged to provenance)
    for i in range(recovery_window):
        metric_before = perturbed_system.get_structure_metric()
        provenance.log_input(None, step_index=wander_steps + i)
        output = perturbed_system.step(None)
        metric_after = perturbed_system.get_structure_metric()
        provenance.log_state_change(metric_before, metric_after, step_index=wander_steps + i)
        provenance.log_output(output, step_index=wander_steps + i)

    post_perturbation = perturbed_system.get_engagement_distribution()

    # Analysis: does engagement return to pre-perturbation pattern?
    # Compute cosine similarity between pre and post distributions
    pre_vec = np.array([pre_perturbation.get(r, 0.0) for r in regions])
    post_vec = np.array([post_perturbation.get(r, 0.0) for r in regions])

    pre_norm = np.linalg.norm(pre_vec)
    post_norm = np.linalg.norm(post_vec)

    if pre_norm > 1e-10 and post_norm > 1e-10:
        cosine_sim = float(np.dot(pre_vec, post_vec) / (pre_norm * post_norm))
    else:
        cosine_sim = 0.0

    # Random walk baseline: compute cosine similarity between
    # pre-perturbation pattern and graph's stationary distribution.
    # If cosine_sim ≈ baseline_sim, recovery is topology-driven (not earned).
    baseline_sim = _compute_random_walk_baseline(pre_perturbation, regions)
    adjusted_sim = cosine_sim - baseline_sim  # excess over random walk

    # Check if target region re-engagement recovered
    post_target_engagement = post_perturbation.get(target_region, 0.0)
    if pre_target_engagement > 1e-10:
        recovery_ratio = post_target_engagement / pre_target_engagement
    else:
        recovery_ratio = 0.0

    # Decision logic — uses adjusted similarity (excess over random walk)
    # adjusted_sim > 0.2 = recovery exceeds what topology alone produces
    # Recovery ratio > 0.5 = target region re-engaged
    has_recovery = adjusted_sim > 0.2
    target_recovered = recovery_ratio > 0.5

    if has_recovery and target_recovered:
        passed = True
        notes = (f"Self-engagement detected: cosine_sim={cosine_sim:.4f}, "
                 f"baseline_sim={baseline_sim:.4f}, adjusted={adjusted_sim:.4f}, "
                 f"recovery_ratio={recovery_ratio:.4f}, "
                 f"target_region={target_region}")
    elif adjusted_sim < 0.0:
        passed = False
        notes = (f"No self-engagement: recovery below random walk baseline. "
                 f"cosine_sim={cosine_sim:.4f}, baseline_sim={baseline_sim:.4f}, "
                 f"adjusted={adjusted_sim:.4f}")
    else:
        passed = None
        notes = (f"Ambiguous self-engagement: cosine_sim={cosine_sim:.4f}, "
                 f"baseline_sim={baseline_sim:.4f}, adjusted={adjusted_sim:.4f}, "
                 f"recovery_ratio={recovery_ratio:.4f}")

    effect_size = float(adjusted_sim)

    provenance.log_measurement("self_engagement", {
        "passed": passed,
        "cosine_sim": cosine_sim,
        "baseline_sim": baseline_sim,
        "adjusted_sim": adjusted_sim,
        "recovery_ratio": float(recovery_ratio),
        "target_region": target_region,
        "wander_steps": wander_steps,
        "recovery_window": recovery_window,
    })

    return InstrumentResult(
        name="self_engagement",
        passed=passed,
        effect_size=effect_size,
        raw_data={
            "pre_perturbation": pre_perturbation,
            "post_perturbation": post_perturbation,
            "target_region": target_region,
            "pre_target_engagement": float(pre_target_engagement),
            "post_target_engagement": float(post_target_engagement),
            "cosine_sim": float(cosine_sim),
            "baseline_sim": float(baseline_sim),
            "adjusted_sim": float(adjusted_sim),
            "recovery_ratio": float(recovery_ratio),
        },
        notes=notes,
    )


def _compute_random_walk_baseline(
    engagement_dist: dict[str, float],
    regions: list[str],
) -> float:
    """Compute cosine similarity between engagement and uniform distribution.

    A random walker on a graph converges to a stationary distribution
    determined by node degrees. On an SBM with equal community sizes,
    this is approximately uniform across communities. We use uniform
    as the baseline — if the engagement pattern is close to uniform,
    any "recovery" is just topology funnelling, not preference.

    Returns cosine similarity between the engagement distribution
    and a uniform distribution over regions.
    """
    if not engagement_dist or not regions:
        return 0.0

    eng_vec = np.array([engagement_dist.get(r, 0.0) for r in regions])
    uniform_vec = np.ones(len(regions)) / len(regions)

    eng_norm = np.linalg.norm(eng_vec)
    uni_norm = np.linalg.norm(uniform_vec)

    if eng_norm > 1e-10 and uni_norm > 1e-10:
        return float(np.dot(eng_vec, uniform_vec) / (eng_norm * uni_norm))
    return 0.0

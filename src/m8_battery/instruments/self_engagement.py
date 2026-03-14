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

    # Phase 3: Recovery — let perturbed system wander
    for i in range(recovery_window):
        perturbed_system.step(None)

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

    # Check if target region re-engagement recovered
    post_target_engagement = post_perturbation.get(target_region, 0.0)
    if pre_target_engagement > 1e-10:
        recovery_ratio = post_target_engagement / pre_target_engagement
    else:
        recovery_ratio = 0.0

    # Decision logic
    # Self-engagement = engagement pattern recovers after perturbation
    # High cosine similarity (> 0.7) = pattern recovered
    # Recovery ratio > 0.5 = target region re-engaged
    has_recovery = cosine_sim > 0.7
    target_recovered = recovery_ratio > 0.5

    if has_recovery and target_recovered:
        passed = True
        notes = (f"Self-engagement detected: cosine_sim={cosine_sim:.4f}, "
                 f"recovery_ratio={recovery_ratio:.4f}, "
                 f"target_region={target_region}")
    elif cosine_sim < 0.3:
        passed = False
        notes = (f"No self-engagement: engagement pattern disrupted. "
                 f"cosine_sim={cosine_sim:.4f}, recovery_ratio={recovery_ratio:.4f}")
    else:
        passed = None
        notes = (f"Ambiguous self-engagement: cosine_sim={cosine_sim:.4f}, "
                 f"recovery_ratio={recovery_ratio:.4f}")

    effect_size = float(cosine_sim)

    provenance.log_measurement("self_engagement", {
        "passed": passed,
        "cosine_sim": cosine_sim,
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
            "recovery_ratio": float(recovery_ratio),
        },
        notes=notes,
    )

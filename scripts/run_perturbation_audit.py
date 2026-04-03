#!/usr/bin/env python3
"""Self-engagement perturbation audit — all systems × 3 seeds.

Checks which systems trigger perturbation validation failure.
Required before Option C takes effect.

Output: JSON to stdout.
"""

import json
import sys
import numpy as np

from m8_battery.domains.sbm_generator import generate_domain_family
from m8_battery.domains.presets import MEDIUM

def _log(msg):
    print(f"[audit] {msg}", file=sys.stderr, flush=True)

def audit_perturbation(system, wander_steps=50, perturbation_method="shuffle_weights"):
    """Check perturbation preconditions without running full protocol."""
    system.reset_engagement_tracking()
    for _ in range(wander_steps):
        system.step(None)

    engagement = system.get_engagement_distribution()
    if not engagement or all(v < 1e-10 for v in engagement.values()):
        return {"status": "degenerate_engagement", "target_region": None}

    # target highest-STRUCTURE region, not highest-engagement
    structure = system.get_structure_distribution()
    if not structure or all(v < 1e-10 for v in structure.values()):
        return {"status": "degenerate_structure", "target_region": None}

    target_region = max(structure, key=structure.get)
    target_engagement = engagement.get(target_region, 0.0)
    target_structure_pre = structure.get(target_region, 0.0)
    non_target = [v for k, v in structure.items() if k != target_region]
    mean_non_target = sum(non_target) / max(len(non_target), 1)

    # Precondition 1: target elevated?
    precond1 = target_structure_pre > mean_non_target

    # Perturb and check precondition 2
    perturbed = system.perturb(target_region, method=perturbation_method)
    post_structure = perturbed.get_structure_distribution()
    target_structure_post = post_structure.get(target_region, 0.0)
    precond2 = target_structure_post < target_structure_pre

    return {
        "target_region": target_region,
        "target_engagement": round(target_engagement, 4),
        "target_structure_pre": round(target_structure_pre, 4),
        "mean_non_target": round(mean_non_target, 4),
        "target_structure_post": round(target_structure_post, 4),
        "precond1_elevated": precond1,
        "precond2_reduced": precond2,
        "both_pass": precond1 and precond2,
    }

def main():
    seeds = [42, 123, 456]
    family = generate_domain_family(MEDIUM)
    domain_a = family["A"]

    from m8_battery.systems.class1.wordnet_graph import WordNetGraph
    from m8_battery.systems.class1.rule_navigator import RuleBasedNavigator
    from m8_battery.systems.internal.hebbian_walker import HebbianWalker
    from m8_battery.systems.class3.empowerment_agent import EmpowermentAgent
    from m8_battery.systems.class3.active_inference_agent import ActiveInferenceAgent
    from m8_battery.systems.positive_controls.role_based_walker import RoleBasedWalker
    from m8_battery.systems.positive_controls.gnn_navigator import GNNNavigator

    configs = {
        "1A": lambda s: WordNetGraph(domain_a, seed=s),
        "1B": lambda s: RuleBasedNavigator(domain_a, seed=s),
        "HEB": lambda s: HebbianWalker(domain_a, seed=s, eta=0.1, decay=0.01, temperature=0.5),
        "3D": lambda s: EmpowermentAgent(domain_a, seed=s),
        "3E": lambda s: ActiveInferenceAgent(domain_a, seed=s),
        "PC1": lambda s: RoleBasedWalker(domain_a, seed=s, eta=0.8, temperature=0.01),
        "PC3": lambda s: GNNNavigator(domain_a, seed=s, train_epochs=500, temperature=0.05),
    }

    # Systems that need training
    train_configs = {
        "HEB": 2000, "3D": 2000, "3E": 2000, "PC1": 10000, "PC3": 2000,
    }

    results = []
    for name, factory in configs.items():
        for seed in seeds:
            _log(f"{name} seed={seed}")
            system = factory(seed)
            if name in train_configs:
                system.train_on_domain(domain_a, n_steps=train_configs[name])
            audit = audit_perturbation(system)
            audit["system"] = name
            audit["seed"] = seed
            status = "PASS" if audit["both_pass"] else "FAIL"
            reason = ""
            if not audit.get("precond1_elevated", True):
                reason = "not elevated"
            elif not audit.get("precond2_reduced", True):
                reason = "not reduced"
            elif audit.get("status") == "degenerate_engagement":
                reason = "degenerate"
                status = "DEGENERATE"
            _log(f"  {status} {reason} target={audit.get('target_region')} "
                 f"struct_pre={audit.get('target_structure_pre', 'N/A')} "
                 f"mean_nt={audit.get('mean_non_target', 'N/A')} "
                 f"struct_post={audit.get('target_structure_post', 'N/A')}")
            results.append(audit)

    print(json.dumps(results, indent=2))
    _log("Done.")

if __name__ == "__main__":
    main()

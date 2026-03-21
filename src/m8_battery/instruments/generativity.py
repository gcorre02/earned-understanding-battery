"""Generativity instrument (DN-18 behavioural JSD + DN-30 coherence).

Tests whether frozen earned structure produces coherent, differentiated
responses on novel domains compared to a fresh system.

Metrics (peer reviewer #2 upgrade):
- PRIMARY: Transition-matrix JSD (community-to-community movement patterns)
- DIAGNOSTIC: Marginal engagement JSD (community visit proportions, retained)
- Self-transition rate (community persistence, replaces autocorrelation)

Pass condition (DN-29 strict + DN-30 + Rigour Principles):
1. JSD > calibrated threshold (trained behaviour differs from fresh)
2. Coherence > 0 (normalised entropy difference; trained more structured)
3. Signal classified (not degenerate, not maximum-divergence, not confounded)

Threshold status: PRELIMINARY until derived from positive/negative distributions.
Results with preliminary thresholds are RAW DATA, not classifications.

Rigour Principles (from F-039 response):
- P1: No conclusions without calibrated thresholds
- P2: Every metric bounded and interpretable
- P4: Signal types distinguished
- P7: Red-flag results investigated before reporting
"""

from __future__ import annotations

import sys
from typing import Any, Callable

import numpy as np
from scipy.spatial.distance import jensenshannon

from m8_battery.core.test_system import TestSystem
from m8_battery.core.types import InstrumentResult
from m8_battery.core.provenance import ProvenanceLog


def _log(msg: str) -> None:
    print(f"[generativity] {msg}", file=sys.stderr, flush=True)


# Maximum JSD for natural log = ln(2) ≈ 0.6931
MAX_JSD = float(np.log(2))

# Minimum communities visited for non-degenerate engagement
MIN_VISITED_COMMUNITIES = 3

# Threshold status
THRESHOLD_STATUS = "PRELIMINARY"  # Change to CALIBRATED after proper derivation


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base e) between two distributions.

    Range: [0, ln(2)] ≈ [0, 0.6931]. Bounded and symmetric.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    jsd = float(0.5 * kl_pm + 0.5 * kl_qm)
    # Clamp to valid range (numerical safety)
    return max(0.0, min(jsd, MAX_JSD))


def _engagement_entropy(dist: dict[str, float]) -> float:
    """Shannon entropy of engagement distribution. Range: [0, ln(n_regions)]."""
    if not dist or all(v <= 0 for v in dist.values()):
        return 0.0
    vals = np.array(list(dist.values()), dtype=np.float64)
    vals = vals + 1e-10
    vals = vals / vals.sum()
    return float(-np.sum(vals * np.log(vals)))


def _count_visited(dist: dict[str, float], min_fraction: float = 0.01) -> int:
    """Count communities with non-trivial engagement (> min_fraction)."""
    return sum(1 for v in dist.values() if v > min_fraction)


def _coherence_normalised(trained_entropy: float, fresh_entropy: float) -> float:
    """Normalised coherence difference. Bounded [-1, 1].

    > 0: trained is more structured (lower entropy)
    = 0: equal structure
    < 0: trained is less structured (maladaptive)
    """
    denom = fresh_entropy + trained_entropy + 1e-10
    return float((fresh_entropy - trained_entropy) / denom)


def _compute_transition_matrix(
    visit_sequence: list,
    node_to_community: dict,
    n_communities: int,
) -> np.ndarray:
    """Compute community-to-community transition matrix from visit sequence.

    Returns (k x k) matrix where entry [i,j] = P(next community = j | current = i).
    Row-normalised. Rows with no transitions are uniform.
    """
    T = np.zeros((n_communities, n_communities))
    for t in range(len(visit_sequence) - 1):
        c_from = node_to_community.get(visit_sequence[t], 0)
        c_to = node_to_community.get(visit_sequence[t + 1], 0)
        T[c_from, c_to] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    T = T / row_sums
    return T


def _transition_jsd(T_trained: np.ndarray, T_fresh: np.ndarray) -> float:
    """Row-wise JSD between two transition matrices, weighted average.

    Each row is a distribution over next-community. Rows where both systems
    have transitions are included; rows where either has no transitions are
    skipped. Weighted by min(row_count) for reliability.

    Range: [0, ln(2)] (same as marginal JSD).
    """
    k = T_trained.shape[0]
    row_jsds = []
    weights = []
    for i in range(k):
        t_sum = T_trained[i].sum()
        f_sum = T_fresh[i].sum()
        if t_sum > 1e-10 and f_sum > 1e-10:
            # scipy jensenshannon returns sqrt(JSD) with base e
            jsd_sqrt = jensenshannon(T_trained[i], T_fresh[i])
            row_jsds.append(jsd_sqrt ** 2)
            weights.append(min(t_sum, f_sum))
    if not row_jsds:
        return 0.0
    return float(np.clip(np.average(row_jsds, weights=np.array(weights)), 0.0, MAX_JSD))


def _transition_entropy(T: np.ndarray) -> float:
    """Average row entropy of transition matrix. Lower = more structured."""
    row_entropies = []
    for i in range(T.shape[0]):
        if T[i].sum() > 1e-10:
            row = T[i] + 1e-10
            row = row / row.sum()
            row_entropies.append(float(-np.sum(row * np.log(row))))
    return float(np.mean(row_entropies)) if row_entropies else 0.0


def _self_transition_rate(T: np.ndarray) -> float:
    """Average diagonal of transition matrix. Higher = more community persistence."""
    active_rows = [i for i in range(T.shape[0]) if T[i].sum() > 1e-10]
    if not active_rows:
        return 0.0
    return float(np.mean([T[i, i] for i in active_rows]))


def _classify_signal(
    jsd: float,
    coherence: float,
    trained_visited: int,
    fresh_visited: int,
    edge_overlap: float | None = None,
) -> str:
    """Classify the generativity signal type (Rigour Principle 4)."""
    if jsd < 1e-6:
        return "absent"
    if trained_visited < MIN_VISITED_COMMUNITIES:
        return "degenerate_trained"
    if fresh_visited < MIN_VISITED_COMMUNITIES:
        return "degenerate_fresh"
    if jsd > 0.99 * MAX_JSD:
        return "maximum_divergence"  # Red flag — investigate
    # PRELIMINARY threshold — 5% edge overlap. Not calibrated from data.
    if edge_overlap is not None and edge_overlap > 0.05:
        return "potentially_confounded"
    if coherence <= 0:
        return "divergent_incoherent"  # Maladaptive bias
    return "candidate_generativity"


def run_generativity(
    system: TestSystem,
    domain_b_inputs: list[Any],
    reference_metric: float,
    provenance: ProvenanceLog | None = None,
    control_factory: Callable[[], TestSystem] | None = None,
    jsd_threshold: float = 0.05,
    edge_overlap: float | None = None,
    domain_b_graph: Any = None,
) -> InstrumentResult:
    """Run the generativity instrument (DN-18 behavioural + DN-30 coherence).

    THRESHOLD STATUS: PRELIMINARY. Results are raw data, not classifications,
    until the threshold is derived from positive/negative distributions.

    Args:
        system: Trained system (frozen via set_training(False) by battery runner)
        domain_b_inputs: Inputs from novel domain B
        reference_metric: Baseline structure metric from domain A (diagnostic)
        provenance: Optional provenance log
        control_factory: Returns fresh matched control for JSD comparison
        jsd_threshold: Preliminary JSD threshold (UNCALIBRATED)
        edge_overlap: Jaccard edge overlap between A and B (for confound report)
    """
    if provenance is None:
        provenance = ProvenanceLog()

    if not domain_b_inputs:
        return InstrumentResult(
            name="generativity",
            passed=None,
            notes="No domain B inputs provided",
        )

    # --- Trained system on domain B (already frozen by battery runner) ---
    # Switch to domain B graph if provided (Issue 3 fix: graph walkers must
    # navigate B's topology, not A's)
    if domain_b_graph is not None:
        system.set_domain(domain_b_graph)

    metric_before = system.get_structure_metric()
    system.reset_engagement_tracking()

    trajectory = [metric_before]
    trained_visit_seq = []  # Node-by-node visit sequence for transition matrix
    n_steps = len(domain_b_inputs)
    for i in range(n_steps):
        # Autonomous navigation: step(None). The system navigates domain B
        # using its frozen structure. No teleporting to input nodes.
        inp = None
        provenance.log_input(inp, step_index=i)
        metric_pre = system.get_structure_metric()
        output = system.step(inp)
        metric_post = system.get_structure_metric()
        provenance.log_state_change(metric_pre, metric_post, step_index=i)
        provenance.log_output(output, step_index=i)
        trajectory.append(metric_post)
        trained_visit_seq.append(output)

    metric_after = system.get_structure_metric()
    trained_engagement = system.get_engagement_distribution()
    trained_entropy = _engagement_entropy(trained_engagement)
    trained_visited = _count_visited(trained_engagement)

    # Structural metric diagnostics (retained, not pass/fail)
    delta = metric_after - metric_before
    relative_delta = delta / reference_metric if abs(reference_metric) > 1e-10 else 0.0
    trajectory_stability = 1.0 - (np.array(trajectory).std() / (np.array(trajectory).mean() + 1e-10))
    retention = metric_after / reference_metric if abs(reference_metric) > 1e-10 else 0.0

    # --- Fresh system on domain B (DN-18 behavioural comparison) ---
    jsd = 0.0
    fresh_entropy = 0.0
    coherence = 0.0
    fresh_engagement = {}
    fresh_visited = 0
    fresh_visit_seq = []
    # Transition-matrix metrics (peer reviewer #2 recommendation)
    t_jsd = 0.0
    t_coherence = 0.0
    trained_self_trans = 0.0
    fresh_self_trans = 0.0

    # Build community mapping from domain B graph for transition matrix
    node_to_community: dict[Any, int] = {}
    n_communities = 0
    if domain_b_graph is not None:
        for node in domain_b_graph.nodes():
            data = domain_b_graph.nodes[node]
            features = data.get("features", {})
            node_to_community[node] = features.get("community", data.get("block", 0))
        n_communities = len(set(node_to_community.values()))

    if control_factory is not None:
        try:
            fresh = control_factory()
            if domain_b_graph is not None:
                fresh.set_domain(domain_b_graph)
            fresh.set_training(False)
            fresh.reset_engagement_tracking()
            for _ in range(n_steps):
                output = fresh.step(None)  # Autonomous navigation on domain B
                fresh_visit_seq.append(output)
            fresh_engagement = fresh.get_engagement_distribution()
            fresh_entropy = _engagement_entropy(fresh_engagement)
            fresh_visited = _count_visited(fresh_engagement)

            # Marginal JS divergence (bounded [0, ln(2)]) — DIAGNOSTIC
            regions = sorted(set(list(trained_engagement.keys()) + list(fresh_engagement.keys())))
            trained_vec = np.array([trained_engagement.get(r, 0.0) for r in regions])
            fresh_vec = np.array([fresh_engagement.get(r, 0.0) for r in regions])
            jsd = _js_divergence(trained_vec, fresh_vec)

            # DN-30 coherence for marginal: normalised difference (bounded [-1, 1])
            coherence = _coherence_normalised(trained_entropy, fresh_entropy)

            # --- Transition-matrix metrics (PRIMARY) ---
            if n_communities > 0 and len(trained_visit_seq) > 1 and len(fresh_visit_seq) > 1:
                T_trained = _compute_transition_matrix(trained_visit_seq, node_to_community, n_communities)
                T_fresh = _compute_transition_matrix(fresh_visit_seq, node_to_community, n_communities)
                t_jsd = _transition_jsd(T_trained, T_fresh)
                trained_t_entropy = _transition_entropy(T_trained)
                fresh_t_entropy = _transition_entropy(T_fresh)
                t_coherence = _coherence_normalised(trained_t_entropy, fresh_t_entropy)
                trained_self_trans = _self_transition_rate(T_trained)
                fresh_self_trans = _self_transition_rate(T_fresh)

            _log(f"  marginal_JSD={jsd:.4f} transition_JSD={t_jsd:.4f} "
                 f"trained_H={trained_entropy:.4f} fresh_H={fresh_entropy:.4f} "
                 f"coherence={coherence:.4f} t_coherence={t_coherence:.4f} "
                 f"self_trans={trained_self_trans:.3f}/{fresh_self_trans:.3f}")
        except Exception as e:
            _log(f"  Fresh baseline failed: {e}")
    else:
        _log("  No control_factory — JSD comparison unavailable")

    # --- Signal classification (Rigour Principle 4) ---
    signal_type = _classify_signal(jsd, coherence, trained_visited, fresh_visited, edge_overlap)
    _log(f"  signal_type={signal_type}")

    # --- Pass/fail: PRELIMINARY threshold (Rigour Principle 1) ---
    # These classifications are PROVISIONAL until threshold is calibrated
    passes_jsd = jsd > jsd_threshold
    passes_coherence = coherence > 0  # Normalised: > 0 means trained more structured

    # Degeneracy overrides
    if signal_type.startswith("degenerate"):
        passed = False
        failure_mode = signal_type
        notes = (f"Degenerate engagement: trained visited {trained_visited}, "
                 f"fresh visited {fresh_visited} communities. "
                 f"JSD={jsd:.4f} is not interpretable.")
    elif signal_type == "maximum_divergence":
        passed = None  # Red flag — cannot classify without investigation
        failure_mode = "maximum_divergence"
        notes = (f"RED FLAG: JSD={jsd:.4f} = maximum ({MAX_JSD:.4f}). "
                 f"Zero overlap between engagement distributions. "
                 f"Investigate before classifying. [PRELIMINARY]")
    elif passes_jsd and passes_coherence and signal_type == "candidate_generativity":
        # Rigour P1: no definitive classification with uncalibrated threshold
        passed = None if THRESHOLD_STATUS != "CALIBRATED" else True
        failure_mode = "earned"
        notes = (f"Candidate generativity: JSD={jsd:.4f}, "
                 f"coherence={coherence:.4f}. [THRESHOLD={THRESHOLD_STATUS}]")
    elif passes_jsd and not passes_coherence:
        passed = False
        failure_mode = "architectural"
        notes = (f"Divergence but incoherent: JSD={jsd:.4f}, coherence={coherence:.4f}<=0. "
                 f"Trained differs but is less structured.")
    elif signal_type == "potentially_confounded":
        passed = None  # Cannot classify until confound resolved
        failure_mode = "potentially_confounded"
        notes = (f"JSD={jsd:.4f} but edge overlap={edge_overlap:.4f}. "
                 f"Signal may be from shared edges, not structural abstraction.")
    elif jsd > 0.001:
        passed = False
        failure_mode = "scale-limited"
        notes = (f"Sub-threshold: JSD={jsd:.6f} < threshold={jsd_threshold}. "
                 f"[THRESHOLD={THRESHOLD_STATUS}]")
    else:
        passed = False
        failure_mode = "absent"
        notes = f"No behavioural divergence: JSD={jsd:.6f}"

    effect_size = float(t_jsd) if t_jsd > 0 else float(jsd)

    provenance.log_measurement("generativity", {
        "passed": passed,
        "marginal_jsd": jsd,
        "transition_jsd": t_jsd,
        "coherence": coherence,
        "transition_coherence": t_coherence,
        "signal_type": signal_type,
        "trained_entropy": trained_entropy,
        "fresh_entropy": fresh_entropy,
        "trained_visited": trained_visited,
        "fresh_visited": fresh_visited,
        "trained_self_transition": trained_self_trans,
        "fresh_self_transition": fresh_self_trans,
        "threshold_status": THRESHOLD_STATUS,
    })

    return InstrumentResult(
        name="generativity",
        passed=passed,
        effect_size=effect_size,
        raw_data={
            # Marginal JSD (diagnostic, retained for comparability)
            "jsd": float(jsd),
            "jsd_max": float(MAX_JSD),
            "jsd_threshold": float(jsd_threshold),
            "threshold_status": THRESHOLD_STATUS,
            "coherence": float(coherence),
            "signal_type": signal_type,
            "trained_entropy": float(trained_entropy),
            "fresh_entropy": float(fresh_entropy),
            "trained_visited": trained_visited,
            "fresh_visited": fresh_visited,
            "trained_engagement": trained_engagement,
            "fresh_engagement": fresh_engagement,
            # Transition-matrix JSD (primary metric — peer reviewer #2)
            "transition_jsd": float(t_jsd),
            "transition_coherence": float(t_coherence),
            "trained_self_transition_rate": float(trained_self_trans),
            "fresh_self_transition_rate": float(fresh_self_trans),
            # Diagnostics
            "edge_overlap": edge_overlap,
            "metric_before": float(metric_before),
            "metric_after": float(metric_after),
            "reference_metric": float(reference_metric),
            "delta": float(delta),
            "relative_delta": float(relative_delta),
            "retention": float(retention),
            "trajectory_stability": float(trajectory_stability),
            "trajectory": [float(x) for x in trajectory],
        },
        notes=notes,
        failure_mode=failure_mode,
    )

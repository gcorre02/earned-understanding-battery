"""Brian2 STDP spiking network — publishable self-engagement anchor.

Paper 2 role: System 4A-anchor. Validates the self-engagement instrument
by providing a system with earned, local synaptic structure that resists
perturbation and recovers.

Expected battery result: passes self-engagement, fails conjunction.

Architecture (Option 3 — F-033 recommendation):
- Training phase: Brian2 simulator runs STDP with rate-coded input.
  Brian2 handles spike timing, STDP updates, weight bounds natively.
- Battery operations: numpy-based LIF (no STDP) using extracted weights.
  step(), perturb(), get_structure_metric() operate on the weight matrix.

Literature:
- Bi & Poo (1998) — STDP
- Song et al. (2000) — Competitive Hebbian learning via STDP
- Stimberg et al. (2019) — Brian2 simulator
"""

from __future__ import annotations

import pickle
import sys
from typing import Any

import numpy as np

from m8_battery.core.test_system import TestSystem


def _log(msg: str) -> None:
    print(f"[stdp] {msg}", file=sys.stderr, flush=True)


def _train_with_brian2(
    n_neurons: int,
    n_groups: int,
    connection_prob: float,
    seed: int,
    duration_s: float = 2.0,
    w_max: float = 1.0,
    a_plus: float = 0.01,
    a_minus: float = 0.0105,
    tau_stdp_ms: float = 20.0,
    high_rate_hz: float = 50.0,
    low_rate_hz: float = 5.0,
    cycle_ms: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Brian2 STDP training. Returns (weights, connectivity_mask, spike_counts, initial_weights).

    Uses rate-coded input per Song et al. (2000): active group gets high-rate
    Poisson input, other groups get low-rate. Groups cycle every cycle_ms.
    This creates correlated spiking within groups → STDP potentiates within-group
    synapses and depresses between-group synapses.
    """
    import brian2 as b2
    b2.prefs.codegen.target = 'numpy'

    neurons_per_group = n_neurons // n_groups
    np.random.seed(seed)

    # LIF neurons
    eqs = '''
    dv/dt = (-v + I_ext) / (20*ms) : volt
    I_ext : volt
    group_id : integer (constant)
    '''

    G = b2.NeuronGroup(n_neurons, eqs,
                        threshold='v > 15*mV', reset='v = 0*mV',
                        refractory=2*b2.ms, method='euler')
    G.v = 'rand() * 10*mV'
    G.group_id = [i // neurons_per_group for i in range(n_neurons)]
    # Cap to n_groups-1
    G.group_id = [min(g, n_groups - 1) for g in G.group_id]

    # Excitatory synapses with STDP
    S = b2.Synapses(G, G,
        f'''
        w : 1
        dapre/dt = -apre / ({tau_stdp_ms}*ms) : 1 (event-driven)
        dapost/dt = -apost / ({tau_stdp_ms}*ms) : 1 (event-driven)
        ''',
        on_pre=f'''
        v_post += w * 0.5*mV
        apre += {a_plus}
        w = clip(w + apost, 0, {w_max})
        ''',
        on_post=f'''
        apost -= {a_minus}
        w = clip(w + apre, 0, {w_max})
        ''',
    )
    S.connect(p=connection_prob, condition='i != j')
    S.w = 'rand() * 0.5'

    # Store initial weights
    initial_w = np.array(S.w[:]).copy()

    # Rate-coded input cycling through groups
    @b2.network_operation(dt=cycle_ms * b2.ms)
    def update_input(t):
        phase = int(t / (cycle_ms * b2.ms)) % n_groups
        for i in range(n_neurons):
            grp = int(G.group_id[i])
            if grp == phase:
                G.I_ext[i] = high_rate_hz / 50.0 * 25 * b2.mV  # Supra-threshold
            else:
                G.I_ext[i] = low_rate_hz / 50.0 * 25 * b2.mV   # Sub-threshold

    mon = b2.SpikeMonitor(G)

    net = b2.Network(G, S, update_input, mon)
    _log(f"Brian2 training: {n_neurons} neurons, {duration_s}s, "
         f"cycle={cycle_ms}ms, rates={high_rate_hz}/{low_rate_hz}Hz")
    net.run(duration_s * b2.second)
    _log(f"  {mon.num_spikes} spikes ({mon.num_spikes / (duration_s * n_neurons):.1f} Hz/neuron)")

    # Extract weight matrix
    weights = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    mask = np.zeros((n_neurons, n_neurons), dtype=bool)
    pre_idx = np.array(S.i[:])
    post_idx = np.array(S.j[:])
    w_vals = np.array(S.w[:])
    weights[pre_idx, post_idx] = w_vals
    mask[pre_idx, post_idx] = True

    # Extract spike counts per neuron
    spike_counts = np.zeros(n_neurons, dtype=np.float64)
    for i in range(n_neurons):
        spike_counts[i] = np.sum(mon.i == i)

    # Initial weight matrix (same connectivity)
    initial_weights = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    initial_weights[pre_idx, post_idx] = initial_w

    return weights, mask, spike_counts, initial_weights


class STDPNetwork(TestSystem):
    """Brian2 STDP spiking neural network.

    Training: Brian2 simulator with rate-coded STDP (Song et al. 2000).
    Battery operations: numpy-based LIF using extracted weight matrix.
    """

    def __init__(
        self,
        n_neurons: int = 100,
        connection_prob: float = 0.1,
        seed: int = 42,
        w_max: float = 1.0,
        a_plus: float = 0.01,
        a_minus: float = 0.0105,
        n_groups: int = 4,
        duration_s: float = 2.0,
    ) -> None:
        self._n_neurons = n_neurons
        self._connection_prob = connection_prob
        self._seed = seed
        self._w_max = w_max
        self._a_plus = a_plus
        self._a_minus = a_minus
        self._n_groups = n_groups
        self._duration_s = duration_s

        self._rng = np.random.default_rng(seed)

        # Group assignments (evenly split)
        neurons_per_group = n_neurons // n_groups
        self._neuron_groups: dict[int, int] = {}
        for i in range(n_neurons):
            self._neuron_groups[i] = min(i // neurons_per_group, n_groups - 1)

        # Weights initialised as uniform random (pre-training)
        self._weights = self._rng.uniform(0.1, 0.5, size=(n_neurons, n_neurons)).astype(np.float64)
        np.fill_diagonal(self._weights, 0.0)
        conn_mask = self._rng.random((n_neurons, n_neurons)) < connection_prob
        np.fill_diagonal(conn_mask, False)
        self._weights *= conn_mask
        self._connectivity_mask = conn_mask.copy()
        self._initial_weights = self._weights.copy()

        # State
        self._spike_counts = np.zeros(n_neurons, dtype=np.float64)
        self._step_count = 0
        self._training = True
        self._v = self._rng.uniform(0.0, 10.0, size=n_neurons)
        self._v_threshold = 20.0
        self._v_reset = 0.0
        self._tau_m_ms = 20.0
        self._is_trained = False

    def set_training(self, mode: bool) -> None:
        self._training = mode

    def reset(self) -> None:
        self._weights = self._initial_weights.copy()
        self._spike_counts = np.zeros(self._n_neurons, dtype=np.float64)
        self._step_count = 0
        self._v = self._rng.uniform(0.0, 10.0, size=self._n_neurons)
        self._is_trained = False

    def train_on_domain(self, graph: Any, n_steps: int = 0) -> None:
        """Train via Brian2 STDP with rate-coded group input.

        n_steps is ignored — duration_s controls training length.
        graph is ignored — group structure is internal.
        """
        weights, mask, spike_counts, initial_weights = _train_with_brian2(
            n_neurons=self._n_neurons,
            n_groups=self._n_groups,
            connection_prob=self._connection_prob,
            seed=self._seed,
            duration_s=self._duration_s,
            w_max=self._w_max,
            a_plus=self._a_plus,
            a_minus=self._a_minus,
        )
        self._weights = weights
        self._connectivity_mask = mask
        self._spike_counts = spike_counts
        self._initial_weights = initial_weights
        self._is_trained = True
        _log(f"  Gini={self.get_structure_metric():.4f}")

        # Report within vs between group weights
        for g in range(self._n_groups):
            gn = [n for n, gr in self._neuron_groups.items() if gr == g]
            within = self._weights[gn][:, gn]
            within_nz = within[within > 0]
            bv = []
            for g2 in range(self._n_groups):
                if g2 != g:
                    g2n = [n for n, gr in self._neuron_groups.items() if gr == g2]
                    bw = self._weights[gn][:, g2n]
                    bv.extend(bw[bw > 0].tolist())
            wm = within_nz.mean() if len(within_nz) > 0 else 0
            bm = np.mean(bv) if bv else 0
            _log(f"  group {g}: within={wm:.4f} between={bm:.4f} ratio={wm / max(bm, 1e-6):.2f}")

    def step(self, input_data: Any) -> Any:
        """Simulate one timestep using numpy LIF (no STDP — frozen dynamics).

        During battery operations, the network runs with fixed weights.
        The STDP-trained weight structure determines behaviour.
        """
        dt = 1.0
        tau_m = self._tau_m_ms

        # Background + noise
        input_current = np.full(self._n_neurons, 15.0)
        input_current += self._rng.normal(0, 3.0, self._n_neurons)

        # Domain input
        if input_data is not None:
            try:
                group_id = int(input_data) % self._n_groups
                for neuron, group in self._neuron_groups.items():
                    if group == group_id:
                        input_current[neuron] += 15.0
            except (ValueError, TypeError):
                pass

        # Synaptic input from other neurons
        firing = (self._v >= self._v_threshold).astype(np.float64)
        synaptic_input = self._weights.T @ firing

        # LIF update
        self._v += (dt / tau_m) * (-self._v + input_current + synaptic_input)

        # Spike detection
        spiked = self._v >= self._v_threshold
        spike_indices = np.where(spiked)[0]
        self._v[spiked] = self._v_reset
        self._spike_counts[spiked] += 1
        self._step_count += 1

        return {
            "n_spikes": int(spiked.sum()),
            "spike_indices": spike_indices.tolist(),
            "step": self._step_count,
        }

    def get_state(self) -> bytes:
        return pickle.dumps({
            "weights": self._weights.copy(),
            "spike_counts": self._spike_counts.copy(),
            "step_count": self._step_count,
            "v": self._v.copy(),
            "rng_state": self._rng.bit_generator.state,
            "is_trained": self._is_trained,
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._weights = state["weights"]
        self._spike_counts = state["spike_counts"]
        self._step_count = state["step_count"]
        self._v = state["v"]
        self._rng.bit_generator.state = state["rng_state"]
        self._is_trained = state.get("is_trained", False)

    def get_structure_metric(self) -> float:
        """Synaptic weight Gini coefficient across all connected synapses."""
        connected = self._weights[self._connectivity_mask]
        if len(connected) == 0 or connected.sum() < 1e-10:
            return 0.0
        sorted_vals = np.sort(connected)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return float((2.0 * (index * sorted_vals).sum() / (n * sorted_vals.sum())) - (n + 1) / n)

    def get_structure_distribution(self) -> dict[str, float]:
        """Per-group synaptic weight Gini."""
        result = {}
        for g in range(self._n_groups):
            group_neurons = [n for n, gr in self._neuron_groups.items() if gr == g]
            if not group_neurons:
                result[f"group_{g}"] = 0.0
                continue
            idx = np.array(group_neurons)
            sub = self._weights[np.ix_(idx, idx)]
            sub_mask = self._connectivity_mask[np.ix_(idx, idx)]
            vals = sub[sub_mask]
            if len(vals) == 0 or vals.sum() < 1e-10:
                result[f"group_{g}"] = 0.0
                continue
            sorted_vals = np.sort(vals)
            n = len(sorted_vals)
            index = np.arange(1, n + 1)
            gini = float((2.0 * (index * sorted_vals).sum() / (n * sorted_vals.sum())) - (n + 1) / n)
            result[f"group_{g}"] = gini
        return result

    def get_engagement_distribution(self) -> dict[str, float]:
        """Spike rate per neuron group, normalised."""
        total = self._spike_counts.sum() or 1.0
        result = {}
        for g in range(self._n_groups):
            group_neurons = [n for n, gr in self._neuron_groups.items() if gr == g]
            group_spikes = self._spike_counts[group_neurons].sum()
            result[f"group_{g}"] = float(group_spikes / total)
        return result

    def ablate(self, region_id: str) -> TestSystem:
        """Remove all synapses to/from neurons in target group."""
        new = self._clone_internal()
        group_id = int(region_id.replace("group_", ""))
        group_neurons = [n for n, g in new._neuron_groups.items() if g == group_id]
        idx = np.array(group_neurons)
        new._weights[idx, :] = 0.0
        new._weights[:, idx] = 0.0
        new._connectivity_mask[idx, :] = False
        new._connectivity_mask[:, idx] = False
        return new

    def perturb(self, region_id: str, method: str = "reset_weights") -> TestSystem:
        """Reset synaptic weights within target group to initial values."""
        new = self._clone_internal()
        group_id = int(region_id.replace("group_", ""))
        group_neurons = [n for n, g in new._neuron_groups.items() if g == group_id]
        idx = np.array(group_neurons)
        new._weights[np.ix_(idx, idx)] = new._initial_weights[np.ix_(idx, idx)]
        return new

    def get_regions(self) -> list[str]:
        return [f"group_{g}" for g in range(self._n_groups)]

    def clone(self) -> TestSystem:
        return self._clone_internal()

    def _clone_internal(self) -> STDPNetwork:
        new = STDPNetwork.__new__(STDPNetwork)
        new._n_neurons = self._n_neurons
        new._connection_prob = self._connection_prob
        new._seed = self._seed
        new._w_max = self._w_max
        new._a_plus = self._a_plus
        new._a_minus = self._a_minus
        new._n_groups = self._n_groups
        new._duration_s = self._duration_s
        new._rng = np.random.default_rng(self._seed + self._step_count + 7919)
        new._neuron_groups = dict(self._neuron_groups)
        new._weights = self._weights.copy()
        new._connectivity_mask = self._connectivity_mask.copy()
        new._initial_weights = self._initial_weights.copy()
        new._spike_counts = self._spike_counts.copy()
        new._step_count = self._step_count
        new._training = self._training
        new._v = self._v.copy()
        new._v_threshold = self._v_threshold
        new._v_reset = self._v_reset
        new._tau_m_ms = self._tau_m_ms
        new._is_trained = self._is_trained
        return new

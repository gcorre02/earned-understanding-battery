"""Brian2 STDP spiking network — publishable self-engagement anchor.

Paper 2 role: System 4A-anchor. Validates the self-engagement instrument.

Architecture (DN-23 — live Brian2):
- Brian2 simulator runs LIVE during ALL battery operations
- STDP is ALWAYS active — weights update based on spike timing
- step() advances the Brian2 simulation
- perturb() modifies weights IN the live network
- Fresh baseline also runs live (STDP active, untrained)

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


def _build_network(
    n_neurons: int = 100,
    n_groups: int = 4,
    connection_prob: float = 0.1,
    seed: int = 42,
    w_max: float = 1.0,
    a_plus: float = 0.01,
    a_minus: float = 0.0105,
    tau_stdp_ms: float = 20.0,
    dt_ms: float = 1.0,
) -> dict:
    """Build a fresh Brian2 STDP network. Returns dict of components."""
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
                        refractory=2*b2.ms, method='euler',
                        dt=dt_ms*b2.ms)
    G.v = 'rand() * 10*mV'
    group_ids = [min(i // neurons_per_group, n_groups - 1) for i in range(n_neurons)]
    G.group_id = group_ids

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
        dt=dt_ms*b2.ms,
    )
    S.connect(p=connection_prob, condition='i != j')
    S.w = 'rand() * 0.5'

    mon = b2.SpikeMonitor(G)

    net = b2.Network(G, S, mon)

    return {
        "net": net,
        "G": G,
        "S": S,
        "mon": mon,
        "n_neurons": n_neurons,
        "n_groups": n_groups,
        "group_ids": group_ids,
        "dt_ms": dt_ms,
    }


class STDPNetwork(TestSystem):
    """Brian2 STDP spiking neural network — live simulator (DN-23).

    Brian2 runs continuously. STDP is always active. The system is
    always earning. step() advances the simulation. perturb() modifies
    weights in the live network. clone() creates a new independent
    Brian2 network with the same weight state.
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
        dt_ms: float = 1.0,
    ) -> None:
        self._n_neurons = n_neurons
        self._connection_prob = connection_prob
        self._seed = seed
        self._w_max = w_max
        self._a_plus = a_plus
        self._a_minus = a_minus
        self._n_groups = n_groups
        self._dt_ms = dt_ms

        # Build live Brian2 network
        self._components = _build_network(
            n_neurons=n_neurons, n_groups=n_groups,
            connection_prob=connection_prob, seed=seed,
            w_max=w_max, a_plus=a_plus, a_minus=a_minus,
            dt_ms=dt_ms,
        )

        self._step_count = 0
        self._training = True  # T1-03: can freeze STDP

        # Track visit counts per group for engagement
        self._group_spike_counts = np.zeros(n_groups, dtype=np.float64)

        # Store initial weights for reset
        self._initial_weights = np.array(self._components["S"].w[:]).copy()

    def set_training(self, mode: bool) -> None:
        """Enable/disable STDP during step().

        When frozen, step() snapshots weights before Brian2 runs and
        restores them after — LIF dynamics run but weight updates are undone.
        """
        self._training = mode

    def reset(self) -> None:
        """Reset all synaptic weights to initial values. Network stays live."""
        S = self._components["S"]
        S.w[:] = self._initial_weights.copy()
        self._step_count = 0
        self._group_spike_counts = np.zeros(self._n_groups, dtype=np.float64)

    def step(self, input_data: Any) -> Any:
        """Advance Brian2 simulation by dt. STDP is active."""
        import brian2 as b2

        G = self._components["G"]
        net = self._components["net"]
        mon = self._components["mon"]
        S = self._components["S"]

        # Set input currents
        # Background: sub-threshold for all
        for i in range(self._n_neurons):
            G.I_ext[i] = 5 * b2.mV

        # Domain input: boost target group
        if input_data is not None:
            try:
                group_id = int(input_data) % self._n_groups
                for i in range(self._n_neurons):
                    if self._components["group_ids"][i] == group_id:
                        G.I_ext[i] = 30 * b2.mV
            except (ValueError, TypeError):
                pass
        else:
            # Default: cycle through groups based on step count
            group = self._step_count % (self._n_groups * 20) // 20
            for i in range(self._n_neurons):
                if self._components["group_ids"][i] == group:
                    G.I_ext[i] = 30 * b2.mV

        # Snapshot weights if frozen (T1-03)
        if not self._training:
            w_before = np.array(S.w[:]).copy()

        # Advance simulation by 1 timestep
        spike_count_before = mon.num_spikes
        net.run(self._dt_ms * b2.ms)
        new_spikes = mon.num_spikes - spike_count_before

        # Restore weights if frozen
        if not self._training:
            S.w[:] = w_before

        # Update group spike counts from recent spikes
        if new_spikes > 0:
            recent_indices = mon.i[spike_count_before:]
            for idx in recent_indices:
                grp = self._components["group_ids"][int(idx)]
                self._group_spike_counts[grp] += 1

        self._step_count += 1

        return {
            "n_spikes": int(new_spikes),
            "step": self._step_count,
        }

    def get_state(self) -> bytes:
        S = self._components["S"]
        return pickle.dumps({
            "weights": np.array(S.w[:]).copy(),
            "step_count": self._step_count,
            "group_spike_counts": self._group_spike_counts.copy(),
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._components["S"].w[:] = state["weights"]
        self._step_count = state["step_count"]
        self._group_spike_counts = state["group_spike_counts"]

    def get_structure_metric(self) -> float:
        """Synaptic weight Gini from LIVE network."""
        w = np.array(self._components["S"].w[:])
        if len(w) == 0 or w.sum() < 1e-10:
            return 0.0
        sorted_vals = np.sort(w)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return float((2.0 * (index * sorted_vals).sum() / (n * sorted_vals.sum())) - (n + 1) / n)

    def get_structure_distribution(self) -> dict[str, float]:
        """Per-group weight Gini from LIVE network."""
        S = self._components["S"]
        pre = np.array(S.i[:])
        post = np.array(S.j[:])
        w = np.array(S.w[:])
        group_ids = self._components["group_ids"]

        result = {}
        for g in range(self._n_groups):
            g_neurons = set(i for i, gid in enumerate(group_ids) if gid == g)
            mask = np.array([int(pre[k]) in g_neurons and int(post[k]) in g_neurons for k in range(len(pre))])
            vals = w[mask]
            if len(vals) == 0 or vals.sum() < 1e-10:
                result[f"group_{g}"] = 0.0
                continue
            sv = np.sort(vals)
            n = len(sv)
            idx = np.arange(1, n + 1)
            result[f"group_{g}"] = float((2.0 * (idx * sv).sum() / (n * sv.sum())) - (n + 1) / n)
        return result

    def get_engagement_distribution(self) -> dict[str, float]:
        """Spike rate per neuron group, normalised."""
        total = self._group_spike_counts.sum() or 1.0
        return {f"group_{g}": float(self._group_spike_counts[g] / total) for g in range(self._n_groups)}

    def ablate(self, region_id: str) -> TestSystem:
        """Remove all synapses to/from neurons in target group. Returns new network."""
        new = self._build_clone()
        group_id = int(region_id.replace("group_", ""))
        S = new._components["S"]
        pre = np.array(S.i[:])
        post = np.array(S.j[:])
        group_ids = new._components["group_ids"]
        for k in range(len(pre)):
            if group_ids[int(pre[k])] == group_id or group_ids[int(post[k])] == group_id:
                S.w[k] = 0.0
        return new

    def perturb(self, region_id: str, method: str = "reset_weights") -> TestSystem:
        """Reset synaptic weights within target group to initial values. Returns new network."""
        new = self._build_clone()
        group_id = int(region_id.replace("group_", ""))
        S = new._components["S"]
        pre = np.array(S.i[:])
        post = np.array(S.j[:])
        group_ids = new._components["group_ids"]
        for k in range(len(pre)):
            if group_ids[int(pre[k])] == group_id and group_ids[int(post[k])] == group_id:
                S.w[k] = new._initial_weights[k]
        return new

    def get_regions(self) -> list[str]:
        return [f"group_{g}" for g in range(self._n_groups)]

    def clone(self) -> TestSystem:
        return self._build_clone()

    def _build_clone(self) -> STDPNetwork:
        """Create independent Brian2 network with same weight state."""
        new = STDPNetwork(
            n_neurons=self._n_neurons,
            connection_prob=self._connection_prob,
            seed=self._seed + self._step_count + 7919,
            w_max=self._w_max,
            a_plus=self._a_plus,
            a_minus=self._a_minus,
            n_groups=self._n_groups,
            dt_ms=self._dt_ms,
        )
        # Copy current weights into the new network
        new._components["S"].w[:] = np.array(self._components["S"].w[:]).copy()
        new._initial_weights = self._initial_weights.copy()
        new._step_count = self._step_count
        new._group_spike_counts = self._group_spike_counts.copy()
        new._training = self._training
        return new

    def train_on_domain(self, graph: Any, n_steps: int = 2000) -> None:
        """Train via rate-coded group input with live STDP.

        Cycles through groups, each getting supra-threshold drive for 20 steps.
        STDP is active — weights update based on spike timing.
        """
        _log(f"Training STDP network: {n_steps} steps, {self._n_neurons} neurons (live Brian2)")
        for i in range(n_steps):
            group = (i // 20) % self._n_groups
            self.step(group)
        _log(f"  done: Gini={self.get_structure_metric():.4f}, "
             f"spikes={self._group_spike_counts.sum():.0f}")

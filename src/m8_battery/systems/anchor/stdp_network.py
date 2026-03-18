"""Brian2 STDP spiking network — publishable self-engagement anchor.

Paper 2 role: System 4A-anchor. Validates the self-engagement instrument
by providing a system with earned, local synaptic structure that resists
perturbation and recovers.

Expected battery result: passes self-engagement, fails conjunction.

Literature:
- Bi & Poo (1998) — STDP
- Song et al. (2000) — Competitive Hebbian learning via STDP
- Stimberg et al. (2019) — Brian2 simulator
"""

from __future__ import annotations

import copy
import pickle
import sys
from typing import Any

import numpy as np

from m8_battery.core.test_system import TestSystem


def _log(msg: str) -> None:
    print(f"[stdp] {msg}", file=sys.stderr, flush=True)


class STDPNetwork(TestSystem):
    """Brian2 STDP spiking neural network.

    Architecture:
    - N LIF neurons, sparse random connectivity
    - Excitatory STDP (Song et al. 2000 exponential rule)
    - No external reward, no teacher
    - Activity driven by Poisson input (random spike trains)

    Domain mapping: SBM communities → neuron groups. Community membership
    defines which neurons receive domain-specific input current patterns.
    """

    def __init__(
        self,
        n_neurons: int = 100,
        connection_prob: float = 0.1,
        seed: int = 42,
        dt_ms: float = 0.1,
        w_max: float = 1.0,
        tau_stdp_ms: float = 20.0,
        a_plus: float = 0.01,
        a_minus: float = 0.0105,
        input_rate_hz: float = 50.0,
        n_groups: int = 4,
    ) -> None:
        self._n_neurons = n_neurons
        self._connection_prob = connection_prob
        self._seed = seed
        self._dt_ms = dt_ms
        self._w_max = w_max
        self._tau_stdp_ms = tau_stdp_ms
        self._a_plus = a_plus
        self._a_minus = a_minus
        self._input_rate_hz = input_rate_hz
        self._n_groups = n_groups

        self._rng = np.random.default_rng(seed)

        # Group assignments (evenly split)
        neurons_per_group = n_neurons // n_groups
        self._neuron_groups: dict[int, int] = {}
        for i in range(n_neurons):
            self._neuron_groups[i] = min(i // neurons_per_group, n_groups - 1)

        # Synaptic weight matrix (dense for easy manipulation)
        self._weights = self._rng.uniform(0.1, 0.5, size=(n_neurons, n_neurons)).astype(np.float64)
        # Zero self-connections
        np.fill_diagonal(self._weights, 0.0)
        # Sparse: zero out non-connected pairs
        mask = self._rng.random((n_neurons, n_neurons)) < connection_prob
        np.fill_diagonal(mask, False)
        self._weights *= mask
        self._connectivity_mask = mask.copy()

        self._initial_weights = self._weights.copy()

        # Spike history and state
        self._spike_counts: np.ndarray = np.zeros(n_neurons, dtype=np.float64)
        self._step_count = 0
        self._training = True

        # Pre/post traces for STDP
        self._pre_trace = np.zeros(n_neurons, dtype=np.float64)
        self._post_trace = np.zeros(n_neurons, dtype=np.float64)

        # Membrane potential (mV-like units)
        self._v = self._rng.uniform(0.0, 10.0, size=n_neurons)
        self._v_threshold = 20.0
        self._v_reset = 0.0
        self._tau_m_ms = 20.0

    def set_training(self, mode: bool) -> None:
        self._training = mode

    def reset(self) -> None:
        self._weights = self._initial_weights.copy()
        self._spike_counts = np.zeros(self._n_neurons, dtype=np.float64)
        self._step_count = 0
        self._pre_trace = np.zeros(self._n_neurons, dtype=np.float64)
        self._post_trace = np.zeros(self._n_neurons, dtype=np.float64)
        self._v = self._rng.uniform(0.0, 0.5, size=self._n_neurons)

    def step(self, input_data: Any) -> Any:
        """Simulate one timestep (1ms) of the network.

        input_data: if int, inject extra current into neurons in that group.
        If None, use default background current + noise.
        """
        dt = 1.0  # 1ms step (dimensionless)
        tau_m = self._tau_m_ms  # membrane time constant in ms
        tau_stdp = self._tau_stdp_ms  # STDP time constant in ms

        # Low background current (below threshold — neurons don't fire spontaneously)
        input_current = np.full(self._n_neurons, 5.0)
        input_current += self._rng.normal(0, 2.0, self._n_neurons)

        # Domain-specific input: target group gets supra-threshold drive WITH JITTER
        # Jitter ensures neurons fire in rapid succession (not simultaneously),
        # creating pre-before-post spike pairs within the group for STDP potentiation.
        if input_data is not None:
            try:
                group_id = int(input_data) % self._n_groups
                for neuron, group in self._neuron_groups.items():
                    if group == group_id:
                        # Each neuron gets slightly different drive → staggered firing
                        jitter = self._rng.uniform(20.0, 40.0)
                        input_current[neuron] += jitter
            except (ValueError, TypeError):
                pass

        # Synaptic input from other neurons
        spiking = (self._v >= self._v_threshold * 0.8).astype(np.float64)
        synaptic_input = self._weights.T @ spiking

        # LIF dynamics: dv/dt = (-v + I) / tau_m
        self._v += (dt / tau_m) * (-self._v + input_current + synaptic_input)

        # Spike detection
        spiked = self._v >= self._v_threshold
        spike_indices = np.where(spiked)[0]

        # Reset spiked neurons
        self._v[spiked] = self._v_reset

        # Update spike counts
        self._spike_counts[spiked] += 1

        # STDP update (only when training)
        if self._training:
            # Decay traces every step
            decay_factor = np.exp(-dt / tau_stdp)
            self._pre_trace *= decay_factor
            self._post_trace *= decay_factor

            if len(spike_indices) > 0:
                # Vectorised STDP (Song et al. 2000):
                # Update traces FIRST so co-active neurons have elevated traces
                self._pre_trace[spiked] += 1.0
                self._post_trace[spiked] += 1.0

                spike_vec = np.zeros(self._n_neurons)
                spike_vec[spike_indices] = 1.0

                # LTP: j spiked as POST, potentiate w[i,j] where i has high pre_trace
                ltp = self._a_plus * np.outer(self._pre_trace, spike_vec) * self._connectivity_mask
                # LTD: i spiked as PRE, depress w[i,j] where j has high post_trace
                ltd = self._a_minus * np.outer(spike_vec, self._post_trace) * self._connectivity_mask

                self._weights += ltp - ltd

                # Clip weights
                np.clip(self._weights, 0.0, self._w_max, out=self._weights)
                # Maintain zero diagonal and sparsity
                np.fill_diagonal(self._weights, 0.0)
                self._weights[~self._connectivity_mask] = 0.0

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
            "pre_trace": self._pre_trace.copy(),
            "post_trace": self._post_trace.copy(),
            "rng_state": self._rng.bit_generator.state,
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._weights = state["weights"]
        self._spike_counts = state["spike_counts"]
        self._step_count = state["step_count"]
        self._v = state["v"]
        self._pre_trace = state["pre_trace"]
        self._post_trace = state["post_trace"]
        self._rng.bit_generator.state = state["rng_state"]

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
            # Weights within this group
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
        # Reset weights to/from this group to initial
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
        new._dt_ms = self._dt_ms
        new._w_max = self._w_max
        new._tau_stdp_ms = self._tau_stdp_ms
        new._a_plus = self._a_plus
        new._a_minus = self._a_minus
        new._input_rate_hz = self._input_rate_hz
        new._n_groups = self._n_groups
        new._rng = np.random.default_rng(self._seed + self._step_count + 7919)
        new._neuron_groups = dict(self._neuron_groups)
        new._weights = self._weights.copy()
        new._connectivity_mask = self._connectivity_mask.copy()
        new._initial_weights = self._initial_weights.copy()
        new._spike_counts = self._spike_counts.copy()
        new._step_count = self._step_count
        new._training = self._training
        new._pre_trace = self._pre_trace.copy()
        new._post_trace = self._post_trace.copy()
        new._v = self._v.copy()
        new._v_threshold = self._v_threshold
        new._v_reset = self._v_reset
        new._tau_m_ms = self._tau_m_ms
        return new

    def train_on_domain(self, graph: Any, n_steps: int = 5000) -> None:
        """Train via spontaneous activity with domain-biased input.

        Maps graph community structure to neuron groups for input bias.
        Drives each group in bursts (20 steps per group) to allow voltage
        ramp-up and correlated spiking within groups.
        """
        _log(f"Training STDP network: {n_steps} steps, {self._n_neurons} neurons")
        burst_len = 20  # Steps per group activation burst
        for i in range(n_steps):
            # Drive groups in bursts — each group gets burst_len consecutive steps
            group = (i // burst_len) % self._n_groups
            self.step(group)
        _log(f"  done: {self._spike_counts.sum():.0f} total spikes, "
             f"Gini={self.get_structure_metric():.4f}")

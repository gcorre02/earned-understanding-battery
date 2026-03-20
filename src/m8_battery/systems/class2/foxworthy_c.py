"""System 2C — Foxworthy Variant C (Latent State, Frozen Weights).

A Class 2 system: recurrent model with deterministic state update.
Frozen weights, transient state vector that is resettable.
Source: Foxworthy (2026) section 2.4, 2.7.

Has latent state that changes through operation but no learning
(weights fixed). Structure metric may vary with state but the
system cannot earn new structure.

Expected battery result: some instruments may partially pass,
but developmental trajectory should fail (no earned structure).
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from m8_battery.core.test_system import TestSystem


class FoxworthyC(TestSystem):
    """Recurrent model with frozen weights and resettable latent state.

    Architecture: input → GRU cell → linear output.
    Weights frozen at init. Latent state evolves through operation
    but can be reset. No learning mechanism.
    """

    def __init__(
        self,
        n_features: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 5,
        seed: int = 42,
    ) -> None:
        self._seed = seed
        self._n_features = n_features
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        torch.manual_seed(seed)

        # Frozen GRU cell + output projection
        self._gru = nn.GRUCell(n_features, hidden_dim)
        self._output = nn.Linear(hidden_dim, output_dim)

        # Freeze all weights
        for param in self._gru.parameters():
            param.requires_grad = False
        for param in self._output.parameters():
            param.requires_grad = False

        # Latent state (resettable)
        self._hidden = torch.zeros(1, hidden_dim)
        self._initial_hidden = self._hidden.clone()

        self._graph: Any = None
        self._current_node: int | None = None
        self._step_count = 0
        self._visit_counts: dict[int, int] = {}

    def set_graph(self, graph) -> None:
        """Attach a graph for navigation."""
        self._graph = graph

    def reset(self) -> None:
        self._hidden = self._initial_hidden.clone()
        self._current_node = None
        self._step_count = 0
        self._visit_counts = {}

    def step(self, input_data: Any) -> Any:
        if self._graph is None:
            return {"error": "no graph attached"}

        nodes = list(self._graph.nodes())
        if not nodes:
            return {"error": "empty graph"}

        if self._current_node is None:
            self._current_node = nodes[0]

        # Get node features
        features = self._graph.nodes[self._current_node].get("features", {})
        x = torch.tensor(
            [[features.get(f"feat_{i}", 0.0) for i in range(self._n_features)]],
            dtype=torch.float32,
        )

        # GRU step (frozen weights, state updates)
        with torch.no_grad():
            self._hidden = self._gru(x, self._hidden)
            out = self._output(self._hidden)

        # Navigate based on output
        successors = list(self._graph.successors(self._current_node))
        if successors:
            idx = int(out.argmax().item()) % len(successors)
            self._current_node = successors[idx]

        self._visit_counts[self._current_node] = (
            self._visit_counts.get(self._current_node, 0) + 1
        )
        self._step_count += 1

        return {
            "current_node": self._current_node,
            "step": self._step_count,
            "hidden_norm": float(self._hidden.norm().item()),
        }

    def get_state(self) -> bytes:
        return pickle.dumps({
            "current_node": self._current_node,
            "step_count": self._step_count,
            "visit_counts": self._visit_counts,
            "hidden": self._hidden.numpy(),
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._current_node = state["current_node"]
        self._step_count = state["step_count"]
        self._visit_counts = state["visit_counts"]
        self._hidden = torch.tensor(state["hidden"])

    def get_representation_state(self):
        """GRU hidden state vector for CKA computation (T1-02)."""
        import numpy as np
        return self._hidden.detach().cpu().numpy().reshape(1, -1)

    def get_structure_metric(self) -> float:
        """Hidden state norm — changes with state but weights are fixed.

        This is NOT earned structure — it's transient state that resets.
        The developmental trajectory may show change but it's not
        compression into stable organisation.
        """
        return float(self._hidden.norm().item())

    def get_structure_distribution(self) -> dict[str, float]:
        """Per-hidden-dimension activation distribution."""
        if self._hidden is None:
            return {}
        h = self._hidden.squeeze().detach().numpy()
        n_regions = min(4, len(h))
        chunk_size = len(h) // n_regions
        return {
            f"region_{i}": float(np.abs(h[i * chunk_size:(i + 1) * chunk_size]).mean())
            for i in range(n_regions)
        }

    def get_engagement_distribution(self) -> dict[str, float]:
        if self._graph is None:
            return {}
        communities: dict[int, list[int]] = {}
        for node in self._graph.nodes():
            block = self._graph.nodes[node].get("features", {}).get("community", 0)
            communities.setdefault(block, []).append(node)
        total = sum(self._visit_counts.values()) or 1
        return {
            f"community_{c}": sum(self._visit_counts.get(n, 0) for n in ns) / total
            for c, ns in communities.items()
        }

    def ablate(self, region_id: str) -> TestSystem:
        """Ablation: zero a chunk of hidden dimensions."""
        new = self._copy()
        idx = int(region_id.split("_")[-1])
        chunk_size = self._hidden_dim // 4
        with torch.no_grad():
            start = idx * chunk_size
            end = min(start + chunk_size, self._hidden_dim)
            # Zero the GRU weights for this region
            new._gru.weight_hh[start:end, :] = 0
            new._gru.weight_ih[start:end, :] = 0

        return new

    def perturb(self, region_id: str, method: str = "shuffle_weights") -> TestSystem:
        new = self._copy()
        idx = int(region_id.split("_")[-1])
        chunk_size = self._hidden_dim // 4
        with torch.no_grad():
            start = idx * chunk_size
            end = min(start + chunk_size, self._hidden_dim)
            rng = torch.Generator().manual_seed(self._seed + 999)
            new._gru.weight_hh[start:end, :] = torch.randn(
                end - start, self._hidden_dim, generator=rng
            ) * 0.1

        return new

    def get_regions(self) -> list[str]:
        return [f"region_{i}" for i in range(4)]

    def clone(self) -> TestSystem:
        return self._copy()

    def _copy(self) -> FoxworthyC:
        new = FoxworthyC(
            n_features=self._n_features,
            hidden_dim=self._hidden_dim,
            output_dim=self._output_dim,
            seed=self._seed,
        )
        new._gru.load_state_dict(self._gru.state_dict())
        new._output.load_state_dict(self._output.state_dict())
        new._graph = self._graph
        return new

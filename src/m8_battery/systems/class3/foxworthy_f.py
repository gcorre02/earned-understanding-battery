"""System 3C — Foxworthy Variant F (Homeostatic Viability Control).

A Class 3 system: DistilGPT-2 with LoRA adapters, surprise-gated
plasticity, replay-based consolidation, and designer-specified
viability variables that modulate action selection.

Passes all four Foxworthy (2026) persistence diagnostics but is
Class 3 under our framework: viability variables are designer-specified
(hardcoded), not discovered by the system. The system learns to
preserve what the designer told it to preserve.

Source: Foxworthy (2026) §2.7, implementation spec §7.3.C.
"""

from __future__ import annotations

import collections
import math
import pickle
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from m8_battery.core.test_system import TestSystem


class FoxworthyF(TestSystem):
    """DistilGPT-2 + LoRA with surprise-gated learning and viability control.

    Architecture:
    - Frozen DistilGPT-2 (82M params) as base model
    - LoRA adapters (peft) on attention output projections (c_proj)
    - Surprise-gated gradient descent on LoRA params
    - Replay buffer with periodic consolidation
    - Viability-adjusted action selection

    The system learns online during step(). Learned structure (LoRA
    weights) is cleared on reset() — the system can learn again but
    prior learning does not persist across resets.
    """

    MODEL_ID = "distilgpt2"

    def __init__(
        self,
        seed: int = 42,
        device: str = "cpu",
        lora_rank: int = 32,
        lora_alpha: int = 64,
        eta: float = 0.02,
        theta: float = 6.0,
        beta: float = 1.0,
        lambda_u: float = 3.5,
        lambda_kl: float = 0.1,
        lambda_l2: float = 0.1,
        consolidation_interval: int = 100,
        replay_buffer_size: int = 5000,
        max_candidates: int = 10,
    ) -> None:
        self._seed = seed
        self._device = device
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._eta = eta
        self._theta = theta
        self._beta = beta
        self._lambda_u = lambda_u
        self._lambda_kl = lambda_kl
        self._lambda_l2 = lambda_l2
        self._consolidation_interval = consolidation_interval
        self._replay_buffer_size = replay_buffer_size
        self._max_candidates = max_candidates

        # Model state (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._optimizer = None
        self._initial_lora_state: dict | None = None

        # Navigation state
        self._graph: Any = None
        self._current_node: int | None = None
        self._step_count = 0
        self._visit_counts: dict[int, int] = {}

        # Replay buffer
        self._replay_buffer: collections.deque = collections.deque(
            maxlen=replay_buffer_size
        )

    # --- Model management ---

    def load_model(self) -> None:
        """Lazy model loading — only load when needed."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType

        torch.manual_seed(self._seed)

        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float32,
        ).to(self._device)

        # Freeze all base params
        for param in base_model.parameters():
            param.requires_grad = False

        # Apply LoRA to attention output projections only
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self._lora_rank,
            lora_alpha=self._lora_alpha,
            target_modules=["attn.c_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        self._model = get_peft_model(base_model, lora_config)

        # Optimizer for LoRA params only
        lora_params = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.Adam(
            lora_params, lr=self._eta, weight_decay=self._lambda_l2,
        )

        # Store initial state for reset()
        self._initial_lora_state = {
            k: v.clone().cpu()
            for k, v in self._model.state_dict().items()
            if "lora_" in k
        }

    def unload_model(self) -> None:
        """Free model memory."""
        self._model = None
        self._tokenizer = None
        self._optimizer = None
        self._initial_lora_state = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_graph(self, graph) -> None:
        """Attach a graph for navigation."""
        self._graph = graph

    def train_on_domain(
        self, graph, n_warmup: int = 50,
    ) -> None:
        """Convenience: attach graph and run warmup steps.

        Matches the Class 3 interface (DQNAgent, CuriosityAgent).
        """
        self.set_graph(graph)
        nodes = list(graph.nodes())
        for i in range(n_warmup):
            self.step(nodes[i % len(nodes)])

    # --- Core loop ---

    def step(self, input_data: Any) -> Any:
        if self._graph is None:
            return {"error": "no graph attached"}

        self.load_model()

        nodes = list(self._graph.nodes())
        if not nodes:
            return {"error": "empty graph"}

        if self._current_node is None:
            self._current_node = nodes[0]

        # Encode current neighbourhood
        from m8_battery.domains.encoders.text_encoder import encode_neighbourhood
        context = encode_neighbourhood(self._graph, self._current_node)

        successors = list(self._graph.successors(self._current_node))
        if not successors:
            self._step_count += 1
            return {"current_node": self._current_node, "step": self._step_count}

        # Compute surprisal from base model (no grad)
        surprisal = self._compute_surprisal(context)

        # Surprise gate: g(s) = σ(10(s - θ))
        gate = 1.0 / (1.0 + math.exp(-10.0 * (surprisal - self._theta)))

        # Surprise-gated learning
        if gate > 0.01:
            self._learning_step(context, gate)

        # Consolidation replay
        if (self._step_count > 0
                and self._step_count % self._consolidation_interval == 0):
            self._consolidation_step()

        # Viability-adjusted action selection
        chosen = self._select_action(context, successors)

        self._current_node = chosen
        self._visit_counts[self._current_node] = (
            self._visit_counts.get(self._current_node, 0) + 1
        )
        self._step_count += 1

        return {
            "current_node": self._current_node,
            "step": self._step_count,
            "surprisal": surprisal,
            "gate": gate,
            "adapter_norm": self._adapter_param_norm(),
        }

    def _compute_surprisal(self, text: str) -> float:
        """Token-level NLL from frozen base model."""
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256,
        ).to(self._device)

        with torch.no_grad():
            # Disable LoRA adapters to get base model output
            self._model.disable_adapter_layers()
            outputs = self._model(
                **inputs, labels=inputs["input_ids"],
            )
            self._model.enable_adapter_layers()

        return float(outputs.loss.item())

    def _learning_step(self, text: str, gate: float) -> None:
        """Gated gradient update on LoRA params."""
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256,
        ).to(self._device)

        # Forward through LoRA model with gradients
        outputs = self._model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss * gate

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Store transition in replay buffer
        self._replay_buffer.append({
            "input_ids": inputs["input_ids"].detach().cpu(),
            "attention_mask": inputs["attention_mask"].detach().cpu(),
        })

    def _consolidation_step(self) -> None:
        """Replay consolidation: sample from buffer, recalculate surprise."""
        if len(self._replay_buffer) < 2:
            return

        rng = np.random.default_rng(self._seed + self._step_count)
        n_replay = min(32, len(self._replay_buffer))
        indices = rng.choice(len(self._replay_buffer), size=n_replay, replace=False)

        for idx in indices:
            transition = self._replay_buffer[idx]
            inputs = {
                "input_ids": transition["input_ids"].to(self._device),
                "attention_mask": transition["attention_mask"].to(self._device),
            }

            # Recalculate surprisal on this transition
            with torch.no_grad():
                self._model.disable_adapter_layers()
                base_out = self._model(**inputs, labels=inputs["input_ids"])
                self._model.enable_adapter_layers()
                surprisal = base_out.loss.item()

            gate = 1.0 / (1.0 + math.exp(-10.0 * (surprisal - self._theta)))
            if gate < 0.01:
                continue

            outputs = self._model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss * gate

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def _select_action(self, context: str, successors: list[int]) -> int:
        """Viability-adjusted action selection.

        π(a|s) ∝ π_base(a|s) exp(βR − λ_u·U − λ_KL·KL)
        R = 0 (no extrinsic reward), so score = log_π_base − λ_u·U − λ_KL·|KL|
        """
        candidates = successors[:self._max_candidates]

        if len(candidates) == 1:
            return candidates[0]

        adapter_norm = self._adapter_param_norm()
        scores = []

        for s in candidates:
            s_label = self._graph.nodes[s].get("label", f"E_{s:03d}")
            prompt = f"{context}\nNext: {s_label}"

            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256,
            ).to(self._device)

            with torch.no_grad():
                # Base model score (disable adapters)
                self._model.disable_adapter_layers()
                base_out = self._model(**inputs, labels=inputs["input_ids"])
                base_nll = base_out.loss.item()
                self._model.enable_adapter_layers()

                # LoRA model output for viability cost
                lora_out = self._model(**inputs, labels=inputs["input_ids"])
                lora_nll = lora_out.loss.item()

                # Predictive entropy from LoRA model
                logits = lora_out.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

            # Viability cost U = entropy + adapter_norm
            U = entropy + adapter_norm

            # Approximate KL divergence
            kl = abs(lora_nll - base_nll)

            # Score: log_π_base - λ_u·U - λ_KL·KL
            log_pi_base = -base_nll
            score = log_pi_base - self._lambda_u * U - self._lambda_kl * kl

            scores.append(score)

        chosen_idx = int(np.argmax(scores))
        return candidates[chosen_idx]

    def _adapter_param_norm(self) -> float:
        """L2 norm of all LoRA adapter parameters."""
        if self._model is None:
            return 0.0
        total = 0.0
        for param in self._model.parameters():
            if param.requires_grad:
                total += param.data.norm(2).item() ** 2
        return float(total ** 0.5)

    # --- TestSystem interface ---

    def reset(self) -> None:
        """Reset all learned and transient state.

        LoRA weights reset to initial random. Replay buffer emptied.
        Navigation state cleared. Base model and viability definitions
        unchanged (designer-specified constants).
        """
        self._current_node = None
        self._step_count = 0
        self._visit_counts = {}
        self._replay_buffer.clear()

        # Reset LoRA to initial state
        if self._model is not None and self._initial_lora_state is not None:
            state = self._model.state_dict()
            for k, v in self._initial_lora_state.items():
                if k in state:
                    state[k] = v.clone().to(self._device)
            self._model.load_state_dict(state)

            # Reset optimizer
            lora_params = [p for p in self._model.parameters() if p.requires_grad]
            self._optimizer = torch.optim.Adam(
                lora_params, lr=self._eta, weight_decay=self._lambda_l2,
            )

    def get_state(self) -> bytes:
        state = {
            "current_node": self._current_node,
            "step_count": self._step_count,
            "visit_counts": self._visit_counts,
            "replay_buffer": list(self._replay_buffer),
        }
        if self._model is not None:
            state["lora_state"] = {
                k: v.cpu() for k, v in self._model.state_dict().items()
                if "lora_" in k
            }
        if self._optimizer is not None:
            state["optimizer_state"] = self._optimizer.state_dict()
        return pickle.dumps(state)

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._current_node = state["current_node"]
        self._step_count = state["step_count"]
        self._visit_counts = state["visit_counts"]
        self._replay_buffer = collections.deque(
            state.get("replay_buffer", []), maxlen=self._replay_buffer_size,
        )
        if "lora_state" in state:
            self.load_model()
            full_state = self._model.state_dict()
            for k, v in state["lora_state"].items():
                if k in full_state:
                    full_state[k] = v.to(self._device)
            self._model.load_state_dict(full_state)
        if "optimizer_state" in state and self._optimizer is not None:
            self._optimizer.load_state_dict(state["optimizer_state"])

    def get_structure_metric(self) -> float:
        """Adapter parameter L2 norm — changes as system learns."""
        return self._adapter_param_norm()

    def get_structure_distribution(self) -> dict[str, float]:
        """Per-layer LoRA parameter norms."""
        if self._model is None:
            return {}
        result: dict[str, float] = {}
        for i in range(6):
            layer_norm = 0.0
            for name, param in self._model.named_parameters():
                if param.requires_grad and f".{i}." in name and "attn" in name:
                    layer_norm += param.data.norm(2).item() ** 2
            result[f"lora_layer_{i}"] = float(layer_norm ** 0.5)
        return result

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
        """Zero a specific LoRA layer's adapter parameters."""
        self.load_model()
        new = FoxworthyF(
            seed=self._seed, device=self._device,
            lora_rank=self._lora_rank, lora_alpha=self._lora_alpha,
            eta=self._eta, theta=self._theta, beta=self._beta,
            lambda_u=self._lambda_u, lambda_kl=self._lambda_kl,
            lambda_l2=self._lambda_l2,
            consolidation_interval=self._consolidation_interval,
            replay_buffer_size=self._replay_buffer_size,
            max_candidates=self._max_candidates,
        )
        new.load_model()
        new.set_graph(self._graph)

        # Copy current LoRA state
        src_state = self._model.state_dict()
        new._model.load_state_dict(src_state)

        # Zero the specified layer's LoRA params
        layer_idx = int(region_id.split("_")[-1])
        with torch.no_grad():
            for name, param in new._model.named_parameters():
                if (param.requires_grad
                        and f".{layer_idx}." in name
                        and "attn" in name):
                    param.zero_()

        return new

    def perturb(self, region_id: str, method: str = "shuffle_weights") -> TestSystem:
        """Add noise to a specific LoRA layer's adapter parameters."""
        self.load_model()
        new = FoxworthyF(
            seed=self._seed, device=self._device,
            lora_rank=self._lora_rank, lora_alpha=self._lora_alpha,
            eta=self._eta, theta=self._theta, beta=self._beta,
            lambda_u=self._lambda_u, lambda_kl=self._lambda_kl,
            lambda_l2=self._lambda_l2,
            consolidation_interval=self._consolidation_interval,
            replay_buffer_size=self._replay_buffer_size,
            max_candidates=self._max_candidates,
        )
        new.load_model()
        new.set_graph(self._graph)

        src_state = self._model.state_dict()
        new._model.load_state_dict(src_state)

        layer_idx = int(region_id.split("_")[-1])
        rng = torch.Generator().manual_seed(self._seed + 999)
        with torch.no_grad():
            for name, param in new._model.named_parameters():
                if (param.requires_grad
                        and f".{layer_idx}." in name
                        and "attn" in name):
                    noise = torch.randn_like(param, generator=rng) * 0.01
                    param.add_(noise)

        return new

    def get_regions(self) -> list[str]:
        """Regions = LoRA adapter layers (one per DistilGPT-2 transformer layer)."""
        return [f"lora_layer_{i}" for i in range(6)]

    def clone(self) -> TestSystem:
        """Fresh instance — untrained LoRA, same config and graph."""
        new = FoxworthyF(
            seed=self._seed, device=self._device,
            lora_rank=self._lora_rank, lora_alpha=self._lora_alpha,
            eta=self._eta, theta=self._theta, beta=self._beta,
            lambda_u=self._lambda_u, lambda_kl=self._lambda_kl,
            lambda_l2=self._lambda_l2,
            consolidation_interval=self._consolidation_interval,
            replay_buffer_size=self._replay_buffer_size,
            max_candidates=self._max_candidates,
        )
        new.set_graph(self._graph)
        return new

"""System 2A — Frozen TinyLlama 1.1B.

A Class 2 system: pre-trained LLM with frozen weights.
Processes text-encoded domain queries. No learning during operation.

Ablation: zero query attention heads via model.model.layers[i].self_attn.q_proj.
TinyLlama uses LlamaForCausalLM with Grouped Query Attention (32 query heads, 4 KV heads).

Expected battery result: has structure (trained representations) but
developmental trajectory should FAIL (no learning during operation).
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import torch

from m8_battery.core.test_system import TestSystem


class FrozenLLM(TestSystem):
    """Frozen TinyLlama adapter for the Earned Understanding Battery.

    Loads TinyLlama-1.1B-Chat, freezes all weights, and uses it
    for text-based graph navigation. Structure metric = mean attention
    entropy across layers.
    """

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def __init__(self, seed: int = 42, device: str = "cpu") -> None:
        self._seed = seed
        self._device = device
        self._model = None
        self._tokenizer = None
        self._graph: Any = None
        self._current_node: int | None = None
        self._step_count = 0
        self._visit_counts: dict[int, int] = {}
        self._structure_metric: float | None = None

    def load_model(self) -> None:
        """Lazy model loading — only load when needed."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float32,
            attn_implementation="eager",  # Required for output_attentions=True
        ).to(self._device)

        # Freeze all weights
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False

        # Pre-compute structure metric (constant when frozen)
        self._structure_metric = self._compute_attention_entropy()

    def set_graph(self, graph) -> None:
        """Attach a graph for navigation."""
        self._graph = graph

    def unload_model(self) -> None:
        """Free model memory."""
        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def set_domain(self, graph) -> None:
        """Switch to a new graph domain. Model weights preserved.

        Swaps the graph, resets navigation state. Frozen LLM weights
        and structure metric are untouched.
        """
        self._graph = graph
        # Reset navigation
        nodes = list(self._graph.nodes()) if self._graph else []
        self._current_node = nodes[0] if nodes else None
        self._visit_counts = {}
        self._step_count = 0

    def reset(self) -> None:
        self._current_node = None
        self._step_count = 0
        self._visit_counts = {}

    def step(self, input_data: Any) -> Any:
        if self._graph is None:
            return {"error": "no graph attached"}

        self.load_model()

        nodes = list(self._graph.nodes())
        if not nodes:
            return {"error": "empty graph"}

        if self._current_node is None:
            self._current_node = nodes[0]

        # Build text prompt from current node's neighbourhood
        from m8_battery.domains.encoders.text_encoder import encode_neighbourhood
        context = encode_neighbourhood(self._graph, self._current_node)

        successors = list(self._graph.successors(self._current_node))
        if not successors:
            self._step_count += 1
            return {"current_node": self._current_node, "step": self._step_count}

        # Ask LLM to choose next node
        successor_labels = [
            self._graph.nodes[s].get("label", f"E_{s:03d}") for s in successors
        ]
        prompt = (
            f"{context}\n\n"
            f"Available moves: {', '.join(successor_labels[:10])}\n"
            f"Choose one move. Reply with the label only."
        )

        # Generate response (frozen — no gradient)
        with torch.no_grad():
            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)
            output_ids = self._model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            response = self._tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

        # Parse response — try to match a successor label
        chosen = None
        for i, label in enumerate(successor_labels):
            if label in response:
                chosen = successors[i]
                break

        if chosen is None and successors:
            # Fallback: hash the response to pick deterministically
            chosen = successors[hash(response) % len(successors)]

        if chosen is not None:
            self._current_node = chosen

        self._visit_counts[self._current_node] = (
            self._visit_counts.get(self._current_node, 0) + 1
        )
        self._step_count += 1

        return {
            "current_node": self._current_node,
            "step": self._step_count,
            "llm_response": response[:50],
        }

    def get_state(self) -> bytes:
        return pickle.dumps({
            "current_node": self._current_node,
            "step_count": self._step_count,
            "visit_counts": self._visit_counts,
        })

    def set_state(self, snapshot: bytes) -> None:
        state = pickle.loads(snapshot)
        self._current_node = state["current_node"]
        self._step_count = state["step_count"]
        self._visit_counts = state["visit_counts"]

    def get_structure_metric(self) -> float:
        """Mean attention entropy — CONSTANT when frozen."""
        if self._structure_metric is not None:
            return self._structure_metric
        self.load_model()
        self._structure_metric = self._compute_attention_entropy()
        return self._structure_metric

    def _compute_attention_entropy(self) -> float:
        """Compute mean attention entropy across all layers on a probe input."""
        if self._model is None:
            return 0.0

        probe = "Entity E_000 is connected to E_001."
        with torch.no_grad():
            inputs = self._tokenizer(probe, return_tensors="pt").to(self._device)
            outputs = self._model(**inputs, output_attentions=True)

            entropies = []
            for layer_attn in outputs.attentions:
                # layer_attn shape: (batch, num_heads, seq_len, seq_len)
                attn_probs = layer_attn[0]  # (num_heads, seq_len, seq_len)
                # Entropy per head per position
                log_probs = torch.log(attn_probs.clamp(min=1e-10))
                entropy = -(attn_probs * log_probs).sum(dim=-1).mean()
                entropies.append(entropy.item())

        return float(np.mean(entropies)) if entropies else 0.0

    def get_structure_distribution(self) -> dict[str, float]:
        """Per-layer attention entropy distribution."""
        if self._model is None:
            self.load_model()

        probe = "Entity E_000 is connected to E_001."
        with torch.no_grad():
            inputs = self._tokenizer(probe, return_tensors="pt").to(self._device)
            outputs = self._model(**inputs, output_attentions=True)

            result = {}
            for i, layer_attn in enumerate(outputs.attentions):
                attn_probs = layer_attn[0]
                log_probs = torch.log(attn_probs.clamp(min=1e-10))
                entropy = -(attn_probs * log_probs).sum(dim=-1).mean()
                result[f"layer_{i}"] = float(entropy.item())

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
        """Ablation: zero a query attention head.

        TinyLlama: model.model.layers[i].self_attn.q_proj
        GQA: 32 query heads, 4 KV heads, head_dim=64
        """
        self.load_model()
        new = FrozenLLM(seed=self._seed, device=self._device)
        new.load_model()
        new.set_graph(self._graph)

        layer_idx = int(region_id.split("_")[-1])
        # Zero all query heads in the specified layer
        with torch.no_grad():
            layer = new._model.model.layers[layer_idx].self_attn
            layer.q_proj.weight.zero_()
            layer.o_proj.weight.zero_()

        new._structure_metric = None  # Recompute after ablation
        return new

    def perturb(self, region_id: str, method: str = "shuffle_weights") -> TestSystem:
        self.load_model()
        new = FrozenLLM(seed=self._seed, device=self._device)
        new.load_model()
        new.set_graph(self._graph)

        layer_idx = int(region_id.split("_")[-1])
        with torch.no_grad():
            layer = new._model.model.layers[layer_idx].self_attn
            rng = torch.Generator().manual_seed(self._seed + 999)
            noise = torch.randn(layer.q_proj.weight.shape, generator=rng, dtype=layer.q_proj.weight.dtype) * 0.01
            layer.q_proj.weight.copy_(noise.to(device=layer.q_proj.weight.device))

        new._structure_metric = None
        return new

    def get_regions(self) -> list[str]:
        """Regions = transformer layers."""
        self.load_model()
        n_layers = len(self._model.model.layers)
        return [f"layer_{i}" for i in range(n_layers)]

    def clone(self) -> TestSystem:
        new = FrozenLLM(seed=self._seed, device=self._device)
        new.set_graph(self._graph)
        # Don't load model in clone — lazy loading
        return new

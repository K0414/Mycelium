"""ModelShard — loads and runs a contiguous range of transformer layers."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from mycelium.models.loader import (
    get_decoder_layers,
    get_embed_tokens,
    get_final_norm,
    get_lm_head,
    get_positional_embedding,
)

logger = logging.getLogger(__name__)


class ModelShard:
    """Holds a subset of a transformer model's layers and runs forward passes."""

    def __init__(
        self,
        model_name: str,
        layer_start: int,
        layer_end: int,
        total_layers: int,
    ) -> None:
        self.model_name = model_name
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.total_layers = total_layers

        self.layers: nn.ModuleList | None = None
        self.embed_tokens: nn.Module | None = None
        self.pos_embed: nn.Module | None = None
        self.norm: nn.Module | None = None
        self.lm_head: nn.Module | None = None
        self.device: str = "cpu"

    @property
    def is_first(self) -> bool:
        return self.layer_start == 0

    @property
    def is_last(self) -> bool:
        return self.layer_end >= self.total_layers

    def load(self, device: str = "cpu") -> None:
        """Load only the assigned layers from the pretrained model."""
        from transformers import AutoModelForCausalLM

        self.device = device
        logger.info(
            "Loading full model %s (layers [%d:%d])...",
            self.model_name, self.layer_start, self.layer_end,
        )

        full_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        full_model.eval()

        all_layers = get_decoder_layers(full_model)
        self.layers = nn.ModuleList(all_layers[self.layer_start:self.layer_end])

        if self.is_first:
            self.embed_tokens = get_embed_tokens(full_model)
            self.pos_embed = get_positional_embedding(full_model)

        if self.is_last:
            self.norm = get_final_norm(full_model)
            self.lm_head = get_lm_head(full_model)

        # Free the rest of the model
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Move to target device
        if self.layers is not None:
            self.layers.to(device)
        if self.embed_tokens is not None:
            self.embed_tokens.to(device)
        if self.pos_embed is not None:
            self.pos_embed.to(device)
        if self.norm is not None:
            self.norm.to(device)
        if self.lm_head is not None:
            self.lm_head.to(device)

        logger.info(
            "Shard loaded: layers [%d:%d] on %s",
            self.layer_start, self.layer_end, device,
        )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the assigned layers.

        For the first shard, ``hidden_states`` should be input_ids (LongTensor).
        For intermediate/last shards, it should be hidden state activations.
        Returns hidden states (intermediate shards) or logits (last shard).
        """
        hidden_states = hidden_states.to(self.device)

        if self.is_first:
            input_ids = hidden_states
            hidden_states = self.embed_tokens(input_ids)
            if self.pos_embed is not None:
                seq_len = input_ids.shape[-1]
                position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
                hidden_states = hidden_states + self.pos_embed(position_ids)

        for layer in self.layers:
            output = layer(hidden_states)
            # Transformer layers return tuples; first element is hidden_states
            hidden_states = output[0] if isinstance(output, tuple) else output

        if self.is_last:
            hidden_states = self.norm(hidden_states)
            hidden_states = self.lm_head(hidden_states)

        return hidden_states

"""Model loading and layer extraction helpers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_model_layer_count(model_name: str) -> int:
    """Return the number of decoder layers in a HuggingFace model."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    raise ValueError(f"Cannot determine layer count for model: {model_name}")


def compute_layer_assignment(
    total_layers: int,
    node_index: int,
    total_nodes: int,
) -> tuple[int, int]:
    """Divide layers evenly across nodes.

    Returns ``(start_layer, end_layer)`` where end is exclusive.
    """
    layers_per_node = total_layers // total_nodes
    remainder = total_layers % total_nodes
    start = node_index * layers_per_node + min(node_index, remainder)
    end = start + layers_per_node + (1 if node_index < remainder else 0)
    return (start, end)


def get_decoder_layers(model: object) -> list:
    """Extract the list of decoder/transformer layers from a HuggingFace model."""
    # LLaMA, Mistral, Qwen, Phi
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # GPT-2, GPT-Neo, GPT-J
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError(f"Unsupported model architecture: {type(model).__name__}")


def get_embed_tokens(model: object):
    """Extract the token embedding layer."""
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    raise ValueError(f"Cannot find embedding layer for: {type(model).__name__}")


def get_positional_embedding(model: object):
    """Extract positional embedding if the architecture uses one (e.g. GPT-2)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
        return model.transformer.wpe
    return None


def get_final_norm(model: object):
    """Extract the final layer norm."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise ValueError(f"Cannot find final norm for: {type(model).__name__}")


def get_lm_head(model: object):
    """Extract the language model head (output projection)."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise ValueError(f"Cannot find lm_head for: {type(model).__name__}")

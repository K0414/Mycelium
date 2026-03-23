"""Tests for model shard loading and forward pass."""

import torch


def test_model_shard_load_full():
    """Load GPT-2 as a single shard covering all layers."""
    from mycelium.inference.shard import ModelShard
    from mycelium.models.loader import get_model_layer_count

    total = get_model_layer_count("gpt2")
    assert total == 12

    shard = ModelShard("gpt2", 0, total, total)
    shard.load()

    assert shard.is_first
    assert shard.is_last
    assert len(shard.layers) == 12
    assert shard.embed_tokens is not None
    assert shard.lm_head is not None


def test_model_shard_split_forward():
    """Split GPT-2 into two shards and verify combined output matches full model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mycelium.inference.shard import ModelShard
    from mycelium.models.loader import get_model_layer_count

    model_name = "gpt2"
    total = get_model_layer_count(model_name)

    # Load two shards
    shard_a = ModelShard(model_name, 0, 6, total)
    shard_a.load()
    shard_b = ModelShard(model_name, 6, 12, total)
    shard_b.load()

    # Run through the split pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode("Hello world", return_tensors="pt")

    with torch.no_grad():
        hidden = shard_a.forward(input_ids)
        logits_split = shard_b.forward(hidden)

    # Compare with full model
    full_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    )
    full_model.eval()
    with torch.no_grad():
        logits_full = full_model(input_ids).logits

    # The split pipeline should produce the same output
    assert logits_split.shape == logits_full.shape
    assert torch.allclose(logits_split, logits_full, atol=1e-4), (
        f"Max diff: {(logits_split - logits_full).abs().max().item()}"
    )


def test_layer_assignment():
    """Test even division of layers across nodes."""
    from mycelium.models.loader import compute_layer_assignment

    # 12 layers, 2 nodes
    assert compute_layer_assignment(12, 0, 2) == (0, 6)
    assert compute_layer_assignment(12, 1, 2) == (6, 12)

    # 12 layers, 3 nodes
    assert compute_layer_assignment(12, 0, 3) == (0, 4)
    assert compute_layer_assignment(12, 1, 3) == (4, 8)
    assert compute_layer_assignment(12, 2, 3) == (8, 12)

    # 7 layers, 3 nodes (uneven)
    assert compute_layer_assignment(7, 0, 3) == (0, 3)
    assert compute_layer_assignment(7, 1, 3) == (3, 5)
    assert compute_layer_assignment(7, 2, 3) == (5, 7)

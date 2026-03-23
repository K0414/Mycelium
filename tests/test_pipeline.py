"""End-to-end inference pipeline test with GPT-2."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mycelium.inference.shard import ModelShard
from mycelium.models.loader import get_model_layer_count
from mycelium.utils.serialization import deserialize_activation, serialize_activation


def test_full_pipeline_two_shards():
    """Simulate a two-node pipeline: shard_a -> serialize -> deserialize -> shard_b."""
    model_name = "gpt2"
    total = get_model_layer_count(model_name)

    shard_a = ModelShard(model_name, 0, 6, total)
    shard_a.load()
    shard_b = ModelShard(model_name, 6, 12, total)
    shard_b.load()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode("The quick brown fox", return_tensors="pt")

    # Shard A processes input
    with torch.no_grad():
        hidden_a = shard_a.forward(input_ids)

    # Simulate network transfer
    request_id = "test-pipeline-001"
    wire_data = serialize_activation(request_id, hidden_a)
    recv_id, hidden_recv = deserialize_activation(wire_data)
    assert recv_id == request_id

    # Shard B processes received hidden states
    with torch.no_grad():
        logits = shard_b.forward(hidden_recv)

    # Get predicted next token
    next_token_id = logits[0, -1].argmax().item()
    next_token = tokenizer.decode([next_token_id])
    assert isinstance(next_token, str)
    assert len(next_token) > 0

    # Verify matches full model
    full_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    )
    full_model.eval()
    with torch.no_grad():
        full_logits = full_model(input_ids).logits
    full_next = full_logits[0, -1].argmax().item()
    assert next_token_id == full_next, (
        f"Split pipeline predicted token {next_token_id} ({next_token!r}), "
        f"full model predicted {full_next} ({tokenizer.decode([full_next])!r})"
    )

"""Tests for tensor serialization round-trip."""

import torch

from mycelium.utils.serialization import deserialize_activation, serialize_activation


def test_serialize_roundtrip_float32():
    tensor = torch.randn(1, 10, 768)
    request_id = "test-123"

    data = serialize_activation(request_id, tensor)
    out_id, out_tensor = deserialize_activation(data)

    assert out_id == request_id
    assert out_tensor.shape == tensor.shape
    assert torch.allclose(tensor, out_tensor, atol=1e-6)


def test_serialize_roundtrip_float16():
    tensor = torch.randn(2, 5, 128).half()
    request_id = "test-fp16"

    data = serialize_activation(request_id, tensor)
    out_id, out_tensor = deserialize_activation(data)

    assert out_id == request_id
    assert out_tensor.dtype == torch.float16
    assert torch.allclose(tensor, out_tensor, atol=1e-3)


def test_serialize_1d_tensor():
    tensor = torch.tensor([1, 2, 3], dtype=torch.long)
    request_id = "test-1d"

    data = serialize_activation(request_id, tensor)
    out_id, out_tensor = deserialize_activation(data)

    assert out_id == request_id
    assert torch.equal(tensor, out_tensor)

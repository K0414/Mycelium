"""Tensor serialization for wire transfer between nodes."""

from __future__ import annotations

import struct

import msgpack
import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save


def serialize_activation(
    request_id: str, tensor: torch.Tensor,
) -> bytes:
    """Pack request metadata and tensor data into bytes.

    Wire format: [4-byte header len][msgpack header][safetensors data]
    """
    header = msgpack.packb({
        "request_id": request_id,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).split(".")[-1],
    })

    tensor_dict = {"activation": tensor.contiguous()}
    tensor_bytes = st_save(tensor_dict)

    return struct.pack(">I", len(header)) + header + tensor_bytes


def deserialize_activation(
    data: bytes,
) -> tuple[str, torch.Tensor]:
    """Unpack wire data back into request_id and tensor."""
    header_len = struct.unpack(">I", data[:4])[0]
    header = msgpack.unpackb(data[4:4 + header_len], raw=False)
    tensor_bytes = data[4 + header_len:]

    tensors = st_load(tensor_bytes)
    tensor = tensors["activation"]

    return header["request_id"], tensor

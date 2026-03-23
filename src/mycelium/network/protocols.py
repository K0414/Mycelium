"""Protocol IDs and network constants."""

from __future__ import annotations

from libp2p.custom_types import TProtocol

INFERENCE_PROTOCOL = TProtocol("/mycelium/inference/1.0.0")
HEALTH_PROTOCOL = TProtocol("/mycelium/health/1.0.0")

GOSSIP_TOPIC = "mycelium/network/v1"
GOSSIPSUB_PROTOCOL_ID = TProtocol("/meshsub/1.0.0")

DHT_NAMESPACE = "mycelium"

MAX_READ_LEN = 2**32 - 1

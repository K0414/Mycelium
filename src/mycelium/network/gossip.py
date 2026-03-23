"""GossipSub-based network announcements."""

from __future__ import annotations

import logging
from enum import Enum

import msgpack
from libp2p.pubsub.pubsub import Pubsub

from mycelium.network.protocols import GOSSIP_TOPIC

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    NODE_JOINED = "node_joined"
    NODE_LEAVING = "node_leaving"
    SHARD_AVAILABLE = "shard_available"
    SHARD_UNAVAILABLE = "shard_unavailable"


async def publish_announcement(
    pubsub: Pubsub,
    msg_type: MessageType,
    peer_id: str,
    **kwargs: object,
) -> None:
    """Publish a network announcement via GossipSub."""
    payload = {
        "type": msg_type.value,
        "peer_id": peer_id,
        **kwargs,
    }
    await pubsub.publish(GOSSIP_TOPIC, msgpack.packb(payload))
    logger.info("Published %s announcement", msg_type.value)


def decode_announcement(data: bytes) -> dict:
    """Decode a gossip announcement message."""
    return msgpack.unpackb(data, raw=False)

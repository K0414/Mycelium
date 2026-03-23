"""DHT-based peer and model shard discovery."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import msgpack
from libp2p.kad_dht.kad_dht import KadDHT
from libp2p.records.validator import Validator

from mycelium.network.protocols import DHT_NAMESPACE

logger = logging.getLogger(__name__)


@dataclass
class ShardInfo:
    """Describes a model shard hosted by a peer."""

    peer_id: str
    model_name: str
    layer_start: int
    layer_end: int


class MyceliumValidator(Validator):
    """Validator for Mycelium DHT records."""

    async def validate(self, key: str, value: bytes) -> None:
        if not value:
            raise ValueError("Empty value")

    async def select(self, key: str, values: list[bytes]) -> int:
        return 0


def _shard_key(model_name: str, peer_id: str) -> str:
    """Build the DHT key for a model shard record."""
    return f"/{DHT_NAMESPACE}/{model_name}/{peer_id}"


async def register_model_shard(
    dht: KadDHT,
    peer_id: str,
    model_name: str,
    layer_start: int,
    layer_end: int,
) -> None:
    """Advertise a model shard on the DHT."""
    info = {
        "peer_id": peer_id,
        "model": model_name,
        "layer_start": layer_start,
        "layer_end": layer_end,
    }
    key = _shard_key(model_name, peer_id)
    await dht.put_value(key, msgpack.packb(info))
    await dht.provide(model_name)
    logger.info(
        "Registered shard %s layers [%d:%d]",
        model_name, layer_start, layer_end,
    )


async def find_model_shards(dht: KadDHT, model_name: str) -> list[ShardInfo]:
    """Discover all shards for a model from the DHT."""
    providers = await dht.find_providers(model_name)
    if not providers:
        return []

    shards: list[ShardInfo] = []
    for provider in providers:
        pid = provider.peer_id.to_string()
        key = _shard_key(model_name, pid)
        try:
            raw = await dht.get_value(key)
            if raw:
                info = msgpack.unpackb(raw, raw=False)
                shards.append(ShardInfo(
                    peer_id=info["peer_id"],
                    model_name=info["model"],
                    layer_start=info["layer_start"],
                    layer_end=info["layer_end"],
                ))
        except Exception:
            logger.warning("Failed to get shard info for peer %s", pid, exc_info=True)

    return shards


def build_pipeline(shards: list[ShardInfo]) -> list[ShardInfo]:
    """Sort shards by layer range and validate contiguous coverage.

    Returns an ordered list from first shard (layer 0) to last.
    Raises ValueError if layers don't form a contiguous range.
    """
    if not shards:
        raise ValueError("No shards available to build pipeline")

    ordered = sorted(shards, key=lambda s: s.layer_start)

    # Validate contiguity
    for i in range(1, len(ordered)):
        if ordered[i].layer_start != ordered[i - 1].layer_end:
            raise ValueError(
                f"Gap in pipeline: shard ending at layer {ordered[i - 1].layer_end} "
                f"followed by shard starting at layer {ordered[i].layer_start}"
            )

    return ordered

"""MyceliumNode — wraps a libp2p host with Mycelium-specific protocols."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import multiaddr
import trio
from libp2p import new_host
from libp2p.kad_dht.kad_dht import DHTMode, KadDHT
from libp2p.pubsub.gossipsub import GossipSub
from libp2p.pubsub.pubsub import Pubsub
from libp2p.stream_muxer.mplex.mplex import MPLEX_PROTOCOL_ID, Mplex
from libp2p.tools.async_service import background_trio_service
from libp2p.tools.utils import info_from_p2p_addr
from libp2p.utils.address_validation import find_free_port, get_available_interfaces

from mycelium.network.protocols import (
    DHT_NAMESPACE,
    GOSSIP_TOPIC,
    GOSSIPSUB_PROTOCOL_ID,
    HEALTH_PROTOCOL,
    INFERENCE_PROTOCOL,
)
from mycelium.node.config import NodeConfig
from mycelium.node.identity import load_or_create_identity

if TYPE_CHECKING:
    from libp2p.abc import IHost
    from libp2p.network.stream.net_stream import INetStream

    from mycelium.inference.shard import ModelShard

logger = logging.getLogger(__name__)


class MyceliumNode:
    """A Mycelium peer that participates in the P2P inference network."""

    def __init__(self, config: NodeConfig) -> None:
        self.config = config
        self.host: IHost | None = None
        self.dht: KadDHT | None = None
        self.pubsub: Pubsub | None = None
        self.gossipsub: GossipSub | None = None
        self.shard: ModelShard | None = None
        self._pipeline: object | None = None  # set after import to avoid circular

    @property
    def peer_id(self) -> str:
        """Return this node's peer ID as a base58 string."""
        if self.host is None:
            raise RuntimeError("Node not started")
        return self.host.get_id().to_string()

    async def start(
        self,
        task_status=trio.TASK_STATUS_IGNORED,
    ) -> None:
        """Start the libp2p host, DHT, gossip, and register protocol handlers.

        Designed to be used with ``nursery.start(node.start)`` so that
        ``task_status.started()`` is called once the node is ready.
        """
        key_pair = load_or_create_identity(self.config.data_dir)

        port = self.config.listen_port
        if port <= 0:
            port = find_free_port()

        listen_addrs = get_available_interfaces(port)

        self.host = new_host(
            key_pair=key_pair,
            muxer_opt={MPLEX_PROTOCOL_ID: Mplex},
        )

        async with (
            self.host.run(listen_addrs=listen_addrs),
            trio.open_nursery() as nursery,
        ):
            nursery.start_soon(self.host.get_peerstore().start_cleanup_task, 60)

            # --- DHT ---
            self.dht = KadDHT(self.host, DHTMode.SERVER)
            from mycelium.network.discovery import MyceliumValidator
            self.dht.register_validator(DHT_NAMESPACE, MyceliumValidator())

            # --- GossipSub ---
            self.gossipsub = GossipSub(
                protocols=[GOSSIPSUB_PROTOCOL_ID],
                degree=3,
                degree_low=2,
                degree_high=4,
                time_to_live=60,
                gossip_window=2,
                gossip_history=5,
                heartbeat_initial_delay=2.0,
                heartbeat_interval=5,
            )
            self.pubsub = Pubsub(self.host, self.gossipsub)

            async with (
                background_trio_service(self.dht),
                background_trio_service(self.pubsub),
                background_trio_service(self.gossipsub),
            ):
                await self.pubsub.wait_until_ready()

                # Connect to bootstrap peers
                for addr_str in self.config.bootstrap_peers:
                    try:
                        maddr = multiaddr.Multiaddr(addr_str)
                        info = info_from_p2p_addr(maddr)
                        await self.host.connect(info)
                        await self.dht.add_peer(info.peer_id)
                        logger.info(
                            "Connected to bootstrap peer %s",
                            info.peer_id,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to connect to bootstrap peer %s",
                            addr_str,
                            exc_info=True,
                        )

                # Register protocol handlers
                self.host.set_stream_handler(HEALTH_PROTOCOL, self._handle_health)
                self.host.set_stream_handler(INFERENCE_PROTOCOL, self._handle_inference)

                # Subscribe to gossip
                self._gossip_sub = await self.pubsub.subscribe(GOSSIP_TOPIC)

                # Load model shard if configured
                if self.config.model_name:
                    await self._load_model_shard()

                # Print listening addresses
                addrs = self.host.get_addrs()
                logger.info(
                    "Mycelium node started — Peer ID: %s",
                    self.peer_id,
                )
                for addr in addrs:
                    logger.info("  Listening on: %s", addr)

                task_status.started()

                await trio.sleep_forever()

    async def _load_model_shard(self) -> None:
        """Load the configured model shard and register it on the DHT."""
        from mycelium.inference.shard import ModelShard
        from mycelium.models.loader import get_model_layer_count
        from mycelium.network.discovery import register_model_shard

        total_layers = await trio.to_thread.run_sync(
            get_model_layer_count, self.config.model_name
        )

        if self.config.layers is not None:
            layer_start, layer_end = self.config.layers
        else:
            # Default: serve all layers
            layer_start, layer_end = 0, total_layers

        logger.info(
            "Loading model %s layers [%d:%d] (total: %d)",
            self.config.model_name, layer_start, layer_end, total_layers,
        )

        self.shard = ModelShard(
            self.config.model_name,
            layer_start, layer_end, total_layers,
        )
        await trio.to_thread.run_sync(self.shard.load)

        # Register on DHT
        await register_model_shard(
            self.dht, self.peer_id, self.config.model_name, layer_start, layer_end,
        )
        logger.info("Model shard registered on DHT")

    async def _handle_health(self, stream: INetStream) -> None:
        """Respond to health check requests."""
        await stream.write(b"ok")
        await stream.close()

    async def _handle_inference(self, stream: INetStream) -> None:
        """Handle incoming inference activation data."""
        from mycelium.inference.pipeline import InferencePipeline

        if self._pipeline is None:
            self._pipeline = InferencePipeline(self)

        await self._pipeline.handle_inference_stream(stream)

    async def run(self) -> None:
        """Start the node and run forever. Convenience for CLI usage."""
        async with trio.open_nursery() as nursery:
            await nursery.start(self.start)
            await trio.sleep_forever()

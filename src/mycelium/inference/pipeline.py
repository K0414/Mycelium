"""InferencePipeline — routes activations through a chain of shard-holding peers."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

import msgpack
import trio

from mycelium.network.discovery import build_pipeline, find_model_shards
from mycelium.network.protocols import INFERENCE_PROTOCOL, MAX_READ_LEN
from mycelium.utils.serialization import deserialize_activation, serialize_activation

if TYPE_CHECKING:
    from libp2p.network.stream.net_stream import INetStream

    from mycelium.node.host import MyceliumNode

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Routes inference requests through the chain of model-shard peers."""

    def __init__(self, node: MyceliumNode) -> None:
        self.node = node

    async def handle_inference_stream(self, stream: INetStream) -> None:
        """Handle incoming inference activation from the previous node.

        Registered as the handler for ``INFERENCE_PROTOCOL``.
        """
        try:
            data = await stream.read(MAX_READ_LEN)
            request_id, hidden_states = deserialize_activation(data)

            logger.info("Received activation for request %s", request_id)

            # Run local shard in a thread (PyTorch is synchronous)
            result = await trio.to_thread.run_sync(
                self.node.shard.forward, hidden_states,
            )

            if self.node.shard.is_last:
                # We are the final shard — return the predicted token IDs
                token_ids = result.argmax(dim=-1).squeeze().tolist()
                if isinstance(token_ids, int):
                    token_ids = [token_ids]
                response = msgpack.packb({
                    "request_id": request_id,
                    "token_ids": token_ids,
                })
                await stream.write(response)
            else:
                # Forward to the next node in the pipeline
                next_peer_id = await self._find_next_peer(request_id)
                if next_peer_id is None:
                    error_response = msgpack.packb({
                        "request_id": request_id,
                        "error": "No next peer found in pipeline",
                    })
                    await stream.write(error_response)
                    await stream.close()
                    return

                from libp2p.peer.id import ID

                next_stream = await self.node.host.new_stream(
                    ID.from_base58(next_peer_id), [INFERENCE_PROTOCOL],
                )
                payload = serialize_activation(request_id, result)
                await next_stream.write(payload)

                # Relay response back
                response_data = await next_stream.read(MAX_READ_LEN)
                await next_stream.close()
                await stream.write(response_data)

            await stream.close()

        except Exception:
            logger.exception("Error handling inference stream")
            try:
                await stream.reset()
            except Exception:
                pass

    async def _find_next_peer(self, request_id: str) -> str | None:
        """Find the peer serving the next range of layers after this node's shard."""
        if self.node.shard is None or self.node.dht is None:
            return None

        shards = await find_model_shards(self.node.dht, self.node.shard.model_name)
        try:
            pipeline = build_pipeline(shards)
        except ValueError:
            logger.warning("Cannot build complete pipeline for request %s", request_id)
            return None

        # Find our position and return the next peer
        for i, shard in enumerate(pipeline):
            if shard.peer_id == self.node.peer_id:
                if i + 1 < len(pipeline):
                    return pipeline[i + 1].peer_id
                break

        return None

    async def submit_request(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 50,
    ) -> str:
        """Submit an inference request as a client.

        Performs autoregressive generation: sends input_ids through
        the pipeline, appends the predicted token, and repeats.
        """
        import torch
        from transformers import AutoTokenizer

        tokenizer = await trio.to_thread.run_sync(
            AutoTokenizer.from_pretrained, model_name,
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Discover the pipeline
        shards = await find_model_shards(self.node.dht, model_name)
        pipeline = build_pipeline(shards)

        if not pipeline:
            raise RuntimeError(
                f"No shards found for model {model_name}"
            )

        first_peer_id = pipeline[0].peer_id
        request_id = str(uuid.uuid4())

        logger.info(
            "Submitting request %s to pipeline of %d shards",
            request_id, len(pipeline),
        )

        from libp2p.peer.id import ID

        eos_token_id = tokenizer.eos_token_id
        generated_ids: list[int] = []

        for _ in range(max_tokens):
            payload = serialize_activation(request_id, input_ids)

            stream = await self.node.host.new_stream(
                ID.from_base58(first_peer_id),
                [INFERENCE_PROTOCOL],
            )
            await stream.write(payload)

            response_data = await stream.read(MAX_READ_LEN)
            await stream.close()

            response = msgpack.unpackb(response_data, raw=False)

            if "error" in response and response["error"]:
                raise RuntimeError(
                    f"Inference error: {response['error']}"
                )

            # Last token predicted
            next_token_id = response["token_ids"][-1]
            generated_ids.append(next_token_id)

            if next_token_id == eos_token_id:
                break

            # Append and continue
            next_token = torch.tensor(
                [[next_token_id]], dtype=torch.long,
            )
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return tokenizer.decode(
            generated_ids, skip_special_tokens=True,
        )

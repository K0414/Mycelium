"""Mycelium CLI."""

from __future__ import annotations

from pathlib import Path

import click


def parse_layers(layers_str: str | None) -> tuple[int, int] | None:
    """Parse a layer range string like '0:6' into (start, end)."""
    if layers_str is None or layers_str == "auto":
        return None
    parts = layers_str.split(":")
    if len(parts) != 2:
        msg = f"Invalid layer range: {layers_str!r}. Use 'start:end'."
        raise click.BadParameter(msg)
    return (int(parts[0]), int(parts[1]))


@click.group()
@click.version_option(package_name="mycelium-ai")
def cli() -> None:
    """Mycelium - Decentralized P2P LLM Inference."""


@cli.command()
@click.option("--port", default=0, help="Port to listen on (0 = auto-select).")
@click.option("--model", required=True, help="HuggingFace model name (e.g. gpt2).")
@click.option("--layers", default=None, help="Layer range, e.g. '0:6'.")
@click.option("--bootstrap", multiple=True, help="Bootstrap peer multiaddr(s).")
@click.option("--data-dir", default="~/.mycelium", help="Data directory.")
def serve(
    port: int,
    model: str,
    layers: str | None,
    bootstrap: tuple[str, ...],
    data_dir: str,
) -> None:
    """Start a Mycelium node and serve model layers."""
    import trio

    from mycelium.node.config import NodeConfig
    from mycelium.node.host import MyceliumNode
    from mycelium.utils.logging import setup_logging

    setup_logging()

    config = NodeConfig(
        listen_port=port,
        model_name=model,
        layers=parse_layers(layers),
        bootstrap_peers=list(bootstrap),
        data_dir=Path(data_dir).expanduser(),
    )
    node = MyceliumNode(config)
    trio.run(node.run)


@cli.command()
@click.argument("prompt")
@click.option("--model", required=True, help="Model to query on the network.")
@click.option("--bootstrap", multiple=True, help="Bootstrap peer multiaddr(s).")
def chat(prompt: str, model: str, bootstrap: tuple[str, ...]) -> None:
    """Send an inference request to the Mycelium network."""
    import trio

    from mycelium.node.config import NodeConfig
    from mycelium.node.host import MyceliumNode

    config = NodeConfig(
        bootstrap_peers=list(bootstrap),
    )

    async def _run_chat() -> None:
        node = MyceliumNode(config)
        async with trio.open_nursery() as nursery:
            await nursery.start(node.start)
            try:
                from mycelium.inference.pipeline import InferencePipeline

                pipeline = InferencePipeline(node)
                result = await pipeline.submit_request(model, prompt)
                click.echo(result)
            finally:
                nursery.cancel_scope.cancel()

    trio.run(_run_chat)

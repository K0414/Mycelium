"""Node configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class NodeConfig(BaseModel):
    """Configuration for a Mycelium node."""

    listen_port: int = 0
    """TCP port to listen on. 0 = auto-select a free port."""

    bootstrap_peers: list[str] = []
    """Multiaddrs of bootstrap peers to connect to on startup."""

    model_name: str = ""
    """HuggingFace model identifier (e.g. 'gpt2', 'meta-llama/Llama-3-8B')."""

    layers: tuple[int, int] | None = None
    """Layer range (start, end) to serve. None = auto-assign."""

    data_dir: Path = Path.home() / ".mycelium"
    """Directory for persistent state (identity key, etc.)."""

    max_batch_size: int = 1
    """Maximum inference batch size."""

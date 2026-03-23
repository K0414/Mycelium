# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mycelium is a decentralized peer-to-peer LLM inference and training framework. It distributes model layers across commodity hardware nodes connected via a P2P network. Currently in v0.1 — core P2P networking + single-model inference.

## Build & Development

```bash
pip install -e ".[dev]"     # Install in dev mode with test/lint deps
ruff check src/             # Lint
ruff format src/ tests/     # Format
pytest tests/               # Run all tests
pytest tests/test_shard.py::test_layer_assignment  # Run a single test
```

## Architecture

Four-layer stack, implemented in Python with py-libp2p (trio async runtime):

1. **CLI** (`src/mycelium/cli/main.py`) — `mycelium serve` and `mycelium chat` commands via Click
2. **Node** (`src/mycelium/node/host.py`) — `MyceliumNode` wraps a libp2p host, manages DHT, GossipSub, and protocol handlers
3. **Network** (`src/mycelium/network/`) — DHT-based model shard discovery (`discovery.py`), GossipSub announcements (`gossip.py`)
4. **Inference** (`src/mycelium/inference/`) — `ModelShard` loads a subset of transformer layers (`shard.py`), `InferencePipeline` routes activations through the node chain (`pipeline.py`)

### Key design decisions

- **trio** (not asyncio) — required by py-libp2p. PyTorch calls run in `trio.to_thread.run_sync()`.
- **Pipeline parallelism only** — each node serves a contiguous range of layers. No tensor parallelism.
- **Wire format** — safetensors for tensor data + msgpack metadata header (`utils/serialization.py`).
- **Model support** — architecture-agnostic layer extraction in `models/loader.py` (supports LLaMA-family and GPT-2-family models).
- **Test model** — GPT-2 (124M, 12 layers) for tests.

### Data flow

`Client (input_ids)` → `Shard 0 (embed + layers 0..N)` → wire transfer → `Shard 1 (layers N..M)` → ... → `Last shard (norm + lm_head → logits)` → `Client (decode tokens)`

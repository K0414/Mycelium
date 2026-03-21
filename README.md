# Mycelium

**Decentralized Peer-to-Peer Large Language Model Inference & Training**

---

## 🍄 What is Mycelium?

Mycelium is an open-source framework that enables distributed Large Language Model (LLM) inference and training across a peer-to-peer network of commodity hardware. Inspired by nature's most resilient decentralized system — the fungal mycelium network — Mycelium connects individual nodes into a unified intelligence mesh, allowing anyone to contribute compute and collectively run models that no single machine could handle alone.

## 🌍 Why Mycelium?

Today's most powerful AI models are locked behind centralized infrastructure controlled by a handful of corporations. Running a 70B+ parameter model requires expensive enterprise-grade GPUs. Mycelium breaks this barrier by:

- **Democratizing access** — Anyone with a consumer GPU (or even a CPU) can join the network and participate in serving large models.
- **Eliminating single points of failure** — No central server, no downtime, no censorship.
- **Preserving privacy** — Queries are distributed across nodes; no single participant sees the full picture.
- **Reducing cost** — Leverage idle compute from participants worldwide instead of renting cloud GPUs.

## ✨ Key Features

- **🔗 P2P Model Sharding** — Automatically partition model layers across available peers using tensor and pipeline parallelism.
- **🔍 Peer Discovery** — DHT-based node discovery with NAT traversal for seamless connectivity.
- **⚡ Adaptive Routing** — Intelligent query routing based on node latency, bandwidth, and available VRAM.
- **🛡️ Byzantine Fault Tolerance** — Consensus mechanisms to detect and exclude malicious or faulty nodes.
- **📦 Model Marketplace** — Browse, share, and collaboratively host open-weight models across the network.
- **🔐 Privacy-Preserving Inference** — Split inference ensures no single node has access to the complete input or output.
- **📊 Incentive Layer** — Optional token-based reward system for compute contributors.
- **🧩 Plugin Architecture** — Extensible middleware for custom quantization, caching, and model formats (GGUF, SafeTensors, etc.).

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  Client SDK                     │
│            (Python / REST / gRPC)                │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              Routing Layer                       │
│     Query planning · Load balancing · Retry      │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│            P2P Overlay Network                   │
│   libp2p · DHT · NAT Traversal · Gossip Protocol│
└───┬──────────┬──────────┬──────────┬────────────┘
    │          │          │          │
  ┌─▼─┐     ┌─▼─┐     ┌─▼─┐     ┌─▼─┐
  │ N1 │◄───►│ N2 │◄───►│ N3 │◄───►│ N4 │  ← Peer Nodes
  │L0-7│     │L8-15│    │L16-23│   │L24-31│    (Model Layers)
  └────┘     └────┘     └────┘     └────┘
```

## 🚀 Quick Start

```bash
pip install mycelium-ai

# Join the network as a compute node
mycelium serve --model meta-llama/Llama-3-70B --layers auto

# Query the network
mycelium chat "Explain quantum computing in simple terms"
```

## 🗺️ Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| **v0.1** | Core P2P networking + single-model inference | 🔨 In Progress |
| **v0.2** | Multi-model support + adaptive sharding | 📋 Planned |
| **v0.3** | Privacy-preserving inference (secure aggregation) | 📋 Planned |
| **v0.4** | Distributed fine-tuning (federated learning) | 📋 Planned |
| **v1.0** | Incentive layer + production hardening | 📋 Planned |

## 🤝 Contributing

Mycelium thrives on community contributions. Whether you're donating GPU cycles, writing code, or improving docs — every node in the network matters.

## 📄 License

Apache 2.0

---

*"In nature, a single mushroom is just the fruiting body. The real power lies underground — in the vast, invisible mycelium network connecting everything together."*

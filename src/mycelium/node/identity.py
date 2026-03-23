"""Node identity management — key generation and persistence."""

from __future__ import annotations

import secrets
from pathlib import Path

from libp2p.crypto.keys import KeyPair
from libp2p.crypto.secp256k1 import create_new_key_pair


def load_or_create_identity(data_dir: Path) -> KeyPair:
    """Load an existing identity key or generate a new one.

    The key is stored as raw 32-byte secp256k1 secret in ``data_dir/identity.key``.
    """
    key_path = data_dir / "identity.key"
    if key_path.exists():
        secret = key_path.read_bytes()
        return create_new_key_pair(secret)

    # Generate a new identity
    secret = secrets.token_bytes(32)
    data_dir.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(secret)
    return create_new_key_pair(secret)

"""Tests for node identity."""

from pathlib import Path

from mycelium.node.identity import load_or_create_identity


def test_create_new_identity(tmp_path: Path):
    key_pair = load_or_create_identity(tmp_path)
    assert key_pair is not None
    assert (tmp_path / "identity.key").exists()


def test_load_existing_identity(tmp_path: Path):
    kp1 = load_or_create_identity(tmp_path)
    kp2 = load_or_create_identity(tmp_path)
    # Same secret should produce the same key pair
    assert kp1.private_key.to_bytes() == kp2.private_key.to_bytes()


def test_different_dirs_different_keys(tmp_path: Path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    kp_a = load_or_create_identity(dir_a)
    kp_b = load_or_create_identity(dir_b)
    assert kp_a.private_key.to_bytes() != kp_b.private_key.to_bytes()

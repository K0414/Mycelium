"""Tests for discovery utilities."""

from mycelium.network.discovery import ShardInfo, build_pipeline


def test_build_pipeline_sorts_by_layer():
    shards = [
        ShardInfo(peer_id="peer-b", model_name="gpt2", layer_start=6, layer_end=12),
        ShardInfo(peer_id="peer-a", model_name="gpt2", layer_start=0, layer_end=6),
    ]
    pipeline = build_pipeline(shards)
    assert pipeline[0].peer_id == "peer-a"
    assert pipeline[1].peer_id == "peer-b"


def test_build_pipeline_detects_gap():
    shards = [
        ShardInfo(peer_id="peer-a", model_name="gpt2", layer_start=0, layer_end=4),
        ShardInfo(peer_id="peer-b", model_name="gpt2", layer_start=6, layer_end=12),
    ]
    try:
        build_pipeline(shards)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Gap" in str(e)


def test_build_pipeline_empty():
    try:
        build_pipeline([])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_build_pipeline_single_shard():
    shards = [
        ShardInfo(peer_id="peer-a", model_name="gpt2", layer_start=0, layer_end=12),
    ]
    pipeline = build_pipeline(shards)
    assert len(pipeline) == 1


def test_build_pipeline_three_shards():
    shards = [
        ShardInfo(peer_id="peer-c", model_name="gpt2", layer_start=8, layer_end=12),
        ShardInfo(peer_id="peer-a", model_name="gpt2", layer_start=0, layer_end=4),
        ShardInfo(peer_id="peer-b", model_name="gpt2", layer_start=4, layer_end=8),
    ]
    pipeline = build_pipeline(shards)
    assert [s.peer_id for s in pipeline] == ["peer-a", "peer-b", "peer-c"]

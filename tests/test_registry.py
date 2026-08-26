from __future__ import annotations

from pathlib import Path

import pytest

from app.services.registry import Resources, build_resources


class TestResources:
    def test_is_ready_only_when_every_component_loaded(self) -> None:
        assert Resources(embedder=object(), reranker=object(), db=object()).ready

    @pytest.mark.parametrize("missing", ["embedder", "reranker", "db"])
    def test_is_not_ready_while_anything_is_missing(self, missing: str) -> None:
        parts = {"embedder": object(), "reranker": object(), "db": object()}
        parts[missing] = None

        assert not Resources(**parts).ready


class TestBuildResources:
    """The models are baked into the image; loading must never touch the network.

    fastembed's HuggingFace path raises under `HF_HUB_OFFLINE` rather than
    falling back to the cache, so without `local_files_only` the container
    starts, retries a download it cannot make, and dies — a failure no unit test
    catches unless it pins these arguments.
    """

    @pytest.fixture
    def spies(self, monkeypatch) -> dict[str, dict]:
        import fastembed
        import fastembed.rerank.cross_encoder as cross_encoder
        import lancedb

        calls: dict[str, dict] = {}

        def record(name):
            def _spy(model_name=None, *args, **kwargs):
                calls[name] = {"model": model_name, **kwargs}
                return object()

            return _spy

        monkeypatch.setattr(fastembed, "TextEmbedding", record("embedder"))
        monkeypatch.setattr(cross_encoder, "TextCrossEncoder", record("reranker"))
        monkeypatch.setattr(lancedb, "connect", record("db"))
        return calls

    def test_loads_models_from_the_baked_cache_without_network(self, spies, settings) -> None:
        build_resources(settings)

        for component in ("embedder", "reranker"):
            assert spies[component]["local_files_only"] is True
            assert spies[component]["cache_dir"] == str(settings.model_cache_path)

    def test_passes_the_configured_model_names(self, spies, settings) -> None:
        build_resources(settings)

        assert spies["embedder"]["model"] == settings.embedding_model
        assert spies["reranker"]["model"] == settings.reranker_model

    def test_creates_the_data_directories(self, spies, settings) -> None:
        build_resources(settings)

        assert settings.lancedb_path.is_dir()
        assert settings.documents_path.is_dir()

    def test_returns_a_ready_bundle_with_a_job_store(self, spies, settings) -> None:
        resources = build_resources(settings)

        assert resources.ready
        assert resources.jobs is not None
        assert Path(settings.jobs_db_path).exists()

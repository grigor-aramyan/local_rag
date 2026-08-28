from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.registry import Resources, build_resources, tokenizer_for


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
    def spies(self, monkeypatch, word_tokenizer) -> dict[str, dict]:
        import anthropic
        import fastembed
        import fastembed.rerank.cross_encoder as cross_encoder
        import lancedb

        calls: dict[str, dict] = {}

        def record(name):
            def _spy(model_name=None, *args, **kwargs):
                calls[name] = {"model": model_name, **kwargs}
                # Mirrors fastembed's shape: chunking borrows `.model.tokenizer`.
                return SimpleNamespace(model=SimpleNamespace(tokenizer=word_tokenizer))

            return _spy

        def record_anthropic(**kwargs):
            calls["llm_client"] = kwargs
            return object()

        monkeypatch.setattr(fastembed, "TextEmbedding", record("embedder"))
        monkeypatch.setattr(cross_encoder, "TextCrossEncoder", record("reranker"))
        monkeypatch.setattr(lancedb, "connect", record("db"))
        monkeypatch.setattr(anthropic, "Anthropic", record_anthropic)
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

    def test_a_chunking_tokenizer_is_prepared_at_startup(self, spies, settings) -> None:
        """Copying the tokenizer per document would re-parse ~700 KB of JSON each time."""
        assert build_resources(settings).tokenizer is not None

    def test_an_llm_client_is_constructed(self, spies, settings) -> None:
        assert build_resources(settings).llm_client is not None

    def test_the_api_key_is_unwrapped_from_the_secret(self, spies, make_settings) -> None:
        settings = make_settings(anthropic_api_key="sk-ant-test-value")

        build_resources(settings)

        assert spies["llm_client"]["api_key"] == "sk-ant-test-value"

    def test_a_missing_api_key_does_not_fail_startup(self, spies, settings) -> None:
        """A missing key is a per-query auth failure at Anthropic, not a startup one."""
        build_resources(settings)

        assert spies["llm_client"]["api_key"] is None


class TestTokenizerFor:
    """Chunking measures text in the embedder's tokens, so it borrows its tokenizer."""

    def test_truncation_is_disabled_on_the_borrowed_copy(self, truncating_tokenizer) -> None:
        """Left on, it would report only the first 512 tokens and chunk away the rest."""
        embedder = SimpleNamespace(model=SimpleNamespace(tokenizer=truncating_tokenizer))

        assert tokenizer_for(embedder).truncation is None

    def test_the_embedders_own_tokenizer_is_not_mutated(self, truncating_tokenizer) -> None:
        """The live ONNX session shares this object; it must keep truncating."""
        embedder = SimpleNamespace(model=SimpleNamespace(tokenizer=truncating_tokenizer))

        tokenizer_for(embedder)

        assert truncating_tokenizer.truncation["max_length"] == 5

    def test_a_moved_fastembed_internal_fails_loudly(self) -> None:
        """`.model.tokenizer` is not public API, so a version bump can silently move it."""
        with pytest.raises(RuntimeError, match="tokenizer"):
            tokenizer_for(object())

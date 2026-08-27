from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer

from app.config import Settings
from app.jobs import JobStore
from app.services.ingestion import run_ingestion
from app.services.registry import Resources
from app.services.vectorstore import record_config, stored_fingerprint

BODY = "alpha beta gamma delta epsilon zeta eta theta"


@pytest.fixture
def ingest_settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(embedding_dim=4, chunk_size=4, chunk_overlap=1)


@pytest.fixture
def resources(
    ingest_settings: Settings, blank_db: Any, fake_embedder: Any, word_tokenizer: Tokenizer
) -> Resources:
    return Resources(
        embedder=fake_embedder,
        reranker=object(),
        db=blank_db,
        jobs=JobStore(ingest_settings.jobs_db_path),
        tokenizer=word_tokenizer,
    )


@pytest.fixture
def add_document(documents_dir: Path) -> Callable[[str, str], None]:
    def _add(name: str, content: str = BODY) -> None:
        (documents_dir / name).write_text(content, encoding="utf-8")

    return _add


def ingest(resources: Resources, settings: Settings, *names: str) -> Any:
    job = resources.jobs.create(list(names))  # type: ignore[union-attr]
    run_ingestion(job.job_id, resources, settings)
    return resources.jobs.get(job.job_id)  # type: ignore[union-attr]


def chunks_table(resources: Resources, settings: Settings) -> Any:
    return resources.db.open_table(settings.table_name)


class TestSuccessfulRun:
    def test_the_job_completes(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        add_document("notes.md")

        job = ingest(resources, ingest_settings, "notes.md")

        assert job.state == "completed"
        assert job.error is None

    def test_the_document_lands_in_the_table(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        add_document("notes.md")

        ingest(resources, ingest_settings, "notes.md")

        rows = chunks_table(resources, ingest_settings).to_arrow().to_pylist()
        assert rows
        assert all(row["source"] == "notes.md" for row in rows)
        assert "alpha" in " ".join(row["text"] for row in rows)

    def test_every_row_carries_an_embedding_of_the_configured_width(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        add_document("notes.md")

        ingest(resources, ingest_settings, "notes.md")

        rows = chunks_table(resources, ingest_settings).to_arrow().to_pylist()
        assert all(len(row["vector"]) == ingest_settings.embedding_dim for row in rows)

    def test_progress_advances_once_per_document(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        add_document("a.md")
        add_document("b.md")

        job = ingest(resources, ingest_settings, "a.md", "b.md")

        assert job.processed == 2
        assert job.total == 2

    def test_several_documents_are_all_indexed(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        add_document("a.md")
        add_document("b.md")

        ingest(resources, ingest_settings, "a.md", "b.md")

        rows = chunks_table(resources, ingest_settings).to_arrow().to_pylist()
        assert {row["source"] for row in rows} == {"a.md", "b.md"}

    def test_html_is_indexed_as_text_not_markup(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        add_document("page.html", "<p>alpha beta</p><script>secret()</script>")

        ingest(resources, ingest_settings, "page.html")

        text = " ".join(
            r["text"] for r in chunks_table(resources, ingest_settings).to_arrow().to_pylist()
        )
        assert "alpha beta" in text
        assert "script" not in text
        assert "secret" not in text

    def test_the_full_text_index_is_built(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        """Hybrid retrieval needs FTS, and retrofitting it later means a full rebuild."""
        add_document("notes.md")

        ingest(resources, ingest_settings, "notes.md")

        indexes = chunks_table(resources, ingest_settings).list_indices()
        assert any(list(i.columns) == ["text"] for i in indexes)

    def test_the_config_fingerprint_is_recorded_on_first_ingest(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        add_document("notes.md")

        ingest(resources, ingest_settings, "notes.md")

        assert stored_fingerprint(resources.db, ingest_settings) is not None


class TestIdempotence:
    def test_re_ingesting_a_document_does_not_duplicate_rows(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        """A re-uploaded name overwrites the file; the index must track it, not grow."""
        add_document("notes.md")

        ingest(resources, ingest_settings, "notes.md")
        first = chunks_table(resources, ingest_settings).count_rows()
        ingest(resources, ingest_settings, "notes.md")

        assert chunks_table(resources, ingest_settings).count_rows() == first

    def test_an_edited_document_replaces_its_old_chunks(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        add_document("notes.md", "alpha beta")
        ingest(resources, ingest_settings, "notes.md")

        add_document("notes.md", "omega psi")
        ingest(resources, ingest_settings, "notes.md")

        texts = " ".join(
            r["text"] for r in chunks_table(resources, ingest_settings).to_arrow().to_pylist()
        )
        assert "omega" in texts
        assert "alpha" not in texts


class TestFailures:
    def test_a_missing_file_fails_the_job_and_names_it(
        self, resources: Resources, ingest_settings: Settings
    ) -> None:
        job = ingest(resources, ingest_settings, "vanished.md")

        assert job.state == "failed"
        assert "vanished.md" in job.error

    def test_a_document_with_no_extractable_text_fails_loudly(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        """Reporting success on a document that indexed nothing is the worse failure."""
        add_document("blank.html", "<script>only()</script>")

        job = ingest(resources, ingest_settings, "blank.html")

        assert job.state == "failed"
        assert "blank.html" in job.error

    def test_an_unexpected_error_fails_the_job_without_leaking_internals(
        self, resources: Resources, ingest_settings: Settings, add_document, monkeypatch
    ) -> None:
        """Job errors are returned over HTTP, so a raw traceback must not reach the client."""
        add_document("notes.md")

        def explode(*args: object, **kwargs: object) -> None:
            raise RuntimeError("/secret/host/path blew up at 0xdeadbeef")

        monkeypatch.setattr(resources.embedder, "embed", explode)

        job = ingest(resources, ingest_settings, "notes.md")

        assert job.state == "failed"
        assert "/secret/host/path" not in job.error
        assert "notes.md" in job.error

    def test_a_traversing_document_name_is_refused(
        self, resources: Resources, ingest_settings: Settings, tmp_path: Path
    ) -> None:
        """Defence in depth: the job store is not the trust boundary the sanitiser is."""
        (tmp_path / "outside.md").write_text(BODY, encoding="utf-8")

        job = ingest(resources, ingest_settings, "../outside.md")

        assert job.state == "failed"

    def test_a_failure_stops_the_run_at_the_broken_document(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        add_document("good.md")

        job = ingest(resources, ingest_settings, "good.md", "missing.md")

        assert job.state == "failed"
        assert job.processed == 1, "documents already indexed stay indexed and stay counted"

    def test_an_incompatible_config_fails_the_job_before_writing(
        self, resources: Resources, ingest_settings: Settings, add_document
    ) -> None:
        """Embedding new chunks with a different model into an old table poisons retrieval."""
        record_config(resources.db, ingest_settings.model_copy(update={"embedding_dim": 768}))
        add_document("notes.md")

        job = ingest(resources, ingest_settings, "notes.md")

        assert job.state == "failed"
        assert "embedding_dim" in job.error
        assert ingest_settings.table_name not in list(resources.db.table_names(limit=100))

    def test_a_run_without_a_job_store_is_a_programming_error(
        self, ingest_settings: Settings, blank_db: Any, fake_embedder: Any
    ) -> None:
        bare = Resources(embedder=fake_embedder, reranker=object(), db=blank_db, jobs=None)

        with pytest.raises(RuntimeError, match="job store"):
            run_ingestion("whatever", bare, ingest_settings)

    def test_an_unknown_job_id_does_not_raise(
        self, resources: Resources, ingest_settings: Settings
    ) -> None:
        """The task is queued after the job row is committed, but never trust that ordering."""
        run_ingestion("no-such-job", resources, ingest_settings)

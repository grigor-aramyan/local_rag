from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.config import Settings
from app.services.chunking import Chunk
from app.services.vectorstore import (
    ACL_ALL,
    ConfigMismatchError,
    UnsafeSourceError,
    compact,
    config_fingerprint,
    ensure_ann_index,
    ensure_fts_index,
    open_chunks_table,
    record_config,
    rows_from_chunks,
    stored_fingerprint,
    verify_config,
)


def make_chunk(source: str = "a.md", ordinal: int = 0, text: str = "hello") -> Chunk:
    return Chunk(
        chunk_id=f"{source}:{ordinal}",
        source=source,
        ordinal=ordinal,
        text=text,
        content_hash="0" * 64,
        page=None,
    )


def write_document(table, source: str, texts: list[str], dim: int = 4) -> None:
    chunks = [make_chunk(source, i, text) for i, text in enumerate(texts)]
    vectors = [[float(i)] * dim for i in range(len(chunks))]
    table_rows = rows_from_chunks(chunks, vectors)
    from app.services.vectorstore import replace_document

    replace_document(table, source, table_rows)


@pytest.fixture
def small_settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(embedding_dim=4)


@pytest.fixture
def db(small_settings: Settings) -> Any:
    import lancedb

    small_settings.lancedb_path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(small_settings.lancedb_path))


@pytest.fixture
def table(db: Any, small_settings: Settings) -> Any:
    return open_chunks_table(db, small_settings)


class TestTable:
    def test_the_table_is_created_on_first_open(self, db: Any, small_settings: Settings) -> None:
        open_chunks_table(db, small_settings)

        assert small_settings.table_name in list(db.table_names(limit=100))

    def test_the_vector_column_matches_the_configured_dimension(self, table: Any) -> None:
        """A dimension mismatch surfaces as a write error, long after the config changed."""
        assert table.schema.field("vector").type.list_size == 4

    def test_the_schema_carries_every_documented_column(self, table: Any) -> None:
        expected = {
            "id",
            "text",
            "vector",
            "source",
            "page",
            "content_hash",
            "ingested_at",
            "acl",
        }

        assert set(table.schema.names) == expected

    def test_opening_twice_reuses_the_same_table(self, db: Any, small_settings: Settings) -> None:
        """Re-creating on open would silently drop the index on every restart."""
        first = open_chunks_table(db, small_settings)
        write_document(first, "a.md", ["hello"])

        second = open_chunks_table(db, small_settings)

        assert second.count_rows() == 1


class TestRows:
    def test_rows_default_to_the_shared_acl(self) -> None:
        """Multi-tenancy is in the schema now so enabling it later is not a full reindex."""
        rows = rows_from_chunks([make_chunk()], [[0.0] * 4])

        assert rows[0]["acl"] == ACL_ALL

    def test_rows_carry_the_chunk_identity(self) -> None:
        chunk = make_chunk("guide.md", 3, "some text")

        row = rows_from_chunks([chunk], [[0.0] * 4])[0]

        assert row["id"] == chunk.chunk_id
        assert row["source"] == "guide.md"
        assert row["text"] == "some text"
        assert row["content_hash"] == chunk.content_hash

    def test_rows_are_stamped_with_an_ingestion_time(self) -> None:
        rows = rows_from_chunks([make_chunk()], [[0.0] * 4])

        assert rows[0]["ingested_at"].tzinfo is not None

    def test_a_vector_per_chunk_is_required(self) -> None:
        with pytest.raises(ValueError, match="vector"):
            rows_from_chunks([make_chunk(), make_chunk(ordinal=1)], [[0.0] * 4])


class TestReplaceDocument:
    def test_chunks_are_written(self, table: Any) -> None:
        write_document(table, "a.md", ["one", "two"])

        assert table.count_rows() == 2

    def test_re_ingesting_a_document_does_not_duplicate_rows(self, table: Any) -> None:
        """A re-uploaded name overwrites; the index must follow, not accumulate."""
        write_document(table, "a.md", ["one", "two"])
        write_document(table, "a.md", ["one", "two"])

        assert table.count_rows() == 2

    def test_a_shortened_document_loses_its_stale_chunks(self, table: Any) -> None:
        """Editing content out of a document must remove it from retrieval too."""
        write_document(table, "a.md", ["one", "two", "three"])
        write_document(table, "a.md", ["one"])

        assert table.count_rows() == 1
        assert table.to_arrow().to_pylist()[0]["text"] == "one"

    def test_an_edited_document_reflects_the_new_text(self, table: Any) -> None:
        write_document(table, "a.md", ["original"])
        write_document(table, "a.md", ["rewritten"])

        assert [r["text"] for r in table.to_arrow().to_pylist()] == ["rewritten"]

    def test_other_documents_are_untouched(self, table: Any) -> None:
        write_document(table, "a.md", ["from a"])
        write_document(table, "b.md", ["from b"])

        write_document(table, "a.md", ["from a, edited"])

        by_source = {r["source"]: r["text"] for r in table.to_arrow().to_pylist()}
        assert by_source["b.md"] == "from b"

    def test_a_document_reduced_to_nothing_is_removed(self, table: Any) -> None:
        from app.services.vectorstore import replace_document

        write_document(table, "a.md", ["one", "two"])
        replace_document(table, "a.md", [])

        assert table.count_rows() == 0

    @pytest.mark.parametrize(
        "source",
        ["a.md' OR '1'='1", "a.md'; DROP TABLE chunks; --", "../etc/passwd", "a b.md", ""],
    )
    def test_a_source_outside_the_sanitised_alphabet_is_refused(
        self, table: Any, source: str
    ) -> None:
        """The source goes into a filter expression, so only sanitised names may reach it."""
        from app.services.vectorstore import replace_document

        with pytest.raises(UnsafeSourceError):
            replace_document(table, source, rows_from_chunks([make_chunk()], [[0.0] * 4]))

    def test_a_quoted_source_cannot_delete_another_document(self, table: Any) -> None:
        from app.services.vectorstore import replace_document

        write_document(table, "keep.md", ["precious"])

        with pytest.raises(UnsafeSourceError):
            replace_document(table, "x.md' OR source LIKE '%", [])

        assert table.count_rows() == 1


class TestIndexes:
    def test_the_fts_index_is_built_on_the_text_column(self, table: Any) -> None:
        """Retrofitting an FTS index needs a full rebuild, so it is created at ingest time."""
        write_document(table, "a.md", ["hello world"])

        ensure_fts_index(table)

        indexes = {i.name: list(i.columns) for i in table.list_indices()}
        assert any(cols == ["text"] for cols in indexes.values())

    def test_rebuilding_the_fts_index_is_safe(self, table: Any) -> None:
        write_document(table, "a.md", ["hello world"])

        ensure_fts_index(table)
        ensure_fts_index(table)

        assert len(table.list_indices()) == 1

    def test_no_index_is_built_for_an_empty_table(self, table: Any) -> None:
        ensure_fts_index(table)

        assert table.list_indices() == []

    def test_the_ann_index_is_skipped_below_the_threshold(
        self, table: Any, small_settings: Settings
    ) -> None:
        """Brute force beats IVF-PQ on a small table, and training it needs rows to learn from."""
        write_document(table, "a.md", ["one", "two"])

        assert ensure_ann_index(table, small_settings) is False
        assert not any(i.index_type == "IVF_PQ" for i in table.list_indices())

    def test_the_ann_index_is_built_above_the_threshold(
        self, table: Any, make_settings: Callable[..., Settings], monkeypatch
    ) -> None:
        settings = make_settings(embedding_dim=4, ann_index_threshold=2)
        write_document(table, "a.md", ["one", "two", "three", "four"])
        recorded: dict[str, Any] = {}
        monkeypatch.setattr(
            type(table), "create_index", lambda self, **kw: recorded.update(kw), raising=True
        )

        assert ensure_ann_index(table, settings) is True
        assert recorded["index_type"] == "IVF_PQ"

    def test_the_ann_index_never_asks_for_more_partitions_than_rows(
        self, table: Any, make_settings: Callable[..., Settings], monkeypatch
    ) -> None:
        """IVF training fails outright when partitions outnumber the vectors to cluster."""
        settings = make_settings(embedding_dim=4, ann_index_threshold=2)
        write_document(table, "a.md", ["one", "two", "three", "four"])
        recorded: dict[str, Any] = {}
        monkeypatch.setattr(
            type(table), "create_index", lambda self, **kw: recorded.update(kw), raising=True
        )

        ensure_ann_index(table, settings)

        assert 1 <= recorded["num_partitions"] <= table.count_rows()
        assert settings.embedding_dim % recorded["num_sub_vectors"] == 0

    def test_compaction_runs_without_error(self, table: Any) -> None:
        write_document(table, "a.md", ["one"])
        write_document(table, "b.md", ["two"])

        compact(table)

        assert table.count_rows() == 2


class TestConfigGuard:
    def test_nothing_is_recorded_before_the_first_ingest(
        self, db: Any, small_settings: Settings
    ) -> None:
        assert stored_fingerprint(db, small_settings) is None

    def test_an_unconfigured_database_verifies_cleanly(
        self, db: Any, small_settings: Settings
    ) -> None:
        """A fresh volume has nothing to be incompatible with."""
        verify_config(db, small_settings)

    def test_the_first_ingest_records_the_fingerprint(
        self, db: Any, small_settings: Settings
    ) -> None:
        record_config(db, small_settings)

        assert stored_fingerprint(db, small_settings) == config_fingerprint(small_settings)

    def test_the_fingerprint_covers_the_retrieval_shaping_settings(
        self, small_settings: Settings
    ) -> None:
        assert set(config_fingerprint(small_settings)) == {
            "embedding_model",
            "embedding_dim",
            "chunk_size",
            "chunk_overlap",
        }

    def test_recording_twice_keeps_one_row(self, db: Any, small_settings: Settings) -> None:
        record_config(db, small_settings)
        record_config(db, small_settings)

        assert db.open_table(f"{small_settings.table_name}__meta").count_rows() == 1

    def test_matching_config_verifies_cleanly(self, db: Any, small_settings: Settings) -> None:
        record_config(db, small_settings)

        verify_config(db, small_settings)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("embedding_model", "BAAI/bge-base-en-v1.5"),
            ("embedding_dim", 768),
            ("chunk_size", 900),
            ("chunk_overlap", 120),
        ],
    )
    def test_a_diverging_setting_refuses_to_serve(
        self,
        db: Any,
        small_settings: Settings,
        make_settings: Callable[..., Settings],
        field: str,
        value: object,
    ) -> None:
        """A changed embedder against an old index degrades retrieval silently, not loudly."""
        record_config(db, small_settings)
        changed = make_settings(**{"embedding_dim": 4, field: value})

        with pytest.raises(ConfigMismatchError) as caught:
            verify_config(db, changed)

        assert field in str(caught.value)

    def test_the_mismatch_error_names_both_values(
        self, db: Any, small_settings: Settings, make_settings: Callable[..., Settings]
    ) -> None:
        """The operator has to know what to change back, so both sides go in the message."""
        record_config(db, small_settings)

        with pytest.raises(ConfigMismatchError) as caught:
            verify_config(db, make_settings(embedding_dim=768))

        assert "768" in str(caught.value)
        assert "4" in str(caught.value)

    def test_recording_a_conflicting_config_is_refused(
        self, db: Any, small_settings: Settings, make_settings: Callable[..., Settings]
    ) -> None:
        """Ingest must not overwrite the fingerprint and bless a half-reindexed table."""
        record_config(db, small_settings)

        with pytest.raises(ConfigMismatchError):
            record_config(db, make_settings(embedding_dim=768))

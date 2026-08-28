from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from app.config import Settings
from app.services.chunking import Chunk

# Multi-tenancy is in the schema from the start with everything sharing one ACL.
# Nothing filters on it yet, but adding a column to a populated LanceDB table
# means rebuilding it, so the cheap move is to carry it from the first write.
ACL_ALL = "all"

# The alphabet `sanitize_upload_filename` leaves behind. A source name is
# interpolated into a LanceDB filter expression, so it is re-checked here rather
# than trusted to have come from the sanitiser — a name carrying a quote would
# otherwise let an upload rewrite the predicate and delete another document.
_SAFE_SOURCE = re.compile(r"\A[A-Za-z0-9._-]{1,255}\Z")

# Fields whose drift invalidates an existing index.
_FINGERPRINT_FIELDS = ("embedding_model", "embedding_dim", "chunk_size", "chunk_overlap")

# PQ needs the vector width to divide evenly by the sub-vector count.
_MAX_SUB_VECTORS = 96
_MAX_PARTITIONS = 256


class ConfigMismatchError(RuntimeError):
    """Current settings disagree with the ones the existing index was built with."""


class UnsafeSourceError(ValueError):
    """A document name reached a filter expression without passing the sanitiser."""


def chunks_schema(dim: int) -> pa.Schema:
    """The table the brief specifies, plus `acl`.

    Raw chunk text is stored, not just its vector: prompts and citations both need
    the text back, and re-deriving it from the source file at query time would
    make retrieval depend on the document still being on disk.
    """
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("page", pa.int32(), nullable=True),
            pa.field("content_hash", pa.string(), nullable=False),
            pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("acl", pa.string(), nullable=False),
        ]
    )


def meta_table_name(settings: Settings) -> str:
    return f"{settings.table_name}__meta"


def open_chunks_table(db: Any, settings: Settings) -> Any:
    """Open the chunks table, creating it on first use.

    Opening must never re-create: `mode="overwrite"` here would drop the FTS index
    (and the corpus) on every restart.
    """
    try:
        return db.open_table(settings.table_name)
    except ValueError:
        return db.create_table(
            settings.table_name,
            schema=chunks_schema(settings.embedding_dim),
            exist_ok=True,
        )


def rows_from_chunks(
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    *,
    acl: str = ACL_ALL,
    ingested_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Pair chunks with their embeddings into rows matching `chunks_schema`."""
    if len(chunks) != len(vectors):
        raise ValueError(
            f"one vector per chunk is required: {len(chunks)} chunks, {len(vectors)} vectors"
        )

    stamp = ingested_at or datetime.now(UTC)
    return [
        {
            "id": chunk.chunk_id,
            "text": chunk.text,
            "vector": [float(value) for value in vector],
            "source": chunk.source,
            "page": chunk.page,
            "content_hash": chunk.content_hash,
            "ingested_at": stamp,
            "acl": acl,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def replace_document(table: Any, source: str, rows: Sequence[dict[str, Any]]) -> None:
    """Make the table's rows for `source` exactly `rows`, in one commit.

    Upsert alone would leave the tail of a shortened document behind as orphaned
    chunks that still answer queries. `when_not_matched_by_source_delete` scopes
    the delete to this document, so the whole replacement is a single commit and a
    crash cannot leave the document half-indexed.
    """
    predicate = f"source = '{_checked_source(source)}'"

    if not rows:
        # An empty merge raises; a document that now yields nothing is a delete.
        table.delete(predicate)
        return

    mismatched = {row["source"] for row in rows} - {source}
    if mismatched:
        raise ValueError(f"rows for {sorted(mismatched)} passed to a replace of {source!r}")

    # Bound to the table's own schema rather than passed as dicts: LanceDB infers
    # every inferred column as nullable, which the non-nullable schema rejects.
    data = pa.Table.from_pylist(list(rows), schema=table.schema)

    (
        table.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .when_not_matched_by_source_delete(predicate)
        .execute(data)
    )


def hybrid_search(
    table: Any, query_vector: Sequence[float], query_text: str, limit: int
) -> list[dict[str, Any]]:
    """Vector + FTS retrieval, combined by LanceDB's reciprocal-rank-fusion reranker.

    The vector is passed in explicitly rather than as a bare string query: LanceDB
    would otherwise look up a registered embedding function for the column, and
    this table has none — embedding happens once, upstream, through the same
    fastembed model the index was built with.

    Returns `[]` for a table nothing has been ingested into yet. Hybrid search
    raises once there is no FTS index, and the FTS index is only built after
    ingestion writes rows — so an empty table is the one case this must not let
    through as an error.
    """
    if table.count_rows() == 0:
        return []

    return (
        table.search(query_type="hybrid")
        .vector(list(query_vector))
        .text(query_text)
        .limit(limit)
        .select(["id", "text", "source", "page", "content_hash"])
        .to_list()
    )


def ensure_fts_index(table: Any) -> bool:
    """(Re)build the full-text index the hybrid retriever needs.

    Built at ingest time deliberately: LanceDB cannot retrofit an FTS index onto
    existing rows without a full rebuild, so waiting until the first query would
    put that cost in the request path.
    """
    if table.count_rows() == 0:
        return False
    table.create_fts_index("text", replace=True)
    return True


def ensure_ann_index(table: Any, settings: Settings) -> bool:
    """Create an IVF-PQ index once the table is large enough to need one.

    Below the threshold a brute-force scan is both faster and exact, and IVF has
    too few vectors to cluster meaningfully. Returns whether an index was built.
    """
    rows = table.count_rows()
    if rows < settings.ann_index_threshold:
        return False

    table.create_index(
        metric="cosine",
        # IVF training fails outright if it is asked for more partitions than
        # there are vectors to cluster.
        num_partitions=max(1, min(_MAX_PARTITIONS, int(math.sqrt(rows)))),
        num_sub_vectors=_sub_vectors_for(settings.embedding_dim),
        vector_column_name="vector",
        replace=True,
        index_type="IVF_PQ",
    )
    return True


def compact(table: Any) -> None:
    """Merge the fragments an ingestion run leaves behind."""
    table.optimize()


def config_fingerprint(settings: Settings) -> dict[str, Any]:
    """The settings an existing index is only valid under."""
    return {field: getattr(settings, field) for field in _FINGERPRINT_FIELDS}


def stored_fingerprint(db: Any, settings: Settings) -> dict[str, Any] | None:
    """The fingerprint recorded at first ingest, or None if nothing is indexed yet."""
    try:
        meta = db.open_table(meta_table_name(settings))
    except ValueError:
        return None

    rows = meta.to_arrow().to_pylist()
    if not rows:
        return None
    return {field: rows[0][field] for field in _FINGERPRINT_FIELDS}


def record_config(db: Any, settings: Settings) -> None:
    """Stamp the current config onto the index, or confirm it still matches.

    Called before each ingest rather than only the first: it is the point where
    new vectors would otherwise be written into a table built by a different
    embedder, which degrades retrieval without ever raising.
    """
    existing = stored_fingerprint(db, settings)
    if existing is not None:
        _compare(existing, config_fingerprint(settings))
        return

    schema = _meta_schema()
    meta = db.create_table(meta_table_name(settings), schema=schema, exist_ok=True)
    row = {**config_fingerprint(settings), "recorded_at": datetime.now(UTC)}
    meta.add(pa.Table.from_pylist([row], schema=schema))


def verify_config(db: Any, settings: Settings) -> None:
    """Refuse to serve an index built under incompatible settings. No-op if empty."""
    existing = stored_fingerprint(db, settings)
    if existing is not None:
        _compare(existing, config_fingerprint(settings))


def _meta_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("embedding_model", pa.string(), nullable=False),
            pa.field("embedding_dim", pa.int32(), nullable=False),
            pa.field("chunk_size", pa.int32(), nullable=False),
            pa.field("chunk_overlap", pa.int32(), nullable=False),
            pa.field("recorded_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    )


def _compare(stored: dict[str, Any], current: dict[str, Any]) -> None:
    drift = [(field, stored.get(field), value) for field, value in current.items()]
    drift = [(field, was, now) for field, was, now in drift if was != now]
    if not drift:
        return

    details = "; ".join(
        f"{field} was {was!r} at index time, config now says {now!r}" for field, was, now in drift
    )
    raise ConfigMismatchError(
        f"the existing index is incompatible with the current configuration ({details}). "
        "Restore the previous settings, or delete the LanceDB volume and re-ingest."
    )


def _checked_source(source: str) -> str:
    if not _SAFE_SOURCE.fullmatch(source):
        raise UnsafeSourceError(f"{source!r} is not a sanitised document name")
    return source


def _sub_vectors_for(dim: int) -> int:
    """Largest sub-vector count that divides `dim` evenly — PQ requires it."""
    for candidate in range(min(_MAX_SUB_VECTORS, dim), 0, -1):
        if dim % candidate == 0:
            return candidate
    return 1

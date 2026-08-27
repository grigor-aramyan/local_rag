from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit of a document.

    `text` is sliced out of the original string rather than rebuilt from tokens:
    detokenising a WordPiece sequence loses casing and spacing, and this text is
    what citations quote back to the user.
    """

    chunk_id: str
    source: str
    ordinal: int
    text: str
    content_hash: str
    page: int | None = None


def content_hash_for(text: str) -> str:
    """Stable across processes — unlike `hash()`, which is salted per interpreter."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id_for(source: str, ordinal: int, content_hash: str) -> str:
    """Derive a chunk's primary key.

    Keying on the content hash is what makes re-ingesting an unchanged document a
    no-op: identical text yields identical ids, so the write updates rows in place
    instead of appending duplicates. Source and ordinal join the hash so that
    boilerplate repeated within a document — or shared between two documents —
    stays as separate rows instead of silently collapsing into one.
    """
    digest = hashlib.sha256()
    digest.update(source.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(ordinal).encode("ascii"))
    digest.update(b"\x00")
    digest.update(content_hash.encode("ascii"))
    return digest.hexdigest()


def chunk_document(
    text: str,
    *,
    source: str,
    tokenizer: Any,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split `text` into overlapping windows of at most `chunk_size` tokens.

    Windows are measured with the embedder's own tokenizer, so `chunk_size` means
    the same thing here as it does to the model. That matters: bge-small truncates
    at 512 tokens, and a chunk measured in words would routinely overshoot and
    lose its tail with no error anywhere.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    encoding = _measuring_tokenizer(tokenizer).encode(text, add_special_tokens=False)
    offsets: list[tuple[int, int]] = list(encoding.offsets)
    if not offsets:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    start = 0

    while start < len(offsets):
        window = offsets[start : start + chunk_size]
        begin = window[0][0]
        # Not `window[-1][1]`: a tokenizer may emit a zero-width token last.
        end = max(stop for _, stop in window)

        if end > begin:
            piece = text[begin:end].strip()
            if piece:
                digest = content_hash_for(piece)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id_for(source, len(chunks), digest),
                        source=source,
                        ordinal=len(chunks),
                        text=piece,
                        content_hash=digest,
                        # Markdown, HTML, and text have no pages. PDFs would set
                        # this; faking a 1 here would make citations lie.
                        page=None,
                    )
                )

        if start + chunk_size >= len(offsets):
            break
        start += step

    return chunks


def _measuring_tokenizer(tokenizer: Any) -> Any:
    """Return a tokenizer that will report every token in the text.

    The embedder's tokenizer has truncation pinned at the model's context length,
    so encoding a long document through it reports only the first 512 tokens and
    everything past that would be chunked away. Truncation is disabled on a copy —
    the original is shared with the running ONNX session and must not be mutated.
    `registry.tokenizer_for` normally hands us an already-safe copy, so in the
    service this check costs one attribute read per document.
    """
    if getattr(tokenizer, "truncation", None) is None:
        return tokenizer

    from tokenizers import Tokenizer

    safe = Tokenizer.from_str(tokenizer.to_str())
    safe.no_truncation()
    safe.no_padding()
    return safe

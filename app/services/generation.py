from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.schemas import Citation

SYSTEM_PROMPT = (
    "Answer the user's question using only the documents provided as context. "
    "Base every claim on them; if they do not contain the answer, say so plainly "
    "instead of guessing."
)

NO_CONTEXT_SYSTEM_PROMPT = (
    "Nothing in the knowledge base was relevant to this question. Answer from "
    "your own knowledge, concisely, and do not imply the answer came from any "
    "particular source."
)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A reranked chunk, ready to become a citable document block."""

    text: str
    source: str
    page: int | None


def generate_answer(
    client: Any, settings: Settings, question: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[Citation]]:
    """Ask Claude to answer, grounded in `chunks` via native citations.

    Generation is streamed internally so a slow response cannot hit the SDK's
    non-streaming timeout guard; the caller still gets back one assembled answer,
    since the query endpoint's response body is a single JSON object rather than
    an event stream.

    With no chunks (nothing relevant was retrieved), no `document` blocks are
    sent and no citations come back — the caller is expected to attach its own
    warning that this answer is not grounded in the corpus.
    """
    if not chunks:
        return _generate_plain(client, settings, question), []

    content: list[dict[str, Any]] = [_document_block(chunk) for chunk in chunks]
    content.append({"type": "text", "text": question})

    with client.messages.stream(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": settings.llm_effort},
        messages=[{"role": "user", "content": content}],
    ) as stream:
        message = stream.get_final_message()

    return _extract_answer(message, chunks)


def _generate_plain(client: Any, settings: Settings, question: str) -> str:
    with client.messages.stream(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        system=NO_CONTEXT_SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": settings.llm_effort},
        messages=[{"role": "user", "content": question}],
    ) as stream:
        message = stream.get_final_message()

    return "".join(block.text for block in message.content if block.type == "text")


def _document_block(chunk: RetrievedChunk) -> dict[str, Any]:
    title = chunk.source if chunk.page is None else f"{chunk.source} (page {chunk.page})"
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
        "title": title,
        "citations": {"enabled": True},
    }


def _extract_answer(message: Any, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    """Concatenate the response's text blocks and dedupe citations by (source, page).

    Native citations key by `document_index` — one per chunk, in the order the
    `document` blocks were sent. The same chunk cited twice keeps the marker it
    was first assigned, like a footnote reused later in the same answer.
    """
    parts: list[str] = []
    markers: dict[tuple[str, int | None], int] = {}
    citations: list[Citation] = []

    for block in message.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        for citation in getattr(block, "citations", None) or []:
            index = citation.document_index
            if index is None or not (0 <= index < len(chunks)):
                continue
            chunk = chunks[index]
            key = (chunk.source, chunk.page)
            if key not in markers:
                markers[key] = len(markers) + 1
                citations.append(
                    Citation(marker=markers[key], source=chunk.source, page=chunk.page)
                )

    return "".join(parts), citations

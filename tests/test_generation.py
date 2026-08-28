from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services.generation import (
    NO_CONTEXT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    RetrievedChunk,
    generate_answer,
)
from tests.conftest import FakeAnthropicClient


def make_chunk(
    source: str = "a.md", text: str = "some text", page: int | None = None
) -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, page=page)


class TestNoContext:
    def test_a_plain_string_question_is_sent_with_no_documents(
        self, settings: Settings, make_fake_message: Callable[..., Any]
    ) -> None:
        client = FakeAnthropicClient(make_fake_message())

        generate_answer(client, settings, "what is this?", [])

        sent = client.calls[0]
        assert sent["messages"][0]["content"] == "what is this?"
        assert sent["system"] == NO_CONTEXT_SYSTEM_PROMPT

    def test_the_answer_is_the_concatenated_text(
        self,
        settings: Settings,
        make_fake_message: Callable[..., Any],
        make_text_block: Callable[..., Any],
    ) -> None:
        client = FakeAnthropicClient(make_fake_message(make_text_block("I don't know.")))

        answer, citations = generate_answer(client, settings, "what is this?", [])

        assert answer == "I don't know."
        assert citations == []


class TestWithContext:
    def test_each_chunk_becomes_a_citable_document_block(
        self, settings: Settings, make_fake_message: Callable[..., Any]
    ) -> None:
        client = FakeAnthropicClient(make_fake_message())
        chunk = make_chunk(source="guide.md", text="the sky is blue")

        generate_answer(client, settings, "what color is the sky?", [chunk])

        content = client.calls[0]["messages"][0]["content"]
        document = content[0]
        assert document["type"] == "document"
        assert document["source"] == {
            "type": "text",
            "media_type": "text/plain",
            "data": "the sky is blue",
        }
        assert document["title"] == "guide.md"
        assert document["citations"] == {"enabled": True}

    def test_the_page_is_folded_into_the_title(
        self, settings: Settings, make_fake_message: Callable[..., Any]
    ) -> None:
        client = FakeAnthropicClient(make_fake_message())
        chunk = make_chunk(source="report.pdf", page=3)

        generate_answer(client, settings, "q", [chunk])

        assert client.calls[0]["messages"][0]["content"][0]["title"] == "report.pdf (page 3)"

    def test_the_question_follows_the_documents(
        self, settings: Settings, make_fake_message: Callable[..., Any]
    ) -> None:
        client = FakeAnthropicClient(make_fake_message())

        generate_answer(
            client, settings, "what color is the sky?", [make_chunk(), make_chunk(source="b.md")]
        )

        content = client.calls[0]["messages"][0]["content"]
        assert content[-1] == {"type": "text", "text": "what color is the sky?"}

    def test_uses_the_grounded_system_prompt(
        self, settings: Settings, make_fake_message: Callable[..., Any]
    ) -> None:
        client = FakeAnthropicClient(make_fake_message())

        generate_answer(client, settings, "q", [make_chunk()])

        assert client.calls[0]["system"] == SYSTEM_PROMPT

    def test_uses_adaptive_thinking_and_the_configured_effort(
        self, settings: Settings, make_fake_message: Callable[..., Any]
    ) -> None:
        client = FakeAnthropicClient(make_fake_message())

        generate_answer(client, settings, "q", [make_chunk()])

        sent = client.calls[0]
        assert sent["thinking"] == {"type": "adaptive"}
        assert sent["output_config"] == {"effort": settings.llm_effort}
        assert sent["model"] == settings.llm_model
        assert sent["max_tokens"] == settings.llm_max_tokens


class TestCitationExtraction:
    def test_text_blocks_are_concatenated_in_order(
        self,
        settings: Settings,
        make_fake_message: Callable[..., Any],
        make_text_block: Callable[..., Any],
    ) -> None:
        client = FakeAnthropicClient(
            make_fake_message(make_text_block("Part one. "), make_text_block("Part two."))
        )

        answer, _ = generate_answer(client, settings, "q", [make_chunk()])

        assert answer == "Part one. Part two."

    def test_a_citation_is_mapped_back_to_its_source(
        self,
        settings: Settings,
        make_fake_message: Callable[..., Any],
        make_text_block: Callable[..., Any],
        make_citation: Callable[..., Any],
    ) -> None:
        chunk = make_chunk(source="guide.md")
        message = make_fake_message(
            make_text_block("the sky is blue", citations=[make_citation(0)])
        )
        client = FakeAnthropicClient(message)

        _, citations = generate_answer(client, settings, "q", [chunk])

        assert len(citations) == 1
        assert citations[0].marker == 1
        assert citations[0].source == "guide.md"
        assert citations[0].page is None

    def test_the_page_is_carried_into_the_citation(
        self,
        settings: Settings,
        make_fake_message: Callable[..., Any],
        make_text_block: Callable[..., Any],
        make_citation: Callable[..., Any],
    ) -> None:
        chunk = make_chunk(source="report.pdf", page=5)
        message = make_fake_message(
            make_text_block("water is essential", citations=[make_citation(0)])
        )
        client = FakeAnthropicClient(message)

        _, citations = generate_answer(client, settings, "q", [chunk])

        assert citations[0].page == 5

    def test_repeated_citations_to_the_same_source_share_a_marker(
        self,
        settings: Settings,
        make_fake_message: Callable[..., Any],
        make_text_block: Callable[..., Any],
        make_citation: Callable[..., Any],
    ) -> None:
        chunk = make_chunk(source="guide.md")
        message = make_fake_message(
            make_text_block("first claim", citations=[make_citation(0)]),
            make_text_block("second claim", citations=[make_citation(0)]),
        )
        client = FakeAnthropicClient(message)

        _, citations = generate_answer(client, settings, "q", [chunk])

        assert len(citations) == 1
        assert citations[0].marker == 1

    def test_distinct_sources_get_distinct_markers_in_first_seen_order(
        self,
        settings: Settings,
        make_fake_message: Callable[..., Any],
        make_text_block: Callable[..., Any],
        make_citation: Callable[..., Any],
    ) -> None:
        chunks = [make_chunk(source="a.md"), make_chunk(source="b.md")]
        message = make_fake_message(
            make_text_block("claim about b", citations=[make_citation(1)]),
            make_text_block("claim about a", citations=[make_citation(0)]),
        )
        client = FakeAnthropicClient(message)

        _, citations = generate_answer(client, settings, "q", chunks)

        assert [(c.marker, c.source) for c in citations] == [(1, "b.md"), (2, "a.md")]

    def test_a_citation_pointing_past_the_chunk_list_is_ignored(
        self,
        settings: Settings,
        make_fake_message: Callable[..., Any],
        make_text_block: Callable[..., Any],
        make_citation: Callable[..., Any],
    ) -> None:
        """Defensive: the SDK is trusted, but an off-by-one here must not crash the request."""
        message = make_fake_message(make_text_block("claim", citations=[make_citation(7)]))
        client = FakeAnthropicClient(message)

        _, citations = generate_answer(client, settings, "q", [make_chunk()])

        assert citations == []

    def test_a_text_block_with_no_citations_contributes_no_markers(
        self,
        settings: Settings,
        make_fake_message: Callable[..., Any],
        make_text_block: Callable[..., Any],
    ) -> None:
        message = make_fake_message(make_text_block("uncited text"))
        client = FakeAnthropicClient(message)

        _, citations = generate_answer(client, settings, "q", [make_chunk()])

        assert citations == []

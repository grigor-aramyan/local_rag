from __future__ import annotations

import pytest
from tokenizers import Tokenizer

from app.services.chunking import Chunk, chunk_document, content_hash_for


def token_count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def chunk(tokenizer: Tokenizer, text: str, size: int, overlap: int, source: str = "doc.md"):
    return chunk_document(
        text, source=source, tokenizer=tokenizer, chunk_size=size, chunk_overlap=overlap
    )


@pytest.fixture
def words() -> str:
    return " ".join(f"w{i}" for i in range(100))


class TestWindowing:
    def test_short_text_is_one_chunk_holding_everything(self, word_tokenizer: Tokenizer) -> None:
        chunks = chunk(word_tokenizer, "alpha beta gamma", size=10, overlap=2)

        assert len(chunks) == 1
        assert chunks[0].text == "alpha beta gamma"

    def test_long_text_is_split_into_several_chunks(
        self, word_tokenizer: Tokenizer, words: str
    ) -> None:
        chunks = chunk(word_tokenizer, words, size=10, overlap=2)

        assert len(chunks) > 1

    def test_no_chunk_exceeds_the_token_budget(self, word_tokenizer: Tokenizer, words: str) -> None:
        """The embedder truncates at 512; a chunk over budget loses its tail silently."""
        chunks = chunk(word_tokenizer, words, size=10, overlap=2)

        assert all(token_count(word_tokenizer, c.text) <= 10 for c in chunks)

    def test_consecutive_chunks_overlap(self, word_tokenizer: Tokenizer, words: str) -> None:
        """Overlap is what stops a fact that straddles a boundary from being unretrievable."""
        first, second = chunk(word_tokenizer, words, size=10, overlap=3)[:2]

        assert first.text.split()[-3:] == second.text.split()[:3]

    def test_zero_overlap_produces_disjoint_windows(
        self, word_tokenizer: Tokenizer, words: str
    ) -> None:
        chunks = chunk(word_tokenizer, words, size=10, overlap=0)

        assert " ".join(c.text for c in chunks).split() == words.split()

    def test_every_token_of_the_source_appears_somewhere(
        self, word_tokenizer: Tokenizer, words: str
    ) -> None:
        chunks = chunk(word_tokenizer, words, size=10, overlap=3)

        covered = {word for c in chunks for word in c.text.split()}
        assert covered == set(words.split())

    def test_the_trailing_remainder_is_not_dropped(self, word_tokenizer: Tokenizer) -> None:
        text = " ".join(f"w{i}" for i in range(23))

        chunks = chunk(word_tokenizer, text, size=10, overlap=0)

        assert chunks[-1].text == "w20 w21 w22"

    def test_a_final_window_is_not_emitted_twice(self, word_tokenizer: Tokenizer) -> None:
        """A window that already reaches the end must terminate the walk, not restart it."""
        text = " ".join(f"w{i}" for i in range(20))

        chunks = chunk(word_tokenizer, text, size=10, overlap=5)

        assert len({c.text for c in chunks}) == len(chunks)

    def test_ordinals_count_up_from_zero(self, word_tokenizer: Tokenizer, words: str) -> None:
        chunks = chunk(word_tokenizer, words, size=10, overlap=2)

        assert [c.ordinal for c in chunks] == list(range(len(chunks)))


class TestChunkText:
    def test_chunk_text_is_sliced_from_the_original(self, word_tokenizer: Tokenizer) -> None:
        """Text is cut by character offset, never rebuilt from tokens, which would lose case."""
        text = "The Quick BROWN fox, jumped; over 3 lazy dogs."

        chunks = chunk(word_tokenizer, text, size=100, overlap=0)

        assert chunks[0].text == text

    def test_whitespace_only_text_yields_no_chunks(self, word_tokenizer: Tokenizer) -> None:
        assert chunk(word_tokenizer, "   \n\n  \t ", size=10, overlap=2) == []

    def test_empty_text_yields_no_chunks(self, word_tokenizer: Tokenizer) -> None:
        assert chunk(word_tokenizer, "", size=10, overlap=2) == []

    def test_chunks_carry_their_source(self, word_tokenizer: Tokenizer) -> None:
        chunks = chunk(word_tokenizer, "alpha beta", size=10, overlap=0, source="guide.md")

        assert all(c.source == "guide.md" for c in chunks)

    def test_page_is_absent_for_paged_less_formats(self, word_tokenizer: Tokenizer) -> None:
        """Markdown, HTML, and text have no pages; the column stays null rather than faking 1."""
        chunks = chunk(word_tokenizer, "alpha beta", size=10, overlap=0)

        assert chunks[0].page is None


class TestIdentity:
    def test_the_same_document_yields_the_same_ids(
        self, word_tokenizer: Tokenizer, words: str
    ) -> None:
        """Stable ids are what make a re-upload update rows in place instead of duplicating."""
        first = chunk(word_tokenizer, words, size=10, overlap=2)
        second = chunk(word_tokenizer, words, size=10, overlap=2)

        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_ids_are_unique_within_a_document(self, word_tokenizer: Tokenizer, words: str) -> None:
        chunks = chunk(word_tokenizer, words, size=10, overlap=2)

        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_repeated_text_at_different_positions_stays_distinct(
        self, word_tokenizer: Tokenizer
    ) -> None:
        """Boilerplate repeated in one document must not collapse into a single row."""
        chunks = chunk(word_tokenizer, "same words here same words here", size=3, overlap=0)

        assert chunks[0].content_hash == chunks[1].content_hash
        assert chunks[0].chunk_id != chunks[1].chunk_id

    def test_the_same_text_in_two_documents_gets_different_ids(
        self, word_tokenizer: Tokenizer
    ) -> None:
        """Ids are per-document so deleting one source cannot orphan another's rows."""
        here = chunk(word_tokenizer, "shared text", size=10, overlap=0, source="a.md")
        there = chunk(word_tokenizer, "shared text", size=10, overlap=0, source="b.md")

        assert here[0].chunk_id != there[0].chunk_id
        assert here[0].content_hash == there[0].content_hash

    def test_changed_text_changes_the_id(self, word_tokenizer: Tokenizer) -> None:
        before = chunk(word_tokenizer, "the original text", size=10, overlap=0)
        after = chunk(word_tokenizer, "the corrected text", size=10, overlap=0)

        assert before[0].chunk_id != after[0].chunk_id

    def test_content_hash_is_stable_across_processes(self) -> None:
        """A salted hash (Python's built-in) would change every restart and defeat merging."""
        assert content_hash_for("alpha") == content_hash_for("alpha")
        assert len(content_hash_for("alpha")) == 64

    def test_chunks_are_immutable(self, word_tokenizer: Tokenizer) -> None:
        chunk_ = chunk(word_tokenizer, "alpha beta", size=10, overlap=0)[0]

        with pytest.raises((AttributeError, TypeError)):
            chunk_.text = "tampered"  # type: ignore[misc]


class TestTokenizerSafety:
    def test_a_truncating_tokenizer_does_not_cut_the_document_short(
        self, truncating_tokenizer: Tokenizer, words: str
    ) -> None:
        """The embedder's own tokenizer truncates at 512 — chunking must see the whole text."""
        chunks = chunk_document(
            words,
            source="doc.md",
            tokenizer=truncating_tokenizer,
            chunk_size=10,
            chunk_overlap=0,
        )

        assert " ".join(c.text for c in chunks).split() == words.split()

    def test_the_shared_tokenizer_is_left_untouched(
        self, truncating_tokenizer: Tokenizer, words: str
    ) -> None:
        """The embedder holds this object; disabling its truncation would break inference."""
        chunk_document(
            words,
            source="doc.md",
            tokenizer=truncating_tokenizer,
            chunk_size=10,
            chunk_overlap=0,
        )

        assert truncating_tokenizer.truncation is not None
        assert truncating_tokenizer.truncation["max_length"] == 5


def test_chunk_document_returns_chunk_instances(word_tokenizer: Tokenizer) -> None:
    assert all(isinstance(c, Chunk) for c in chunk(word_tokenizer, "alpha beta", 10, 0))

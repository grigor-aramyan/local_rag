from __future__ import annotations

from pathlib import Path

import pytest

from app.services.extraction import ExtractionError, extract_html, extract_text


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestPlainFormats:
    def test_markdown_is_kept_verbatim(self, tmp_path: Path) -> None:
        """Chunk text is what citations quote, so markdown stays as the author wrote it."""
        source = "# Title\n\nA paragraph with **bold** and a [link](http://example.com).\n"

        assert extract_text(write(tmp_path, "notes.md", source)) == source.strip()

    def test_plain_text_is_kept_verbatim(self, tmp_path: Path) -> None:
        assert extract_text(write(tmp_path, "notes.txt", "line one\nline two")) == (
            "line one\nline two"
        )

    @pytest.mark.parametrize("name", ["a.md", "a.markdown", "a.txt", "a.text"])
    def test_every_plain_suffix_is_handled(self, tmp_path: Path, name: str) -> None:
        assert extract_text(write(tmp_path, name, "content")) == "content"

    def test_windows_line_endings_are_normalised(self, tmp_path: Path) -> None:
        """Offsets index the extracted string, so a stray \\r would land inside chunk text."""
        assert extract_text(write(tmp_path, "a.txt", "one\r\ntwo\r\n")) == "one\ntwo"


class TestHtml:
    def test_tags_are_stripped_and_text_kept(self) -> None:
        assert extract_html("<p>Hello <em>world</em>.</p>") == "Hello world."

    def test_script_contents_never_reach_the_index(self) -> None:
        markup = "<p>Real text.</p><script>var secret = 'do not index me';</script>"

        assert extract_html(markup) == "Real text."

    def test_style_contents_never_reach_the_index(self) -> None:
        markup = "<style>body { color: red; }</style><p>Real text.</p>"

        assert extract_html(markup) == "Real text."

    def test_head_metadata_is_dropped(self) -> None:
        markup = "<html><head><title>Tab title</title></head><body><p>Body.</p></body></html>"

        assert extract_html(markup) == "Body."

    def test_entities_are_decoded(self) -> None:
        assert extract_html("<p>Tom &amp; Jerry &lt;3 caf&eacute;</p>") == "Tom & Jerry <3 café"

    def test_block_elements_become_line_breaks(self) -> None:
        markup = "<p>First.</p><p>Second.</p>"

        assert extract_html(markup) == "First.\n\nSecond."

    def test_inline_elements_do_not_break_a_sentence(self) -> None:
        """A <span> splitting a sentence must not become two chunks' worth of fragments."""
        assert extract_html("<p>One <span>whole</span> sentence.</p>") == "One whole sentence."

    def test_line_break_tags_become_newlines(self) -> None:
        assert extract_html("<p>One<br>Two</p>") == "One\nTwo"

    def test_list_items_are_separated(self) -> None:
        assert extract_html("<ul><li>Alpha</li><li>Beta</li></ul>") == "Alpha\nBeta"

    def test_runs_of_blank_lines_collapse(self) -> None:
        markup = "<div><p>A</p><div><div><p>B</p></div></div></div>"

        assert extract_html(markup) == "A\n\nB"

    def test_horizontal_whitespace_collapses(self) -> None:
        assert extract_html("<p>lots     of\t\tspace</p>") == "lots of space"

    def test_unclosed_tags_do_not_raise(self) -> None:
        """Uploaded HTML is untrusted and often malformed; extraction must not be a crash path."""
        assert extract_html("<p>Dangling <b>bold") == "Dangling bold"

    def test_comments_are_dropped(self) -> None:
        assert extract_html("<p>Visible<!-- hidden note --></p>") == "Visible"

    @pytest.mark.parametrize("name", ["page.html", "page.htm"])
    def test_html_files_are_routed_to_the_html_extractor(self, tmp_path: Path, name: str) -> None:
        path = write(tmp_path, name, "<p>Hello <b>there</b></p>")

        assert extract_text(path) == "Hello there"


class TestRejections:
    def test_an_unknown_suffix_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractionError, match="pdf"):
            extract_text(write(tmp_path, "paper.pdf", "content"))

    def test_a_missing_file_is_refused_without_leaking_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractionError) as caught:
            extract_text(tmp_path / "gone.md")

        assert "gone.md" in str(caught.value)
        assert str(tmp_path) not in str(caught.value)

    def test_undecodable_bytes_are_refused(self, tmp_path: Path) -> None:
        """Uploads are UTF-8 validated, but a file on the bind mount was never checked."""
        path = tmp_path / "binary.md"
        path.write_bytes(b"\xff\xfe\x00 not utf-8")

        with pytest.raises(ExtractionError, match="UTF-8"):
            extract_text(path)

    def test_a_document_with_no_extractable_text_comes_back_empty(self, tmp_path: Path) -> None:
        """Signalling emptiness is the caller's job; extraction just reports what it found."""
        path = write(tmp_path, "empty.html", "<script>only()</script>")

        assert extract_text(path) == ""

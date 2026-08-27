from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

PLAIN_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".text"})
HTML_SUFFIXES = frozenset({".html", ".htm"})

# Content that is markup machinery rather than prose. Indexing a stylesheet or an
# inline script fills the retriever with tokens no question will ever be about.
_SKIPPED = frozenset(
    {"script", "style", "head", "title", "noscript", "template", "svg", "iframe", "object"}
)

# Elements that end a paragraph: they become a blank line.
_BLOCKS = frozenset(
    {
        "address", "article", "aside", "blockquote", "div", "dl", "dd", "dt",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3",
        "h4", "h5", "h6", "header", "hr", "main", "nav", "ol", "p", "pre",
        "section", "table", "ul",
    }
)  # fmt: skip

# Elements that end a line but not a paragraph.
_LINE_BREAKS = frozenset({"br", "li", "tr", "option"})

_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
_BLANK_RUN = re.compile(r"\n{3,}")


class ExtractionError(ValueError):
    """A document could not be turned into indexable text."""


def extract_text(path: Path) -> str:
    """Read one document and return its plain text.

    Errors name the document but never its full path: the message travels to the
    client as a job error, and the container's layout is not the caller's business.
    Returning `""` is a legitimate outcome — deciding that an empty document is a
    failure belongs to the ingest pipeline, not here.
    """
    suffix = path.suffix.lower()
    if suffix not in PLAIN_SUFFIXES | HTML_SUFFIXES:
        raise ExtractionError(f"{suffix or 'no extension'} is not a supported document format")

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ExtractionError(f"{path.name} is no longer on disk") from exc
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"{path.name} is not valid UTF-8 text") from exc
    except OSError as exc:
        raise ExtractionError(f"{path.name} could not be read") from exc

    if suffix in HTML_SUFFIXES:
        return extract_html(raw)
    return _normalise_newlines(raw).strip()


def extract_html(markup: str) -> str:
    """Strip markup down to the prose a reader would see."""
    parser = _HtmlText()
    parser.feed(markup)
    parser.close()
    return parser.text()


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


class _HtmlText(HTMLParser):
    """Collect visible text, dropping markup machinery.

    `convert_charrefs` (on by default) decodes entities for us. Uploaded HTML is
    untrusted and frequently malformed, so nothing here may depend on tags being
    balanced — hence a skip *depth* rather than a flag, and `<body>` resetting it
    so an unclosed `<head>` cannot swallow the whole document.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag == "body":
            self._skip_depth = 0
        if tag in _SKIPPED:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in _BLOCKS:
            self._parts.append("\n\n")
        elif tag in _LINE_BREAKS:
            # Only on the opening tag: `</li><li>` must not stack two newlines.
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in _BLOCKS:
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        joined = _normalise_newlines("".join(self._parts))
        joined = _HORIZONTAL_SPACE.sub(" ", joined)
        joined = _SPACE_AROUND_NEWLINE.sub("\n", joined)
        return _BLANK_RUN.sub("\n\n", joined).strip()

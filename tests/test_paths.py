from __future__ import annotations

from pathlib import Path

import pytest

from app.paths import (
    MAX_FILENAME_LENGTH,
    PathTraversalError,
    UnsupportedFormatError,
    resolve_within,
    sanitize_upload_filename,
)

ALLOWED = frozenset({".md", ".markdown", ".txt", ".text", ".html", ".htm"})


def sanitize(name: str) -> str:
    return sanitize_upload_filename(name, ALLOWED)


class TestSanitizeUploadFilename:
    def test_keeps_an_ordinary_name(self) -> None:
        assert sanitize("notes.md") == "notes.md"

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("../../etc/passwd.txt", "passwd.txt"),
            ("..\\..\\windows\\evil.txt", "evil.txt"),
            ("/absolute/path/doc.html", "doc.html"),
            ("nested/deeper/note.md", "note.md"),
        ],
    )
    def test_strips_directory_components(self, name: str, expected: str) -> None:
        assert sanitize(name) == expected

    def test_strips_leading_dots_so_uploads_cannot_be_hidden_files(self) -> None:
        assert sanitize(".hidden.md") == "hidden.md"

    def test_replaces_shell_and_glob_characters(self) -> None:
        assert sanitize("my report;rm -rf *.md") == "my_report_rm_-rf_.md"

    def test_rejects_a_null_byte(self) -> None:
        with pytest.raises(PathTraversalError):
            sanitize("notes.md\x00.png")

    @pytest.mark.parametrize("name", ["", "..", "/", "..///"])
    def test_rejects_names_with_nothing_usable_left(self, name: str) -> None:
        with pytest.raises((PathTraversalError, UnsupportedFormatError)):
            sanitize(name)

    @pytest.mark.parametrize("name", ["payload.exe", "archive.zip", "script.js", "noextension"])
    def test_rejects_extensions_outside_the_allowlist(self, name: str) -> None:
        with pytest.raises(UnsupportedFormatError):
            sanitize(name)

    def test_extension_check_is_case_insensitive(self) -> None:
        assert sanitize("README.MD") == "README.md"

    def test_truncates_an_overlong_name_but_keeps_the_extension(self) -> None:
        result = sanitize("a" * 500 + ".md")

        assert len(result) <= MAX_FILENAME_LENGTH
        assert result.endswith(".md")

    def test_a_double_extension_is_judged_on_the_last_one(self) -> None:
        """`invoice.md.exe` is an .exe, not a .md — the allowlist sees the last suffix."""
        with pytest.raises(UnsupportedFormatError):
            sanitize("invoice.md.exe")

        assert sanitize("invoice.exe.md") == "invoice.exe.md"


class TestResolveWithin:
    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        (tmp_path / "docs" / "nested").mkdir(parents=True)
        (tmp_path / "docs" / "nested" / "note.md").write_text("hello")
        return tmp_path / "docs"

    def test_accepts_a_relative_path_inside_the_root(self, root: Path) -> None:
        assert resolve_within(root, "nested/note.md") == (root / "nested" / "note.md").resolve()

    def test_accepts_the_root_itself(self, root: Path) -> None:
        assert resolve_within(root, ".") == root.resolve()

    @pytest.mark.parametrize(
        "candidate",
        ["../outside.md", "nested/../../outside.md", "/etc/passwd", "nested/../../../../etc"],
    )
    def test_rejects_escapes_from_the_root(self, root: Path, candidate: str) -> None:
        with pytest.raises(PathTraversalError):
            resolve_within(root, candidate)

    def test_rejects_a_symlink_pointing_outside_the_root(self, root: Path, tmp_path: Path) -> None:
        secret = tmp_path / "secret.md"
        secret.write_text("private")
        (root / "link.md").symlink_to(secret)

        with pytest.raises(PathTraversalError):
            resolve_within(root, "link.md")

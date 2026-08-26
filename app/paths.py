from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

MAX_FILENAME_LENGTH = 200

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class PathTraversalError(ValueError):
    """A caller-supplied name or path resolved outside its permitted root."""


class UnsupportedFormatError(ValueError):
    """An upload carried an extension outside the allowlist."""


def sanitize_upload_filename(name: str, allowed_extensions: frozenset[str]) -> str:
    """Reduce an uploaded `filename` to a safe basename inside the documents root.

    The multipart `filename` is attacker-controlled: it can carry POSIX or
    Windows separators, `..` segments, null bytes, or a name long enough to be
    truncated by the filesystem into something else. Everything below strips
    rather than trusts, and the extension allowlist is the format gate.
    """
    if "\x00" in name:
        raise PathTraversalError("filename contains a null byte")

    # Strip directory components under both separator conventions — a POSIX
    # basename of "..\\..\\evil.md" is the whole string.
    basename = PureWindowsPath(PurePosixPath(name).name).name
    basename = _UNSAFE.sub("_", basename).lstrip(".")

    if not basename or basename in {".", ".."}:
        raise PathTraversalError(f"{name!r} has no usable filename")

    suffix = Path(basename).suffix.lower()
    if suffix not in allowed_extensions:
        raise UnsupportedFormatError(
            f"{suffix or 'no extension'} is not supported; "
            f"allowed: {', '.join(sorted(allowed_extensions))}"
        )

    stem = basename[: -len(suffix)]
    if len(basename) > MAX_FILENAME_LENGTH:
        stem = stem[: MAX_FILENAME_LENGTH - len(suffix)]
    if not stem:
        raise PathTraversalError(f"{name!r} has no usable filename")

    return f"{stem}{suffix}"


def resolve_within(root: Path, candidate: str | Path) -> Path:
    """Resolve `candidate` under `root`, refusing anything that escapes it.

    Defence in depth behind `sanitize_upload_filename`: `resolve()` collapses
    `..` and follows symlinks before the containment check, so a symlink planted
    inside the documents mount cannot be used to write outside it.
    """
    text = str(candidate)
    if "\x00" in text:
        raise PathTraversalError("path contains a null byte")

    root_resolved = root.resolve()
    target = (root_resolved / text).resolve()

    if target != root_resolved and root_resolved not in target.parents:
        raise PathTraversalError(f"{candidate!r} resolves outside {root_resolved}")

    return target

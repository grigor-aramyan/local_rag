from __future__ import annotations

import codecs
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.paths import (
    PathTraversalError,
    UnsupportedFormatError,
    resolve_within,
    sanitize_upload_filename,
)

READ_CHUNK = 64 * 1024
STAGING_PREFIX = ".staging-"


class UploadTooLargeError(ValueError):
    """An upload exceeded `max_upload_bytes`."""


@dataclass(slots=True)
class _Staged:
    filename: str
    temp_path: Path
    final_path: Path


async def save_uploads(files: Sequence[UploadFile], settings: Settings) -> list[str]:
    """Persist uploads into the documents directory, all-or-nothing.

    Each file is streamed to a staging file beside its destination and only
    renamed into place once every file in the batch has passed validation, so a
    rejected upload never leaves a half-written batch for the ingester to find.
    Bytes are counted as they arrive rather than trusting Content-Length, and
    the write is abandoned the moment the cap is passed.
    """
    root = settings.documents_path
    root.mkdir(parents=True, exist_ok=True)

    staged: list[_Staged] = []
    try:
        for upload in files:
            staged.append(await _stage(upload, root, settings))

        stored: list[str] = []
        for item in staged:
            # Same directory, so this is an atomic replace rather than a copy.
            os.replace(item.temp_path, item.final_path)
            stored.append(item.filename)
    except BaseException:
        for item in staged:
            item.temp_path.unlink(missing_ok=True)
        raise

    return list(dict.fromkeys(stored))


async def _stage(upload: UploadFile, root: Path, settings: Settings) -> _Staged:
    filename = sanitize_upload_filename(upload.filename or "", settings.allowed_extensions)
    final_path = resolve_within(root, filename)
    if final_path == root.resolve():
        raise PathTraversalError(f"{upload.filename!r} does not name a file")

    temp_path = root / f"{STAGING_PREFIX}{uuid.uuid4().hex}"
    decoder = codecs.getincrementaldecoder("utf-8")()
    written = 0

    try:
        with temp_path.open("wb") as handle:
            while chunk := await upload.read(READ_CHUNK):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise UploadTooLargeError(
                        f"{filename} exceeds the {settings.max_upload_bytes} byte limit"
                    )
                try:
                    decoder.decode(chunk)
                except UnicodeDecodeError as exc:
                    raise UnsupportedFormatError(f"{filename} is not valid UTF-8 text") from exc
                await run_in_threadpool(handle.write, chunk)

            try:
                decoder.decode(b"", final=True)
            except UnicodeDecodeError as exc:
                raise UnsupportedFormatError(f"{filename} is not valid UTF-8 text") from exc

        if written == 0:
            raise UnsupportedFormatError(f"{filename} is empty")
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    return _Staged(filename=filename, temp_path=temp_path, final_path=final_path)

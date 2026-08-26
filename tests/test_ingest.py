from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MARKDOWN = b"# Title\n\nSome body text.\n"


def upload(client: TestClient, *files: tuple[str, bytes]):
    return client.post(
        "/ingest",
        files=[("files", (name, content, "text/markdown")) for name, content in files],
    )


class TestUploadHandling:
    def test_stores_the_document_and_returns_a_job(
        self, client: TestClient, documents_dir: Path
    ) -> None:
        response = upload(client, ("notes.md", MARKDOWN))

        assert response.status_code == 202
        body = response.json()
        assert body["documents"] == ["notes.md"]
        assert body["job_id"]
        assert (documents_dir / "notes.md").read_bytes() == MARKDOWN

    def test_several_files_share_one_job(self, client: TestClient, documents_dir: Path) -> None:
        response = upload(client, ("a.md", MARKDOWN), ("b.txt", b"plain"), ("c.html", b"<p>x"))

        assert response.status_code == 202
        assert response.json()["documents"] == ["a.md", "b.txt", "c.html"]
        assert {p.name for p in documents_dir.iterdir()} == {"a.md", "b.txt", "c.html"}

    def test_requires_at_least_one_file(self, client: TestClient) -> None:
        assert client.post("/ingest").status_code == 422

    def test_reuploading_a_name_overwrites_it(
        self, client: TestClient, documents_dir: Path
    ) -> None:
        upload(client, ("notes.md", b"first version"))
        upload(client, ("notes.md", b"second version"))

        assert (documents_dir / "notes.md").read_bytes() == b"second version"
        assert len(list(documents_dir.iterdir())) == 1


class TestUploadRejections:
    def test_a_traversing_filename_is_stored_as_a_plain_basename(
        self, client: TestClient, documents_dir: Path
    ) -> None:
        response = upload(client, ("../../etc/passwd.md", MARKDOWN))

        assert response.status_code == 202
        assert response.json()["documents"] == ["passwd.md"]
        assert (documents_dir / "passwd.md").exists()
        assert not (documents_dir.parent.parent / "etc" / "passwd.md").exists()

    @pytest.mark.parametrize("name", ["payload.exe", "archive.zip", "noextension"])
    def test_rejects_formats_outside_the_allowlist(
        self, client: TestClient, documents_dir: Path, name: str
    ) -> None:
        response = upload(client, (name, MARKDOWN))

        assert response.status_code == 415
        assert list(documents_dir.iterdir()) == []

    def test_rejects_content_that_is_not_text(
        self, client: TestClient, documents_dir: Path
    ) -> None:
        response = upload(client, ("image.md", b"\x89PNG\r\n\x1a\n\xff\xfe binary"))

        assert response.status_code == 415
        assert list(documents_dir.iterdir()) == []

    def test_rejects_an_oversized_upload(
        self, make_client: Callable[..., TestClient], documents_dir: Path
    ) -> None:
        client = make_client(max_upload_bytes=1024)

        response = upload(client, ("big.md", b"x" * 4096))

        assert response.status_code == 413
        assert list(documents_dir.iterdir()) == [], "no partial file may be left behind"

    def test_one_bad_file_rejects_the_whole_batch(
        self, client: TestClient, documents_dir: Path
    ) -> None:
        response = upload(client, ("good.md", MARKDOWN), ("bad.exe", MARKDOWN))

        assert response.status_code == 415
        assert list(documents_dir.iterdir()) == [], "the batch must not land half-applied"


class TestJobStatus:
    def test_reports_the_job_created_by_an_upload(self, client: TestClient) -> None:
        job_id = upload(client, ("notes.md", MARKDOWN)).json()["job_id"]

        response = client.get(f"/jobs/{job_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["job_id"] == job_id
        assert body["documents"] == ["notes.md"]
        assert body["total"] == 1

    def test_unknown_job_is_a_404(self, client: TestClient) -> None:
        assert client.get("/jobs/nope").status_code == 404

"""HTTP client wrapping the Open WebUI Knowledge Base sync API."""

from __future__ import annotations

import json
from typing import Any

import httpx


class OikbClient:
    """Stateless HTTP client for the Open WebUI KB API.

    All methods are synchronous — httpx handles connection pooling internally.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 120.0):
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=f"{self._base_url}/api/v1",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

    def __enter__(self) -> OikbClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # ── Sync API ────────────────────────────────────────────────

    def sync_diff(
        self,
        kb_id: str,
        manifest: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """POST /knowledge/{id}/sync/diff — compute diff from manifest."""
        resp = self._http.post(
            f"/knowledge/{kb_id}/sync/diff",
            json={"manifest": manifest},
        )
        resp.raise_for_status()
        return resp.json()

    def sync_cleanup(
        self,
        kb_id: str,
        file_ids: list[str],
        dir_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """POST /knowledge/{id}/sync/cleanup — remove stale files and dirs."""
        payload: dict[str, Any] = {"file_ids": file_ids}
        if dir_ids:
            payload["dir_ids"] = dir_ids
        resp = self._http.post(
            f"/knowledge/{kb_id}/sync/cleanup",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    # ── File upload ─────────────────────────────────────────────

    def upload_file(
        self,
        file_content: bytes,
        filename: str,
        kb_id: str,
        file_hash: str,
        directory_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /files/ — upload a single file to the KB.

        `extra_metadata` (source-specific, from `BaseConnector.file_metadata`)
        is merged into the multipart `metadata` payload so it lands on the
        file row and can be picked up by Filter Functions rendering citations.
        Reserved keys (`knowledge_id`, `file_hash`, `directory_id`) always win.
        """

        metadata: dict[str, Any] = {}
        if extra_metadata:
            metadata.update(extra_metadata)
        metadata["knowledge_id"] = kb_id
        metadata["file_hash"] = file_hash
        if directory_id:
            metadata["directory_id"] = directory_id

        # ?process_in_background=false blocks until embedding + KB link
        # complete. Upstream default is True, which returns 200 while the
        # file is still `pending` — a subsequent /sync/diff can then race
        # against the background task and see stale state.
        resp = self._http.post(
            "/files/",
            params={"process_in_background": "false"},
            files={"file": (filename, file_content)},
            data={"metadata": json.dumps(metadata)},
        )
        resp.raise_for_status()
        return resp.json()

    # ── Directory management ────────────────────────────────────

    def create_directory(
        self,
        kb_id: str,
        name: str,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """POST /knowledge/{id}/dirs/create — create a directory."""
        payload: dict[str, Any] = {"name": name}
        if parent_id:
            payload["parent_id"] = parent_id
        resp = self._http.post(
            f"/knowledge/{kb_id}/dirs/create",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    # ── KB management ───────────────────────────────────────────

    def reset_kb(
        self,
        kb_id: str,
        include_directories: bool = True,
    ) -> dict[str, Any]:
        """POST /knowledge/{id}/reset — reset the KB."""
        resp = self._http.post(
            f"/knowledge/{kb_id}/reset",
            params={"include_directories": include_directories},
        )
        resp.raise_for_status()
        return resp.json()

    def get_kb(self, kb_id: str) -> dict[str, Any]:
        """GET /knowledge/{id} — get KB info."""
        resp = self._http.get(f"/knowledge/{kb_id}")
        resp.raise_for_status()
        return resp.json()

    def list_kb_files(
        self,
        kb_id: str,
        page_size: int = 1000,
    ) -> list[dict[str, Any]]:
        """GET /knowledge/{id}/files — every file in the KB, with meta.

        Walks the paginated endpoint (admins can override `limit`, so we ask
        for a big page to keep round trips low). Each returned item carries
        `.meta.data` — whatever the connector attached at upload time — which
        is what the meta-drift detection in sync.py needs.
        """
        files: list[dict[str, Any]] = []
        page = 1
        while True:
            resp = self._http.get(
                f"/knowledge/{kb_id}/files",
                params={"page": page, "limit": page_size},
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            files.extend(items)
            if not items or len(files) >= data.get("total", 0):
                break
            page += 1
        return files

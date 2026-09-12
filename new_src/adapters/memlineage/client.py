"""
A minimal HTTP client for a running MemLineage backend.

Deliberately stdlib-only and proxy-free. `urllib` is used instead of httpx because the
project's proxy environment variables break local connections, and an adapter that silently
picks up a proxy would fail in a way that looks like MemLineage being down.

Nothing here is MemLineage-specific beyond the base URL and the `KMS_API_KEY` header its
own integration contract defines (INTEGRATION.md, "Runtime env").
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_BASE_URL = os.getenv("KMS_BASE_URL", "http://127.0.0.1:8077")
DEFAULT_API_KEY = os.getenv("KMS_API_KEY", "dev-api-key")


class MemLineageError(RuntimeError):
    pass


class MemLineageClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        # An opener with an empty ProxyHandler ignores HTTP_PROXY/ALL_PROXY entirely.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, method: str, path: str, payload: dict | None = None):
        url = f"{self.base_url}{path}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, method=method)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if self.api_key:
            # MemLineage's own header name; harmless when AFKMS_REQUIRE_AUTH=false.
            request.add_header("X-API-Key", self.api_key)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:  # surface the backend's own error body
            detail = error.read().decode("utf-8", "replace")
            raise MemLineageError(f"{method} {path} -> HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise MemLineageError(
                f"{method} {path} -> cannot reach MemLineage at {self.base_url}: {error.reason}. "
                "Start its backend first (see new_src/adapters/memlineage/README.md)."
            ) from error
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def get(self, path: str):
        return self._request("GET", path)

    def post(self, path: str, payload: dict):
        return self._request("POST", path, payload)

    def delete(self, path: str):
        return self._request("DELETE", path)

    def health(self) -> bool:
        try:
            return bool((self.get("/health") or {}).get("ok"))
        except MemLineageError:
            return False

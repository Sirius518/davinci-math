from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pipeline.utils.jsonx import dumps as json_dumps
from pipeline.utils.jsonx import loads as json_loads


@dataclass(slots=True)
class InferenceRequest:
    model: str
    prompt: str
    system_prompt: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    stop: list[str] = field(default_factory=list)
    extra_body: dict[str, Any] = field(default_factory=dict)

    def cache_key(self, backend: str) -> str:
        payload = asdict(self)
        payload["backend"] = backend
        text = json_dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class InferenceResponse:
    text: str
    raw: dict[str, Any]
    cached: bool = False


@dataclass(slots=True)
class BackendConfig:
    backend: str
    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: int = 120
    max_retries: int = 3
    cache_dir: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackendConfig":
        return cls(
            backend=str(data.get("backend", "")),
            base_url=str(data.get("base_url", "")),
            model=str(data.get("model", "")),
            api_key=str(data.get("api_key", "")),
            timeout_seconds=int(data.get("timeout_seconds", 120)),
            max_retries=int(data.get("max_retries", 3)),
            cache_dir=str(data.get("cache_dir", "")),
            headers=dict(data.get("headers", {})),
        )


class InferenceClient(ABC):
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def backend_name(self) -> str:
        return self.config.backend

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        cached = self._read_cache(request)
        if cached is not None:
            return cached
        attempt = 0
        last_error: Exception | None = None
        while attempt < self.config.max_retries:
            try:
                response = self._infer_once(request)
                self._write_cache(request, response)
                return response
            except Exception as error:  # pragma: no cover
                last_error = error
                attempt += 1
                if attempt >= self.config.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Inference failed after retries: {last_error}") from last_error

    @abstractmethod
    def _infer_once(self, request: InferenceRequest) -> InferenceResponse:
        raise NotImplementedError

    def _read_cache(self, request: InferenceRequest) -> InferenceResponse | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{request.cache_key(self.backend_name)}.json"
        if not path.exists():
            return None
        data = json_loads(path.read_text(encoding="utf-8"))
        return InferenceResponse(text=str(data.get("text", "")), raw=dict(data.get("raw", {})), cached=True)

    def _write_cache(self, request: InferenceRequest, response: InferenceResponse) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{request.cache_key(self.backend_name)}.json"
        payload = {"text": response.text, "raw": response.raw}
        path.write_text(json_dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", **self.config.headers}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        body = json_dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json_loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:  # pragma: no cover
            detail = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error

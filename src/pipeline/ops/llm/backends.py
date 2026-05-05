from __future__ import annotations

from typing import Any

from pipeline.core.io import load_yaml
from .client import BackendConfig, InferenceClient, InferenceRequest, InferenceResponse


class SGLangClient(InferenceClient):
    def _infer_once(self, request: InferenceRequest) -> InferenceResponse:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, Any] = {
            "model": request.model or self.config.model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        if request.stop:
            payload["stop"] = request.stop
        payload.update(request.extra_body)
        raw = self._post_json(f"{self.config.base_url.rstrip('/')}/v1/chat/completions", payload)
        text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        return InferenceResponse(text=str(text), raw=raw)


class VLLMClient(InferenceClient):
    def _infer_once(self, request: InferenceRequest) -> InferenceResponse:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, Any] = {
            "model": request.model or self.config.model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        if request.stop:
            payload["stop"] = request.stop
        payload.update(request.extra_body)
        raw = self._post_json(f"{self.config.base_url.rstrip('/')}/v1/chat/completions", payload)
        text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        return InferenceResponse(text=str(text), raw=raw)


def build_inference_client(config: BackendConfig) -> InferenceClient:
    backend = config.backend.lower().strip()
    if backend == "sglang":
        return SGLangClient(config)
    if backend == "vllm":
        return VLLMClient(config)
    raise ValueError(f"Unsupported backend: {config.backend}")


def load_backend_config(path: str) -> BackendConfig:
    return BackendConfig.from_dict(load_yaml(path))

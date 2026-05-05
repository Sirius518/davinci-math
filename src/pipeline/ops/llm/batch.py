from __future__ import annotations

from typing import Iterable

from .client import InferenceClient, InferenceRequest, InferenceResponse


def batch_infer(client: InferenceClient, requests: Iterable[InferenceRequest]) -> list[InferenceResponse]:
    return [client.infer(request) for request in requests]

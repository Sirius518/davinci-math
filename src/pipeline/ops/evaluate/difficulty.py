from __future__ import annotations

import re
from typing import Any

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, DatasetProcessor, DatasetProcessorResult
from pipeline.ops.llm.backends import build_inference_client, load_backend_config
from pipeline.ops.llm.client import InferenceClient, InferenceRequest


SCORE_PATTERN = re.compile(r"(?<!\d)(10|[1-9])(?:\.\d+)?(?!\d)")


def annotate_difficulty(
    client: InferenceClient,
    records: list[CanonicalRecord],
    *,
    model: str,
) -> list[CanonicalRecord]:
    annotated: list[CanonicalRecord] = []
    for record in records:
        response = client.infer(
            InferenceRequest(
                model=model,
                prompt=(
                    "Rate the following math problem difficulty on an AoPS-like 1-10 scale. "
                    "Return only the number.\n\n"
                    f"{record.question}"
                ),
                temperature=0.0,
            )
        )
        match = SCORE_PATTERN.search(response.text)
        difficulty = float(match.group(1)) if match else 0.0
        verification = dict(record.verification)
        verification["difficulty_annotation"] = {
            "score": difficulty,
            "backend": client.backend_name,
            "model": model,
            "raw_response": response.text,
        }
        updated = record.clone(verification=verification)
        updated.add_trace(stage="evaluate", processor="difficulty_annotation", status="kept", details={"score": difficulty})
        annotated.append(updated)
    return annotated


@register_processor("difficulty_annotation")
class DifficultyAnnotationProcessor(DatasetProcessor):
    name = "difficulty_annotation"

    def process(self, records: list[CanonicalRecord], *, pipeline_artifacts: dict[str, Any] | None = None) -> DatasetProcessorResult:
        model_config = self.config.get("model_config")
        if not model_config:
            return DatasetProcessorResult(kept_records=records)
        backend = load_backend_config(str(model_config))
        client = build_inference_client(backend)
        model = str(self.config.get("model", backend.model))
        return DatasetProcessorResult(kept_records=annotate_difficulty(client, records, model=model))

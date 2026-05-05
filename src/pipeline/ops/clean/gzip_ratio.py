from __future__ import annotations

import gzip

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor


def _resolve_text(record: CanonicalRecord, source_field: str) -> str:
    if source_field == "question":
        return record.question
    if source_field in {"cot_text", "distillation"}:
        parts: list[str] = []
        for payload in dict(record.distillation).values():
            if not isinstance(payload, dict):
                continue
            text = str(payload.get("reasoning", payload.get("solution", ""))).strip()
            if text:
                parts.append(text)
        return "\n".join(parts) or record.question
    if source_field in {"generated_responses", "verification_samples"}:
        parts: list[str] = []
        for item in list(dict(record.verification).get("samples", [])):
            if not isinstance(item, dict):
                continue
            text = str(item.get("reasoning", item.get("solution", ""))).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(record.meta.get(source_field, ""))


def gzip_ratio(text: str) -> float:
    payload = text.encode("utf-8")
    if not payload:
        return 1.0
    compressed = gzip.compress(payload)
    return len(compressed) / len(payload)


@register_processor("gzip_ratio_annotator")
class GzipRatioAnnotator(RecordProcessor):
    name = "gzip_ratio_annotator"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        source_field = str(self.config.get("source_field", "cot_text"))
        output_key = str(self.config.get("output_key", "gzip_ratio"))
        text = _resolve_text(record, source_field)
        ratio = gzip_ratio(text)
        updated_meta = dict(record.meta)
        updated_meta[output_key] = ratio
        updated = record.clone(meta=updated_meta)
        updated.add_trace(
            stage="clean",
            processor=self.name,
            status="kept",
            details={"source_field": source_field, "output_key": output_key, "value": ratio},
        )
        return ProcessorResult(keep=True, record=updated, stage="clean", processor=self.name)

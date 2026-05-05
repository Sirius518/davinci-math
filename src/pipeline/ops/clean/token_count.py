from __future__ import annotations

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor

import tiktoken  # type: ignore[reportMissingImports]


_ENCODING_CACHE: dict[str, object] = {}


def _encoding_name(tokenizer: str) -> str:
    lowered = tokenizer.strip().lower()
    if lowered in {"cl100k", "cl100k_base"}:
        return "cl100k_base"
    if lowered in {"p50k", "p50k_base"}:
        return "p50k_base"
    raise ValueError(f"Unsupported tokenizer: {tokenizer}")


def _get_encoding(tokenizer: str) -> object:
    name = _encoding_name(tokenizer)
    cached = _ENCODING_CACHE.get(name)
    if cached is not None:
        return cached
    encoding = tiktoken.get_encoding(name)
    _ENCODING_CACHE[name] = encoding
    return encoding


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
        return "\n".join(parts)
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


def token_count(text: str, *, tokenizer: str) -> int:
    encoding = _get_encoding(tokenizer)
    return len(encoding.encode(text))  # type: ignore[attr-defined]


@register_processor("token_count_annotator")
class TokenCountAnnotator(RecordProcessor):
    name = "token_count_annotator"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        tokenizer = str(self.config.get("tokenizer", "cl100k_base"))
        source_field = str(self.config.get("source_field", "question"))
        output_key = str(self.config.get("output_key", "token_count"))
        text = _resolve_text(record, source_field)
        count = token_count(text, tokenizer=tokenizer)
        updated_meta = dict(record.meta)
        updated_meta[output_key] = count
        updated = record.clone(meta=updated_meta)
        updated.add_trace(
            stage="clean",
            processor=self.name,
            status="kept",
            details={"source_field": source_field, "output_key": output_key, "tokenizer": tokenizer, "value": count},
        )
        return ProcessorResult(keep=True, record=updated, stage="clean", processor=self.name)

from __future__ import annotations

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor
from pipeline.utils.text import normalize_answer_text

import tiktoken  # type: ignore[reportMissingImports]


_ENCODING_CACHE: dict[str, object] = {}


def _encoding_name(tokenizer: str) -> str:
    lowered = tokenizer.strip().lower()
    if lowered in {"char", "character"}:
        return "char"
    if lowered in {"cl100k", "cl100k_base"}:
        return "cl100k_base"
    if lowered in {"p50k", "p50k_base"}:
        return "p50k_base"
    raise ValueError(f"Unsupported tokenizer: {tokenizer}")


def _get_encoding(tokenizer: str) -> object | None:
    name = _encoding_name(tokenizer)
    if name == "char":
        return None
    cached = _ENCODING_CACHE.get(name)
    if cached is not None:
        return cached
    encoding = tiktoken.get_encoding(name)
    _ENCODING_CACHE[name] = encoding
    return encoding


def _texts_for_detection(record: CanonicalRecord, *, inspect_cot: bool, inspect_generated: bool) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    if inspect_cot:
        for model_name, payload in dict(record.distillation).items():
            if not isinstance(payload, dict):
                continue
            text = str(payload.get("reasoning", payload.get("solution", ""))).strip()
            if text:
                texts.append((f"distillation:{model_name}", text))
    if inspect_generated:
        for index, payload in enumerate(list(dict(record.verification).get("samples", []))):
            if not isinstance(payload, dict):
                continue
            text = str(payload.get("reasoning", payload.get("solution", ""))).strip()
            if text:
                texts.append((f"verification:samples:{index}", text))
    return texts


def _tokenize(text: str, *, tokenizer: str) -> list[int | str]:
    encoding = _get_encoding(tokenizer)
    if encoding is not None:
        return list(encoding.encode(text))  # type: ignore[attr-defined]
    normalized = normalize_answer_text(text)
    return [part for part in normalized.split(" ") if part]


def _max_ngram_repetition(tokens: list[int | str], *, ngram_size: int) -> int:
    if ngram_size <= 0 or len(tokens) < ngram_size * 2:
        return 1
    best = 1
    for start in range(0, len(tokens) - ngram_size + 1):
        pattern = tokens[start : start + ngram_size]
        run = 1
        cursor = start + ngram_size
        while cursor + ngram_size <= len(tokens):
            if tokens[cursor : cursor + ngram_size] != pattern:
                break
            run += 1
            cursor += ngram_size
        if run > best:
            best = run
    return best


@register_processor("ngram_repetition_filter")
class NgramRepetitionFilterProcessor(RecordProcessor):
    name = "ngram_repetition_filter"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        repeat_threshold = max(int(self.config.get("repeat_threshold", 16)), 2)
        ngram_size = max(int(self.config.get("ngram_size", 4)), 1)
        tokenizer = str(self.config.get("tokenizer", "cl100k_base"))
        inspect_cot = bool(self.config.get("inspect_cot", True))
        inspect_generated = bool(self.config.get("inspect_generated_responses", True))
        for source_name, text in _texts_for_detection(
            record,
            inspect_cot=inspect_cot,
            inspect_generated=inspect_generated,
        ):
            tokens = _tokenize(text, tokenizer=tokenizer)
            max_repeat = _max_ngram_repetition(tokens, ngram_size=ngram_size)
            if max_repeat >= repeat_threshold:
                updated = record.clone()
                updated.add_trace(
                    stage="clean",
                    processor=self.name,
                    status="dropped",
                    reason_code="ngram_repetition",
                    details={
                        "source": source_name,
                        "max_repeat": max_repeat,
                        "repeat_threshold": repeat_threshold,
                        "ngram_size": ngram_size,
                        "tokenizer": tokenizer,
                    },
                )
                return ProcessorResult(
                    keep=False,
                    record=updated,
                    stage="clean",
                    processor=self.name,
                    reason_code="ngram_repetition",
                    details={"source": source_name, "max_repeat": max_repeat},
                )
        updated = record.clone()
        updated.add_trace(stage="clean", processor=self.name, status="kept")
        return ProcessorResult(keep=True, record=updated, stage="clean", processor=self.name)

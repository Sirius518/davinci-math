from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
from pipeline.utils.jsonx import dumps as json_dumps
from pipeline.utils.jsonx import _LazyJsonDict, _LazyTraceList


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sorted_json(value: dict[str, Any] | list[Any] | None) -> str:
    if value is None:
        value = {}
    return json_dumps(value, ensure_ascii=False, sort_keys=True)


def _trace_event_from_value(event: Any) -> "TraceEvent":
    if isinstance(event, TraceEvent):
        return event
    payload = dict(event or {})
    return TraceEvent(
        stage=str(payload.get("stage", "")),
        processor=str(payload.get("processor", "")),
        status=str(payload.get("status", "")),
        reason_code=str(payload.get("reason_code", "")),
        details=dict(payload.get("details", {})),
        created_at=str(payload.get("created_at", utc_now_iso())),
    )


def _storage_json_text(value: object, default: str) -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def _column_values(table: pa.Table, name: str, default: object) -> list[object]:
    if name not in table.column_names:
        return [default] * table.num_rows
    return table.column(name).to_pylist()


def _legacy_distillation(legacy_cot: str) -> dict[str, Any]:
    return {
        "legacy": {
            "model": "legacy",
            "solution": "",
            "reasoning": legacy_cot,
            "quality_status": "valid",
            "quality_tags": [],
        }
    }


@dataclass(slots=True)
class TraceEvent:
    stage: str
    processor: str
    status: str
    reason_code: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CanonicalRecord:
    record_id: str
    question: str
    raw_dataset_answer: str
    confirmed_answer: str = ""
    dataset_name: str = ""
    training_phase: str = ""
    filter_tag: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
    distillation: dict[str, Any] = field(default_factory=dict)
    decontamination: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)

    def clone(self, **updates: Any) -> "CanonicalRecord":
        payload = self.to_dict()
        payload.update(updates)
        payload["trace"] = [TraceEvent(**event) for event in payload["trace"]]
        return CanonicalRecord.from_dict(payload)

    def add_trace(
        self,
        *,
        stage: str,
        processor: str,
        status: str,
        reason_code: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.trace.append(
            TraceEvent(
                stage=stage,
                processor=processor,
                status=status,
                reason_code=reason_code,
                details=details or {},
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "question": self.question,
            "raw_dataset_answer": self.raw_dataset_answer,
            "confirmed_answer": self.confirmed_answer,
            "dataset_name": self.dataset_name,
            "training_phase": self.training_phase,
            "filter_tag": self.filter_tag,
            "verification": dict(self.verification),
            "distillation": dict(self.distillation),
            "decontamination": dict(self.decontamination),
            "meta": dict(self.meta),
            "trace": [event.to_dict() for event in self.trace],
        }

    def to_storage_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {
            "record_id": payload["record_id"],
            "question": payload["question"],
            "raw_dataset_answer": payload["raw_dataset_answer"],
            "confirmed_answer": payload["confirmed_answer"],
            "dataset_name": payload["dataset_name"],
            "training_phase": payload["training_phase"],
            "filter_tag": payload["filter_tag"],
            "verification": _sorted_json(payload["verification"]),
            "distillation": _sorted_json(payload["distillation"]),
            "decontamination": _sorted_json(payload["decontamination"]),
            "meta": _sorted_json(payload["meta"]),
            "trace": _sorted_json(payload["trace"]),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalRecord":
        trace_events: list[TraceEvent] = []
        for event in value.get("trace", []):
            trace_events.append(_trace_event_from_value(event))
        return cls(
            record_id=str(value.get("record_id", "")),
            question=str(value.get("question", "")),
            raw_dataset_answer=str(value.get("raw_dataset_answer", value.get("ground_truth", ""))),
            confirmed_answer=str(
                value.get(
                    "confirmed_answer",
                    value.get("final_answer", value.get("raw_dataset_answer", value.get("ground_truth", ""))),
                )
            ),
            dataset_name=str(value.get("dataset_name", "")),
            training_phase=str(value.get("training_phase", "")),
            filter_tag=str(value.get("filter_tag", "")),
            verification=dict(value.get("verification", {})),
            distillation=dict(value.get("distillation", {})),
            decontamination=dict(value.get("decontamination", {})),
            meta=dict(value.get("meta", {})),
            trace=trace_events,
        )

    @classmethod
    def from_storage_dict(cls, value: dict[str, Any]) -> "CanonicalRecord":
        distillation_text = _storage_json_text(value.get("distillation", "{}"), "{}")
        legacy_cot = str(value.get("cot_text", "") or "")
        if distillation_text in {"{}", "null"} and legacy_cot:
            distillation: dict[str, Any] = _legacy_distillation(legacy_cot)
        else:
            distillation = _LazyJsonDict(distillation_text)
        return cls(
            record_id=str(value.get("record_id", "") or ""),
            question=str(value.get("question", "") or ""),
            raw_dataset_answer=str(value.get("raw_dataset_answer", value.get("ground_truth", "")) or ""),
            confirmed_answer=str(
                value.get(
                    "confirmed_answer",
                    value.get("final_answer", value.get("raw_dataset_answer", value.get("ground_truth", ""))),
                )
                or ""
            ),
            dataset_name=str(value.get("dataset_name", "") or ""),
            training_phase=str(value.get("training_phase", "") or ""),
            filter_tag=str(value.get("filter_tag", "") or ""),
            verification=_LazyJsonDict(_storage_json_text(value.get("verification", "{}"), "{}")),
            distillation=distillation,
            decontamination=_LazyJsonDict(_storage_json_text(value.get("decontamination", "{}"), "{}")),
            meta=_LazyJsonDict(_storage_json_text(value.get("meta", "{}"), "{}")),
            trace=_LazyTraceList(_storage_json_text(value.get("trace", "[]"), "[]"), _trace_event_from_value),
        )

    @classmethod
    def from_table(cls, table: pa.Table) -> list["CanonicalRecord"]:
        record_count = table.num_rows
        if record_count == 0:
            return []

        record_ids = _column_values(table, "record_id", "")
        questions = _column_values(table, "question", "")
        raw_answers = _column_values(table, "raw_dataset_answer", "")
        confirmed_answers = _column_values(table, "confirmed_answer", "")
        dataset_names = _column_values(table, "dataset_name", "")
        training_phases = _column_values(table, "training_phase", "")
        filter_tags = _column_values(table, "filter_tag", "")
        verification_texts = _column_values(table, "verification", "{}")
        distillation_texts = _column_values(table, "distillation", "{}")
        decontamination_texts = _column_values(table, "decontamination", "{}")
        meta_texts = _column_values(table, "meta", "{}")
        trace_texts = _column_values(table, "trace", "[]")
        legacy_cots = _column_values(table, "cot_text", "")

        records: list[CanonicalRecord] = []
        for index in range(record_count):
            distillation_text = _storage_json_text(distillation_texts[index], "{}")
            legacy_cot = str(legacy_cots[index] or "")
            if distillation_text in {"{}", "null"} and legacy_cot:
                distillation: dict[str, Any] = _legacy_distillation(legacy_cot)
            else:
                distillation = _LazyJsonDict(distillation_text)
            records.append(
                cls(
                    record_id=str(record_ids[index] or ""),
                    question=str(questions[index] or ""),
                    raw_dataset_answer=str(raw_answers[index] or ""),
                    confirmed_answer=str(confirmed_answers[index] or ""),
                    dataset_name=str(dataset_names[index] or ""),
                    training_phase=str(training_phases[index] or ""),
                    filter_tag=str(filter_tags[index] or ""),
                    verification=_LazyJsonDict(_storage_json_text(verification_texts[index], "{}")),
                    distillation=distillation,
                    decontamination=_LazyJsonDict(_storage_json_text(decontamination_texts[index], "{}")),
                    meta=_LazyJsonDict(_storage_json_text(meta_texts[index], "{}")),
                    trace=_LazyTraceList(_storage_json_text(trace_texts[index], "[]"), _trace_event_from_value),
                )
            )
        return records


@dataclass(slots=True)
class ProcessorResult:
    keep: bool
    record: CanonicalRecord
    stage: str
    processor: str
    reason_code: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetProcessorResult:
    kept_records: list[CanonicalRecord]
    processor_results: list[ProcessorResult] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DedupPair:
    left_id: str
    right_id: str
    left_question: str
    right_question: str
    method: str
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_storage_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = json_dumps(payload["metadata"], ensure_ascii=False, sort_keys=True)
        return payload


@dataclass(slots=True)
class DedupJudgement:
    left_id: str
    right_id: str
    keep_pair: bool
    decision: str
    raw_response: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_storage_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = json_dumps(payload["metadata"], ensure_ascii=False, sort_keys=True)
        return payload


@dataclass(slots=True)
class RolloutResult:
    record_id: str
    backend: str
    model: str
    prompt_name: str
    samples: list[str]
    parsed_answers: list[str]
    majority_answer: str
    majority_count: int
    is_correct: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_storage_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("samples", "parsed_answers", "metadata"):
            payload[key] = json_dumps(payload[key], ensure_ascii=False, sort_keys=True)
        return payload


class RecordProcessor(ABC):
    name: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def process(self, record: CanonicalRecord) -> ProcessorResult:
        raise NotImplementedError


class DatasetProcessor(ABC):
    name: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def process(
        self,
        records: list[CanonicalRecord],
        *,
        pipeline_artifacts: dict[str, Any] | None = None,
    ) -> DatasetProcessorResult:
        raise NotImplementedError


CANONICAL_RECORD_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string()),
        pa.field("question", pa.string()),
        pa.field("raw_dataset_answer", pa.string()),
        pa.field("confirmed_answer", pa.string()),
        pa.field("dataset_name", pa.string()),
        pa.field("training_phase", pa.string()),
        pa.field("filter_tag", pa.string()),
        pa.field("verification", pa.large_string()),
        pa.field("distillation", pa.large_string()),
        pa.field("decontamination", pa.large_string()),
        pa.field("meta", pa.large_string()),
        pa.field("trace", pa.large_string()),
    ]
)

PROCESSOR_RESULT_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string()),
        pa.field("keep", pa.bool_()),
        pa.field("stage", pa.string()),
        pa.field("processor", pa.string()),
        pa.field("reason_code", pa.string()),
        pa.field("details", pa.large_string()),
    ]
)

DEDUP_PAIR_SCHEMA = pa.schema(
    [
        pa.field("left_id", pa.string()),
        pa.field("right_id", pa.string()),
        pa.field("left_question", pa.string()),
        pa.field("right_question", pa.string()),
        pa.field("method", pa.string()),
        pa.field("similarity", pa.float64()),
        pa.field("metadata", pa.large_string()),
    ]
)

DEDUP_JUDGEMENT_SCHEMA = pa.schema(
    [
        pa.field("left_id", pa.string()),
        pa.field("right_id", pa.string()),
        pa.field("keep_pair", pa.bool_()),
        pa.field("decision", pa.string()),
        pa.field("raw_response", pa.large_string()),
        pa.field("metadata", pa.large_string()),
    ]
)

ROLLOUT_RESULT_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string()),
        pa.field("backend", pa.string()),
        pa.field("model", pa.string()),
        pa.field("prompt_name", pa.string()),
        pa.field("samples", pa.large_string()),
        pa.field("parsed_answers", pa.large_string()),
        pa.field("majority_answer", pa.string()),
        pa.field("majority_count", pa.int64()),
        pa.field("is_correct", pa.bool_()),
        pa.field("metadata", pa.large_string()),
    ]
)

STAGE_REASON_CODES = {
    "non_english_question",
    "short_question",
    "mcq_conversion_failed",
    "rule_verify_failed",
    "exact_duplicate",
    "near_duplicate",
    "dedup_model_duplicate",
    "cot_non_english",
    "cot_truncated",
    "ngram_repetition",
    "pass_ratio_too_low",
    "pass_ratio_too_high",
}

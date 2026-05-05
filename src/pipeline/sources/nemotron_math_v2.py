from __future__ import annotations

from typing import Any

import pyarrow.dataset as ds

from pipeline.core.schema import CanonicalRecord
from pipeline.sources.base import BaseSourceAdapter
from pipeline.utils.hashing import sha256_text


def _normalize_string(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _extract_assistant_message(messages: list[dict[str, Any]] | None) -> tuple[str, str]:
    for message in messages or []:
        if str((message or {}).get("role", "")).strip().lower() != "assistant":
            continue
        return (
            _normalize_string((message or {}).get("reasoning_content")),
            _normalize_string((message or {}).get("content")),
        )
    return "", ""


class NemotronMathV2Adapter(BaseSourceAdapter):
    def load(self) -> list[CanonicalRecord]:
        options = self.config.options or {}
        source_configs = options.get("source_configs", [])
        if not source_configs:
            raise ValueError("Nemotron-Math-v2 adapter requires options.source_configs")

        aggregated: dict[str, dict[str, Any]] = {}
        for source_config in source_configs:
            source_path = self.config.input / str(source_config["file_name"])
            model_name = str(source_config["model"])
            dataset = ds.dataset(source_path, format="parquet")
            scanner = dataset.scanner(
                batch_size=int(options.get("batch_size", 20_000)),
                columns=["problem", "expected_answer", "data_source", "license", "messages"],
                use_threads=True,
            )
            for batch in scanner.to_batches():
                for row in batch.to_pylist():
                    question = _normalize_string(row.get("problem"))
                    record_id = sha256_text(question)
                    reasoning_content, content = _extract_assistant_message(row.get("messages"))
                    item = aggregated.setdefault(
                        record_id,
                        {
                            "question": question,
                            "ground_truth": _normalize_string(row.get("expected_answer")),
                            "data_source": _normalize_string(row.get("data_source")),
                            "license": _normalize_string(row.get("license")),
                            "cot_solutions": [],
                        },
                    )
                    item["ground_truth"] = item["ground_truth"] or _normalize_string(row.get("expected_answer"))
                    item["data_source"] = item["data_source"] or _normalize_string(row.get("data_source"))
                    item["license"] = item["license"] or _normalize_string(row.get("license"))
                    item["cot_solutions"].append(
                        {
                            "model": model_name,
                            "reasoning_content": reasoning_content,
                            "content": content,
                        }
                    )

        records: list[CanonicalRecord] = []
        for record_id, item in aggregated.items():
            cot_solutions = sorted(item["cot_solutions"], key=lambda current: current.get("model", ""))
            distillation: dict[str, Any] = {}
            for solution in cot_solutions:
                model_name = str(solution.get("model", "")).strip()
                if not model_name:
                    continue
                distillation[model_name] = {
                    "model": model_name,
                    "solution": str(solution.get("content", "")),
                    "reasoning": str(solution.get("reasoning_content", "")),
                    "quality_status": "valid",
                    "quality_tags": [],
                }
            records.append(
                CanonicalRecord(
                    record_id=record_id,
                    question=str(item["question"]),
                    raw_dataset_answer=str(item["ground_truth"]),
                    confirmed_answer=str(item["ground_truth"]),
                    dataset_name=self.config.dataset_name,
                    training_phase="",
                    filter_tag="",
                    verification={},
                    distillation=distillation,
                    decontamination={},
                    meta={},
                )
            )
        return records

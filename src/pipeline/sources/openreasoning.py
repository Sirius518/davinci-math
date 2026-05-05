from __future__ import annotations

import json
from typing import Any

from pipeline.core.schema import CanonicalRecord
from pipeline.sources.base import BaseSourceAdapter
from pipeline.utils.hashing import sha256_text


def _normalize_message(message: dict[str, Any] | None) -> dict[str, Any]:
    item = message or {}
    ground_truth = item.get("ground_truth")
    if isinstance(ground_truth, dict):
        normalized_ground_truth: dict[str, Any] | None = {"value": str(ground_truth.get("value", ""))}
    else:
        normalized_ground_truth = None
    return {
        "from": str(item.get("from", "")),
        "value": str(item.get("value", "")),
        "ground_truth": normalized_ground_truth,
    }


def _extract_question_and_ground_truth(messages: list[dict[str, Any]]) -> tuple[str, str]:
    question = ""
    ground_truth = ""
    for message in messages:
        role = str(message.get("from", "")).strip().lower()
        if role == "human" and not question:
            question = str(message.get("value", ""))
        elif role == "assistant" and not ground_truth:
            answer = message.get("ground_truth") or {}
            if isinstance(answer, dict):
                ground_truth = str(answer.get("value", ""))
    return question, ground_truth


class OpenReasoningAdapter(BaseSourceAdapter):
    def load(self) -> list[CanonicalRecord]:
        input_path = self.config.input
        if input_path.is_dir():
            file_names = self.config.options.get("source_files", []) if self.config.options else []
            paths = [input_path / file_name for file_name in file_names]
        else:
            paths = [input_path]

        records: list[CanonicalRecord] = []
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            for sample in data:
                messages = [_normalize_message(message) for message in sample]
                question, ground_truth = _extract_question_and_ground_truth(messages)
                records.append(
                    CanonicalRecord(
                        record_id=sha256_text(question),
                        question=question,
                        raw_dataset_answer=ground_truth,
                        confirmed_answer=ground_truth,
                        dataset_name=self.config.dataset_name,
                        training_phase="",
                        filter_tag="",
                        verification={},
                        distillation={},
                        decontamination={},
                        meta={},
                    )
                )
        return records

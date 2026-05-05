from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, DatasetProcessor, DatasetProcessorResult, DedupJudgement, DedupPair
from pipeline.utils.hashing import fast_text_hash


def exact_deduplicate(
    records: list[CanonicalRecord],
    *,
    priority_map: dict[str, int] | None = None,
) -> tuple[list[CanonicalRecord], list[DedupPair]]:
    seen_ids: set[str] = set()
    seen_question_hashes: dict[str, CanonicalRecord] = {}
    kept: list[CanonicalRecord] = []
    duplicates: list[DedupPair] = []
    _prio = priority_map or {}

    def _sort_key(item: CanonicalRecord) -> tuple[int, str]:
        return (_prio.get(item.record_id, 0), item.record_id)

    for record in sorted(records, key=_sort_key):
        question_hash = fast_text_hash(record.question.strip().lower())
        if record.record_id in seen_ids:
            duplicates.append(
                DedupPair(
                    left_id=record.record_id,
                    right_id=record.record_id,
                    left_question="",
                    right_question="",
                    method="exact",
                    similarity=1.0,
                )
            )
            continue
        if question_hash in seen_question_hashes:
            original = seen_question_hashes[question_hash]
            duplicates.append(
                DedupPair(
                    left_id=original.record_id,
                    right_id=record.record_id,
                    left_question="",
                    right_question="",
                    method="exact",
                    similarity=1.0,
                )
            )
            continue
        seen_ids.add(record.record_id)
        seen_question_hashes[question_hash] = record
        kept.append(record)
    return kept, duplicates


def _union_find_drop_ids(
    edges: list[tuple[str, str]],
    *,
    priority_map: dict[str, int] | None = None,
) -> set[str]:
    _prio = priority_map or {}
    id_to_index: dict[str, int] = {}
    index_to_id: list[str] = []
    for left, right in edges:
        if left not in id_to_index:
            id_to_index[left] = len(index_to_id)
            index_to_id.append(left)
        if right not in id_to_index:
            id_to_index[right] = len(index_to_id)
            index_to_id.append(right)
    parent = list(range(len(index_to_id)))
    rank = [0] * len(index_to_id)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1

    for left, right in edges:
        union(id_to_index[left], id_to_index[right])

    root_best_id: dict[int, str] = {}
    for index, record_id in enumerate(index_to_id):
        root = find(index)
        current = root_best_id.get(root)
        if current is None or (_prio.get(record_id, 0), record_id) < (_prio.get(current, 0), current):
            root_best_id[root] = record_id

    to_drop: set[str] = set()
    for index, record_id in enumerate(index_to_id):
        root = find(index)
        if record_id != root_best_id[root]:
            to_drop.add(record_id)
    return to_drop


def postprocess_duplicates(
    records: list[CanonicalRecord],
    exact_duplicates: list[DedupPair],
    candidate_pairs: list[DedupPair],
    judgements: list[DedupJudgement],
    *,
    priority_map: dict[str, int] | None = None,
) -> list[CanonicalRecord]:
    edges = [(item.left_id, item.right_id) for item in exact_duplicates if item.left_id != item.right_id]
    edges.extend((item.left_id, item.right_id) for item in candidate_pairs if item.method == "minhash")
    edges.extend((item.left_id, item.right_id) for item in judgements if item.keep_pair)
    if not edges:
        return sorted(records, key=lambda item: item.record_id)
    to_drop = _union_find_drop_ids(edges, priority_map=priority_map)
    return [record for record in sorted(records, key=lambda item: item.record_id) if record.record_id not in to_drop]


@dataclass(slots=True)
class ExactDedupOutput:
    kept_records: list[CanonicalRecord]
    exact_duplicates: list[DedupPair]


@register_processor("exact_dedup")
class ExactDedupProcessor(DatasetProcessor):
    name = "exact_dedup"

    def process(self, records: list[CanonicalRecord], *, pipeline_artifacts: dict[str, Any] | None = None) -> DatasetProcessorResult:
        from pipeline.utils.priority import build_record_priority_map, parse_tier_config

        tier_config = parse_tier_config(self.config.get("dataset_priority"))
        priority_map: dict[str, int] | None = None
        if tier_config is not None:
            priority_map = build_record_priority_map(records, tier_config)
        kept, duplicates = exact_deduplicate(records, priority_map=priority_map)
        return DatasetProcessorResult(kept_records=kept, artifacts={"exact_duplicates": duplicates})

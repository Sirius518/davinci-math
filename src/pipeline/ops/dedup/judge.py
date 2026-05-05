from __future__ import annotations

from typing import Any

from pipeline.core.registry import register_processor
from pipeline.core.schema import (
    CanonicalRecord,
    DatasetProcessor,
    DatasetProcessorResult,
    DedupJudgement,
    DedupPair,
)
from pipeline.ops.dedup._minhash_engine import VerifyResult
from pipeline.ops.dedup.exact import postprocess_duplicates
from pipeline.ops.llm.client import InferenceClient, InferenceRequest
from pipeline.ops.llm.backends import build_inference_client, load_backend_config


def judge_candidate_pairs(
    client: InferenceClient,
    pairs: list[DedupPair],
    *,
    model: str,
) -> list[DedupJudgement]:
    judgements: list[DedupJudgement] = []
    for pair in pairs:
        prompt = (
            "You are judging whether two dataset questions are duplicates.\n"
            "Answer with exactly one line: Decision: YES or Decision: NO.\n\n"
            f"Question A:\n{pair.left_question}\n\n"
            f"Question B:\n{pair.right_question}\n"
        )
        response = client.infer(InferenceRequest(model=model, prompt=prompt, temperature=0.0))
        keep_pair = "Decision: YES" in response.text
        judgements.append(
            DedupJudgement(
                left_id=pair.left_id,
                right_id=pair.right_id,
                keep_pair=keep_pair,
                decision="YES" if keep_pair else "NO",
                raw_response=response.text,
                metadata={"method": pair.method, "similarity": pair.similarity, "cached": response.cached},
            )
        )
    return judgements


def _collect_candidate_pairs(pipeline_artifacts: dict[str, Any] | None) -> list[DedupPair]:
    if not pipeline_artifacts:
        return []
    pairs: list[DedupPair] = []
    for step_artifacts in pipeline_artifacts.values():
        if not isinstance(step_artifacts, dict):
            continue
        for key in ("candidate_pairs", "exact_duplicates"):
            items = step_artifacts.get(key, [])
            if isinstance(items, list):
                pairs.extend(item for item in items if isinstance(item, DedupPair))
            elif isinstance(items, VerifyResult):
                pairs.extend(items.to_dedup_pairs())
    return pairs


@register_processor("dedup_judge")
class DedupJudgeProcessor(DatasetProcessor):
    name = "dedup_judge"

    def process(
        self,
        records: list[CanonicalRecord],
        *,
        pipeline_artifacts: dict[str, Any] | None = None,
    ) -> DatasetProcessorResult:
        model_config = self.config.get("model_config")
        if not model_config:
            return DatasetProcessorResult(kept_records=records, artifacts={})

        backend = load_backend_config(str(model_config))
        client = build_inference_client(backend)
        model = str(self.config.get("model", backend.model))

        candidate_pairs = _collect_candidate_pairs(pipeline_artifacts)
        if not candidate_pairs:
            return DatasetProcessorResult(kept_records=records, artifacts={"judgements": []})

        judgements = judge_candidate_pairs(client, candidate_pairs, model=model)
        kept = postprocess_duplicates(records, [], candidate_pairs, judgements)
        return DatasetProcessorResult(
            kept_records=kept,
            artifacts={"judgements": judgements, "candidate_pairs": candidate_pairs},
        )

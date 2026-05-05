from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from pipeline.core.io import load_yaml
from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, DatasetProcessor, DatasetProcessorResult, ProcessorResult
from pipeline.ops.evaluate.rollout import parse_answer, ANSWER_PATTERN

log = logging.getLogger(__name__)


def _extract_prompt_parts(prompt_cfg: dict[str, Any]) -> tuple[str, str]:
    system_prompt = str(prompt_cfg.get("system_prompt", "")).strip()
    template = str(prompt_cfg.get("template", "")).strip()
    output_format = str(prompt_cfg.get("output_format", "")).strip()
    if output_format:
        template = f"{template}\n\n{output_format}".strip()
    return system_prompt, template


def _build_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "majority_answer": {"type": "string"},
            "majority_count": {"type": "integer"},
            "gt_quality": {"type": "string", "enum": ["clear", "unclear"]},
            "judgement": {
                "type": "string",
                "enum": ["equivalent", "not_equivalent", "not_applicable"],
            },
        },
        "required": ["majority_answer", "majority_count", "gt_quality", "judgement"],
    }


def _parse_judge_response(text: str) -> tuple[str | None, str | None, str | None, int | None]:
    """Parse LLM judge response.

    Returns (gt_quality, judgement, majority_answer, majority_count) or all-None on failure.
    """
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, None, None, None
    if not isinstance(obj, dict):
        return None, None, None, None

    gt_quality = str(obj.get("gt_quality", "")).strip().lower()
    judgement = str(obj.get("judgement", "")).strip().lower()
    majority_answer = str(obj.get("majority_answer", "")).strip()
    try:
        majority_count = int(obj.get("majority_count", 0))
    except (TypeError, ValueError):
        return None, None, None, None

    if gt_quality not in {"clear", "unclear"}:
        return None, None, None, None
    if judgement not in {"equivalent", "not_equivalent", "not_applicable"}:
        return None, None, None, None
    if gt_quality == "unclear" and judgement != "not_applicable":
        return None, None, None, None
    return gt_quality, judgement, majority_answer, majority_count


_BOXED_RE = re.compile(r"\\boxed\s*\{")
_LATEX_JUNK_RE = re.compile(r"^[\s\\\[\]$(){}*#`]+$")


def _extract_boxed(text: str) -> str:
    """Extract content from the last ``\\boxed{...}``, handling nested braces."""
    last_start = -1
    for m in _BOXED_RE.finditer(text):
        last_start = m.end()
    if last_start < 0:
        return ""
    depth, pos = 1, last_start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[last_start : pos - 1].strip() if depth == 0 else ""


def _is_junk(text: str) -> bool:
    return not text or bool(_LATEX_JUNK_RE.match(text))


def _extract_candidate_answers(
    samples: list[dict[str, Any]],
    answer_pattern: re.Pattern | None = None,
) -> list[str]:
    """Extract answers with priority: \\boxed > parse_answer > pre-extracted."""
    answers: list[str] = []
    for sample in samples:
        solution = sample.get("solution", "") or ""
        reasoning = sample.get("reasoning", "") or ""
        if reasoning == solution:
            reasoning = ""

        boxed = _extract_boxed(solution)
        if not boxed and reasoning:
            boxed = _extract_boxed(reasoning)
        if boxed and not _is_junk(boxed):
            answers.append(boxed)
            continue

        parsed = parse_answer(solution, answer_pattern)
        if not parsed and reasoning:
            parsed = parse_answer(reasoning, answer_pattern)
        if parsed and not _is_junk(parsed):
            answers.append(parsed)
            continue

        pre = str(sample.get("answer", "") or "").strip()
        if pre and not _is_junk(pre):
            answers.append(pre)
            continue

        answers.append(parsed or pre or "")
    return answers


def _format_candidate_answers(answers: list[str]) -> str:
    """Format candidate answers as a numbered list for the prompt."""
    lines: list[str] = []
    for i, ans in enumerate(answers, 1):
        display = ans if ans else "[empty/unparseable]"
        lines.append(f"  [{i}] {display}")
    return "\n".join(lines)


def _is_flagged_record(record: CanonicalRecord) -> bool:
    verification = dict(record.verification)
    return verification.get("majority_matches_gt") is False


def _has_rollout_samples(verification: dict[str, Any]) -> bool:
    samples = verification.get("samples", [])
    return bool(samples) and int(verification.get("num_samples", len(samples)) or 0) > 0


def _has_judge_result(verification: dict[str, Any]) -> bool:
    return bool(verification.get("llm_judge_gt_quality"))


def _majority_is_confident(verification: dict[str, Any]) -> bool:
    samples = list(verification.get("samples", []))
    num_samples = int(verification.get("num_samples", len(samples)) or len(samples) or 0)
    majority_count = int(verification.get("majority_count", 0) or 0)
    if num_samples <= 0:
        return False
    return majority_count > (num_samples / 2.0)


async def _call_judge_api(
    session,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    record_id: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    api_timeout: int,
) -> tuple[str, str, str, int]:
    import aiohttp

    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=api_timeout),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning(
                            "rid=%s attempt=%d judge status=%d body=%s",
                            record_id[:12],
                            attempt,
                            resp.status,
                            body[:400],
                        )
                        continue
                    data = await resp.json()
            except Exception as exc:
                log.warning("rid=%s attempt=%d judge error=%s", record_id[:12], attempt, exc)
                continue

        try:
            raw = str(data["choices"][0]["message"]["content"])
            finish_reason = str(data["choices"][0].get("finish_reason", ""))
        except (KeyError, IndexError, TypeError):
            log.warning("rid=%s attempt=%d judge bad response structure", record_id[:12], attempt)
            continue

        if finish_reason == "length":
            log.warning("rid=%s attempt=%d judge truncated", record_id[:12], attempt)
            continue

        gt_quality, judgement, maj_ans, maj_cnt = _parse_judge_response(raw)
        if gt_quality is not None and judgement is not None:
            return record_id, raw, "", attempt

        log.warning("rid=%s attempt=%d judge invalid response=%s", record_id[:12], attempt, raw[:200])

    return record_id, "", "api_error", max_retries


async def _judge_batch(
    rows: list[dict[str, Any]],
    *,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    prompt_template: str,
    concurrency: int,
    max_retries: int,
    api_timeout: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, tuple[str, str, int]]:
    import aiohttp

    url = f"{api_base.rstrip('/')}/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 32, ttl_dns_cache=300)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_judge",
            "schema": _build_json_schema(),
        },
    }

    results: dict[str, tuple[str, str, int]] = {}
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for row in rows:
            user_prompt = prompt_template.format(
                question=row["question"],
                gt_answer=row["gt_answer"],
                num_samples=row["num_samples"],
                candidate_answers=row["candidate_answers"],
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
            tasks.append(
                _call_judge_api(
                    session=session,
                    url=url,
                    payload=payload,
                    headers=headers,
                    record_id=row["record_id"],
                    semaphore=semaphore,
                    max_retries=max_retries,
                    api_timeout=api_timeout,
                )
            )

        done = 0
        total = len(tasks)
        t0 = time.monotonic()
        for coro in asyncio.as_completed(tasks):
            rid, raw, error, attempts = await coro
            results[rid] = (raw, error, attempts)
            done += 1
            if done % 100 == 0 or done == total:
                elapsed = time.monotonic() - t0
                speed = done / elapsed if elapsed > 0 else 0.0
                remaining = total - done
                eta = remaining / speed if speed > 0 else 0.0
                log.info(
                    "AnswerJudge progress %d/%d | remaining %d | %.1f req/s | ETA %.0fs",
                    done,
                    total,
                    remaining,
                    speed,
                    eta,
                )

    return results


@register_processor("answer_judge")
class AnswerJudgeProcessor(DatasetProcessor):
    name = "answer_judge"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.api_base = str(self.config.get("api_base", "http://localhost:8000/v1"))
        self.api_key = str(self.config.get("api_key", ""))
        self.model = str(self.config.get("model", "gpt-oss-120b"))
        self.concurrency = int(self.config.get("concurrency", 512))
        self.max_retries = int(self.config.get("max_retries", 3))
        self.api_timeout = int(self.config.get("api_timeout", 120))
        self.max_tokens = int(self.config.get("max_tokens", 16))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.top_p = float(self.config.get("top_p", 1.0))
        self.target_phase = str(self.config.get("target_phase", "posttrain"))

        prompt_path = str(self.config.get("prompt_path", "configs/prompts/answer_judge.yaml"))
        prompt_cfg = load_yaml(prompt_path)
        self.system_prompt, self.prompt_template = _extract_prompt_parts(prompt_cfg)

        answer_pattern_str = str(self.config.get("answer_pattern", ""))
        if answer_pattern_str:
            self.answer_pattern: re.Pattern | None = re.compile(
                answer_pattern_str, re.IGNORECASE | re.MULTILINE
            )
        else:
            self.answer_pattern = ANSWER_PATTERN

        log.info(
            "AnswerJudge: api=%s model=%s concurrency=%d target_phase=%s",
            self.api_base,
            self.model,
            self.concurrency,
            self.target_phase,
        )

    def _apply_api_error(self, record: CanonicalRecord, attempts: int) -> CanonicalRecord:
        verification = dict(record.verification)
        verification["llm_judge_gt_quality"] = "unclear"
        verification["llm_judge_equivalence"] = "not_applicable"
        verification["llm_judge_model"] = self.model
        verification["llm_judge_attempts"] = attempts
        verification["llm_judge_error"] = "api_error"

        updated = record.clone(
            verification=verification,
            training_phase="midtrain",
            filter_tag="answer_uncertain",
        )
        updated.add_trace(
            stage="evaluate",
            processor=self.name,
            status="routed",
            reason_code="answer_uncertain",
            details={"attempts": attempts, "error": "api_error"},
        )
        return updated

    def _apply_decision(
        self,
        record: CanonicalRecord,
        *,
        gt_quality: str,
        judgement: str,
        llm_majority_answer: str,
        llm_majority_count: int,
        raw_response: str,
        attempts: int,
    ) -> CanonicalRecord:
        verification = dict(record.verification)
        verification["llm_judge_gt_quality"] = gt_quality
        verification["llm_judge_equivalence"] = judgement
        verification["llm_judge_majority_answer"] = llm_majority_answer
        verification["llm_judge_majority_count"] = llm_majority_count
        verification["llm_judge_model"] = self.model
        verification["llm_judge_attempts"] = attempts
        verification["llm_judge_raw_response"] = raw_response

        if gt_quality == "clear":
            if judgement == "equivalent":
                verification["majority_matches_gt"] = True
                verification["majority_answer"] = llm_majority_answer
                verification["majority_count"] = llm_majority_count
                updated = record.clone(
                    verification=verification,
                    training_phase="posttrain",
                    filter_tag="equivalent_rescued",
                )
                updated.add_trace(
                    stage="evaluate",
                    processor=self.name,
                    status="kept",
                    reason_code="equivalent",
                    details={"attempts": attempts, "llm_majority_answer": llm_majority_answer},
                )
                return updated

            updated = record.clone(
                verification=verification,
                training_phase="midtrain",
                filter_tag="answer_mismatch",
            )
            updated.add_trace(
                stage="evaluate",
                processor=self.name,
                status="routed",
                reason_code="answer_mismatch",
                details={"attempts": attempts, "llm_majority_answer": llm_majority_answer},
            )
            return updated

        if llm_majority_count > (int(verification.get("num_samples", 0) or 0) / 2.0):
            updated = record.clone(
                verification=verification,
                training_phase="midtrain",
                filter_tag="gt_unclear",
            )
            updated.add_trace(
                stage="evaluate",
                processor=self.name,
                status="routed",
                reason_code="gt_unclear",
                details={"attempts": attempts, "llm_majority_answer": llm_majority_answer},
            )
            return updated

        updated = record.clone(
            verification=verification,
            training_phase="midtrain",
            filter_tag="answer_uncertain",
        )
        updated.add_trace(
            stage="evaluate",
            processor=self.name,
            status="routed",
            reason_code="answer_uncertain",
            details={"attempts": attempts, "llm_majority_answer": llm_majority_answer},
        )
        return updated

    def process(
        self,
        records: list[CanonicalRecord],
        *,
        pipeline_artifacts: dict[str, Any] | None = None,
    ) -> DatasetProcessorResult:
        rows: list[dict[str, Any]] = []
        row_idx: dict[str, int] = {}
        updated_records = list(records)
        processor_results: list[ProcessorResult] = []

        skipped_done = 0
        skipped_not_flagged = 0
        no_rollout = 0

        for idx, record in enumerate(records):
            if self.target_phase and record.training_phase != self.target_phase:
                continue

            verification = dict(record.verification)
            if not _is_flagged_record(record):
                skipped_not_flagged += 1
                continue

            if _has_judge_result(verification):
                skipped_done += 1
                continue

            if not _has_rollout_samples(verification):
                no_rollout += 1
                updated = self._apply_api_error(record, attempts=0)
                updated_records[idx] = updated
                processor_results.append(
                    ProcessorResult(
                        keep=True,
                        record=updated,
                        stage="evaluate",
                        processor=self.name,
                        reason_code="answer_uncertain",
                    )
                )
                continue

            samples = list(verification.get("samples", []))
            candidate_answers = _extract_candidate_answers(samples, self.answer_pattern)
            num_samples = len(candidate_answers)
            candidate_answers_text = _format_candidate_answers(candidate_answers)

            rows.append(
                {
                    "record_id": record.record_id,
                    "question": record.question,
                    "gt_answer": str(verification.get("raw_dataset_answer", record.raw_dataset_answer or "")),
                    "num_samples": num_samples,
                    "candidate_answers": candidate_answers_text,
                }
            )
            row_idx[record.record_id] = idx

        log.info(
            "AnswerJudge: total=%d matched=%d skipped_not_flagged=%d skipped_done=%d no_rollout=%d",
            len(records),
            len(rows),
            skipped_not_flagged,
            skipped_done,
            no_rollout,
        )

        if rows:
            api_results = asyncio.run(
                _judge_batch(
                    rows,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    model=self.model,
                    system_prompt=self.system_prompt,
                    prompt_template=self.prompt_template,
                    concurrency=self.concurrency,
                    max_retries=self.max_retries,
                    api_timeout=self.api_timeout,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            )

            for rid, (raw_response, error, attempts) in api_results.items():
                idx = row_idx[rid]
                record = updated_records[idx]
                if error:
                    updated = self._apply_api_error(record, attempts)
                    reason_code = "answer_uncertain"
                else:
                    gt_quality, judgement, llm_maj_ans, llm_maj_cnt = _parse_judge_response(raw_response)
                    if gt_quality is None or judgement is None:
                        updated = self._apply_api_error(record, attempts)
                        reason_code = "answer_uncertain"
                    else:
                        updated = self._apply_decision(
                            record,
                            gt_quality=gt_quality,
                            judgement=judgement,
                            llm_majority_answer=llm_maj_ans or "",
                            llm_majority_count=llm_maj_cnt or 0,
                            raw_response=raw_response,
                            attempts=attempts,
                        )
                        reason_code = updated.filter_tag or "equivalent"
                updated_records[idx] = updated
                processor_results.append(
                    ProcessorResult(
                        keep=True,
                        record=updated,
                        stage="evaluate",
                        processor=self.name,
                        reason_code=reason_code,
                    )
                )

        return DatasetProcessorResult(
            kept_records=updated_records,
            processor_results=processor_results,
        )

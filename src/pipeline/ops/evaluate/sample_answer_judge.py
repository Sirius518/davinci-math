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
from pipeline.ops.evaluate.rollout import ANSWER_PATTERN, parse_answer

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
            "judgement": {
                "type": "string",
                "enum": ["equivalent", "not_equivalent"],
            }
        },
        "required": ["judgement"],
    }


def _parse_judge_response(text: str) -> str | None:
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    judgement = str(obj.get("judgement", "")).strip().lower()
    if judgement not in {"equivalent", "not_equivalent"}:
        return None
    return judgement


_BOXED_RE = re.compile(r"\\boxed\s*\{")
_LATEX_JUNK_RE = re.compile(r"^[\s\\\[\]$(){}*#`]+$")


def _extract_boxed(text: str) -> str:
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


def _extract_sample_answer(
    sample: dict[str, Any],
    answer_pattern: re.Pattern | None = None,
) -> str:
    solution = sample.get("solution", "") or ""
    reasoning = sample.get("reasoning", "") or ""
    if reasoning == solution:
        reasoning = ""

    boxed = _extract_boxed(solution)
    if not boxed and reasoning:
        boxed = _extract_boxed(reasoning)
    if boxed and not _is_junk(boxed):
        return boxed

    parsed = parse_answer(solution, answer_pattern)
    if not parsed and reasoning:
        parsed = parse_answer(reasoning, answer_pattern)
    if parsed and not _is_junk(parsed):
        return parsed

    pre = str(sample.get("answer", "") or "").strip()
    if pre and not _is_junk(pre):
        return pre

    return parsed or pre or ""


def _has_rollout_samples(verification: dict[str, Any]) -> bool:
    samples = verification.get("samples", [])
    return bool(samples) and int(verification.get("num_samples", len(samples)) or 0) > 0


async def _call_judge_api(
    session,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    request_id: str,
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
                            "sid=%s attempt=%d judge status=%d body=%s",
                            request_id[:20],
                            attempt,
                            resp.status,
                            body[:400],
                        )
                        continue
                    data = await resp.json()
            except Exception as exc:
                log.warning("sid=%s attempt=%d judge error=%s", request_id[:20], attempt, exc)
                continue

        try:
            raw = str(data["choices"][0]["message"]["content"])
            finish_reason = str(data["choices"][0].get("finish_reason", ""))
        except (KeyError, IndexError, TypeError):
            log.warning("sid=%s attempt=%d judge bad response structure", request_id[:20], attempt)
            continue

        if finish_reason == "length":
            log.warning("sid=%s attempt=%d judge truncated", request_id[:20], attempt)
            continue

        judgement = _parse_judge_response(raw)
        if judgement is not None:
            return request_id, raw, "", attempt

        log.warning("sid=%s attempt=%d judge invalid response=%s", request_id[:20], attempt, raw[:200])

    return request_id, "", "api_error", max_retries


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
            "name": "sample_answer_judge",
            "schema": _build_json_schema(),
        },
    }

    results: dict[str, tuple[str, str, int]] = {}
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for row in rows:
            user_prompt = prompt_template.format(
                reference_answer=row["reference_answer"],
                candidate_answer=row["candidate_answer"],
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
                    request_id=row["request_id"],
                    semaphore=semaphore,
                    max_retries=max_retries,
                    api_timeout=api_timeout,
                )
            )

        done = 0
        total = len(tasks)
        t0 = time.monotonic()
        for coro in asyncio.as_completed(tasks):
            request_id, raw, error, attempts = await coro
            results[request_id] = (raw, error, attempts)
            done += 1
            if done % 1000 == 0 or done == total:
                elapsed = time.monotonic() - t0
                speed = done / elapsed if elapsed > 0 else 0.0
                remaining = total - done
                eta = remaining / speed if speed > 0 else 0.0
                log.info(
                    "SampleAnswerJudge progress %d/%d | remaining %d | %.1f req/s | ETA %.0fs",
                    done,
                    total,
                    remaining,
                    speed,
                    eta,
                )

    return results


@register_processor("sample_answer_judge")
class SampleAnswerJudgeProcessor(DatasetProcessor):
    name = "sample_answer_judge"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.api_base = str(self.config.get("api_base", "http://localhost:8000/v1"))
        self.api_key = str(self.config.get("api_key", ""))
        self.model = str(self.config.get("model", "gpt-oss-20b"))
        self.concurrency = int(self.config.get("concurrency", 2048))
        self.max_retries = int(self.config.get("max_retries", 3))
        self.api_timeout = int(self.config.get("api_timeout", 30))
        self.max_tokens = int(self.config.get("max_tokens", 16))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.top_p = float(self.config.get("top_p", 1.0))
        self.target_phase = str(self.config.get("target_phase", "posttrain"))
        self.min_majority_count = int(self.config.get("min_majority_count", 2))

        prompt_path = str(
            self.config.get("prompt_path", "configs/prompts/sample_answer_judge.yaml")
        )
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
            "SampleAnswerJudge: api=%s model=%s concurrency=%d target_phase=%s "
            "min_majority_count=%d",
            self.api_base,
            self.model,
            self.concurrency,
            self.target_phase,
            self.min_majority_count,
        )

    def _refresh_verification_summary(self, verification: dict[str, Any]) -> None:
        samples = list(verification.get("samples", []))
        num_samples = int(verification.get("num_samples", len(samples)) or len(samples) or 0)
        judged_values = [sample.get("judge_correct") for sample in samples if isinstance(sample.get("judge_correct"), bool)]
        correct_count = sum(1 for value in judged_values if value)
        verification["pass_ratio"] = (correct_count / num_samples) if num_samples > 0 else 0.0

    def _maybe_demote_low_confidence(
        self,
        record: CanonicalRecord,
        verification: dict[str, Any],
    ) -> tuple[CanonicalRecord, bool]:
        majority_count = int(verification.get("majority_count", 0) or 0)
        if record.training_phase != "posttrain" or majority_count >= self.min_majority_count:
            return record.clone(verification=verification), False

        updated = record.clone(
            verification=verification,
            training_phase="midtrain",
            filter_tag="low_confidence",
        )
        updated.add_trace(
            stage="evaluate",
            processor=self.name,
            status="kept",
            reason_code="low_confidence",
            details={
                "majority_count": majority_count,
                "min_majority_count": self.min_majority_count,
                "pass_ratio": verification.get("pass_ratio", 0.0),
            },
        )
        return updated, True

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

        skipped_phase = 0
        skipped_no_rollout = 0
        skipped_done = 0

        for idx, record in enumerate(records):
            if self.target_phase and record.training_phase != self.target_phase:
                skipped_phase += 1
                continue

            verification = dict(record.verification)
            if not _has_rollout_samples(verification):
                skipped_no_rollout += 1
                continue

            samples = list(verification.get("samples", []))
            reference_answer = (
                str(record.confirmed_answer or "").strip()
                or str(verification.get("confirmed_answer", "") or "").strip()
                or str(verification.get("majority_answer", "") or "").strip()
                or str(record.raw_dataset_answer or "").strip()
            )
            if not reference_answer:
                skipped_no_rollout += 1
                continue

            missing_judgements = 0
            for sample_idx, sample in enumerate(samples):
                if isinstance(sample.get("judge_correct"), bool):
                    continue
                candidate_answer = _extract_sample_answer(sample, self.answer_pattern).strip()
                request_id = f"{record.record_id}:{sample_idx}"
                rows.append(
                    {
                        "request_id": request_id,
                        "reference_answer": reference_answer,
                        "candidate_answer": candidate_answer,
                    }
                )
                row_idx[request_id] = idx
                missing_judgements += 1

            if missing_judgements == 0:
                self._refresh_verification_summary(verification)
                updated, demoted = self._maybe_demote_low_confidence(record, verification)
                updated_records[idx] = updated
                processor_results.append(
                    ProcessorResult(
                        keep=True,
                        record=updated,
                        stage="evaluate",
                        processor=self.name,
                        reason_code="low_confidence" if demoted else "already_judged",
                    )
                )
                skipped_done += 1

        n_records_to_judge = len(records) - skipped_phase - skipped_done - skipped_no_rollout
        log.info(
            "SampleAnswerJudge: total_records=%d posttrain=%d records_to_judge=%d "
            "samples_to_judge=%d skipped_done=%d skipped_no_rollout=%d",
            len(records),
            len(records) - skipped_phase,
            n_records_to_judge,
            len(rows),
            skipped_done,
            skipped_no_rollout,
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

            touched_records: set[int] = set()

            for request_id, (raw_response, error, _attempts) in api_results.items():
                idx = row_idx[request_id]
                touched_records.add(idx)
                record = updated_records[idx]
                verification = dict(record.verification)
                samples = list(verification.get("samples", []))

                _, sample_idx_str = request_id.rsplit(":", 1)
                sample_idx = int(sample_idx_str)
                if sample_idx >= len(samples):
                    continue

                if error:
                    continue

                judgement = _parse_judge_response(raw_response)
                if judgement is None:
                    continue

                samples[sample_idx]["judge_correct"] = judgement == "equivalent"
                verification["samples"] = samples
                updated_records[idx] = record.clone(verification=verification)

            for idx in touched_records:
                record = updated_records[idx]
                verification = dict(record.verification)
                self._refresh_verification_summary(verification)
                judged = record.clone(verification=verification)
                judged.add_trace(
                    stage="evaluate",
                    processor=self.name,
                    status="kept",
                    reason_code="sample_answer_judged",
                    details={
                        "pass_ratio": verification.get("pass_ratio", 0.0),
                    },
                )
                updated, demoted = self._maybe_demote_low_confidence(judged, verification)
                updated_records[idx] = updated
                processor_results.append(
                    ProcessorResult(
                        keep=True,
                        record=updated,
                        stage="evaluate",
                        processor=self.name,
                        reason_code="low_confidence" if demoted else "sample_answer_judged",
                    )
                )

        return DatasetProcessorResult(
            kept_records=updated_records,
            processor_results=processor_results,
        )

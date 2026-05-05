"""Unit tests for rollout correctness processor."""
from __future__ import annotations

import asyncio
import json
import re
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.core.io import load_yaml
from pipeline.core.schema import CanonicalRecord
from pipeline.ops.evaluate.rollout import (
    ANSWER_PATTERN,
    RolloutCorrectnessProcessor,
    _truncate_question,
    build_rollout_prompt,
    parse_answer,
    verify_equivalent,
)


PROMPT_YAML_PATH = str(ROOT / "configs" / "prompts" / "rollout_correctness.yaml")


# ========================================================================
# 1. Prompt YAML tests
# ========================================================================

class TestPromptYAML(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_yaml(PROMPT_YAML_PATH)

    def test_yaml_loads(self) -> None:
        self.assertIn("template", self.cfg)
        self.assertIn("answer_pattern", self.cfg)
        self.assertEqual(self.cfg["name"], "rollout_correctness")

    def test_template_has_problem_placeholder(self) -> None:
        self.assertIn("{problem}", self.cfg["template"])

    def test_template_has_boxed(self) -> None:
        self.assertIn("\\boxed{}", self.cfg["template"])

    def test_build_rollout_prompt_template_mode(self) -> None:
        template, system_prompt = build_rollout_prompt(self.cfg)
        self.assertIn("{problem}", template)
        self.assertIn("\\boxed{}", template)

    def test_template_rendering_preserves_boxed(self) -> None:
        template, _ = build_rollout_prompt(self.cfg)
        question = "Find all real solutions to $x^2 - 5x + 6 = 0$."
        rendered = template.replace("{problem}", question)
        self.assertIn(question, rendered)
        self.assertIn("\\boxed{}", rendered)

    def test_rendered_prompt_format(self) -> None:
        """Verify the exact format: question first, then instruction."""
        template, _ = build_rollout_prompt(self.cfg)
        rendered = template.replace("{problem}", "What is 2+3?")
        self.assertIn("What is 2+3?", rendered)
        self.assertIn("\\boxed{}", rendered)
        idx_q = rendered.index("What is 2+3?")
        idx_b = rendered.index("\\boxed{}")
        self.assertLess(idx_q, idx_b)

    def test_legacy_prefix_mode_fallback(self) -> None:
        """When no 'template' key, fall back to instruction+suffix mode."""
        legacy_cfg = {
            "instruction": "Solve step by step.\nFinal Answer: <answer>",
            "suffix": "\nProblem:\n",
        }
        template, sys_prompt = build_rollout_prompt(legacy_cfg)
        self.assertIn("{problem}", template)
        self.assertEqual(sys_prompt, "")


# ========================================================================
# 2. parse_answer tests
# ========================================================================

class TestParseAnswer(unittest.TestCase):
    def setUp(self) -> None:
        cfg = load_yaml(PROMPT_YAML_PATH)
        self.pattern = re.compile(
            cfg["answer_pattern"], re.IGNORECASE | re.MULTILINE
        )

    def test_simple_boxed(self) -> None:
        self.assertEqual(parse_answer("So \\boxed{42}", self.pattern), "42")

    def test_boxed_with_fraction(self) -> None:
        self.assertEqual(
            parse_answer("Therefore \\boxed{\\frac{1}{2}}", self.pattern),
            "\\frac{1}{2}",
        )

    def test_boxed_with_exponent(self) -> None:
        self.assertEqual(
            parse_answer("The answer is \\boxed{x^{2} + 1}", self.pattern),
            "x^{2} + 1",
        )

    def test_multiple_boxed_takes_last(self) -> None:
        text = "First we compute $a = \\boxed{5}$.\nAfter combining: $\\boxed{42}$"
        self.assertEqual(parse_answer(text, self.pattern), "42")

    def test_boxed_negative_number(self) -> None:
        self.assertEqual(parse_answer("\\boxed{-7}", self.pattern), "-7")

    def test_boxed_expression(self) -> None:
        self.assertEqual(
            parse_answer("\\boxed{2\\sqrt{3}}", self.pattern), "2\\sqrt{3}"
        )

    def test_no_boxed_fallback_to_last_line(self) -> None:
        text = "Step 1: ...\nStep 2: ...\n42"
        self.assertEqual(parse_answer(text, self.pattern), "42")

    def test_empty_input(self) -> None:
        self.assertEqual(parse_answer("", self.pattern), "")

    def test_legacy_final_answer_pattern(self) -> None:
        """Default ANSWER_PATTERN still works for backward compat."""
        text = "Reasoning...\nFinal Answer: 42"
        self.assertEqual(parse_answer(text, ANSWER_PATTERN), "42")


# ========================================================================
# 3. verify_equivalent tests
# ========================================================================

class TestVerifyEquivalent(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertTrue(verify_equivalent("42", "42"))

    def test_whitespace_insensitive(self) -> None:
        self.assertTrue(verify_equivalent(" 42 ", "42"))

    def test_case_insensitive(self) -> None:
        self.assertTrue(verify_equivalent("ABC", "abc"))

    def test_different_values(self) -> None:
        self.assertFalse(verify_equivalent("42", "43"))


# ========================================================================
# 4. Truncation tests
# ========================================================================

class TestTruncation(unittest.TestCase):
    def test_no_truncation_when_under_limit(self) -> None:
        self.assertEqual(_truncate_question("hello", 100), "hello")

    def test_truncation_when_over_limit(self) -> None:
        long_text = "a" * 200
        result = _truncate_question(long_text, 100)
        self.assertIn("[TRUNCATED]", result)
        self.assertLess(len(result), 200)

    def test_zero_limit_means_disabled(self) -> None:
        result = _truncate_question("anything", 0)
        self.assertEqual(result, "anything")


# ========================================================================
# 5. Processor init tests
# ========================================================================

class TestProcessorInit(unittest.TestCase):
    def _make_processor(self) -> RolloutCorrectnessProcessor:
        return RolloutCorrectnessProcessor(config={
            "api_base": "http://localhost:8000/v1",
            "model": "test-model",
            "num_samples": 4,
            "concurrency": 32,
            "max_retries": 2,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 2048,
            "reasoning_effort": "medium",
            "prompt_path": PROMPT_YAML_PATH,
        })

    def test_config_values(self) -> None:
        proc = self._make_processor()
        self.assertEqual(proc.model, "test-model")
        self.assertEqual(proc.num_samples, 4)
        self.assertEqual(proc.concurrency, 32)
        self.assertAlmostEqual(proc.temperature, 0.8)
        self.assertEqual(proc.reasoning_effort, "medium")

    def test_answer_pattern_from_yaml(self) -> None:
        proc = self._make_processor()
        self.assertIsNotNone(proc.answer_pattern.search("\\boxed{42}"))

    def test_template_loaded(self) -> None:
        proc = self._make_processor()
        self.assertIn("{problem}", proc.prompt_template)
        self.assertIn("\\boxed{}", proc.prompt_template)


# ========================================================================
# 6. Mock helpers for end-to-end tests
# ========================================================================

def _make_api_response(
    answers: list[str],
    reasoning_contents: list[str] | None = None,
    finish_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Build a fake OpenAI chat completion response with n choices."""
    if finish_reasons is None:
        finish_reasons = ["stop"] * len(answers)
    if reasoning_contents is None:
        reasoning_contents = [""] * len(answers)
    choices = []
    for i, (ans, rc, fr) in enumerate(zip(answers, reasoning_contents, finish_reasons)):
        msg: dict[str, Any] = {"role": "assistant", "content": ans}
        if rc:
            msg["reasoning_content"] = rc
        choices.append({
            "index": i,
            "message": msg,
            "finish_reason": fr,
        })
    return {"choices": choices}


class _AsyncCtx:
    def __init__(self, val: Any) -> None:
        self._val = val

    async def __aenter__(self) -> Any:
        return self._val

    async def __aexit__(self, *args: Any) -> None:
        pass


# ========================================================================
# 7. End-to-end processor tests
# ========================================================================

class TestProcessorEndToEnd(unittest.TestCase):
    """Full processor test with a mock aiohttp session."""

    def _make_processor(self, **overrides: Any) -> RolloutCorrectnessProcessor:
        cfg: dict[str, Any] = {
            "api_base": "http://fake:8000/v1",
            "model": "test-model",
            "num_samples": 8,
            "concurrency": 4,
            "max_retries": 1,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": -1,
            "max_tokens": 4096,
            "split_n": False,
            "prompt_path": PROMPT_YAML_PATH,
        }
        cfg.update(overrides)
        return RolloutCorrectnessProcessor(config=cfg)

    def _run_with_mock_responses(
        self,
        proc: RolloutCorrectnessProcessor,
        records: list[CanonicalRecord],
        responses: dict[str, dict[str, Any]],
    ) -> Any:
        """Run the processor, mocking aiohttp to return preset responses.

        ``responses`` maps record_id -> API response dict.
        """
        def fake_post(url: str, json: dict[str, Any], timeout: Any = None) -> _AsyncCtx:
            question = json["messages"][-1]["content"]
            matched_rid = None
            for rid in responses:
                for rec in records:
                    if rec.record_id == rid and rec.question in question:
                        matched_rid = rid
                        break
                if matched_rid:
                    break

            if matched_rid is None:
                matched_rid = list(responses.keys())[0]

            full_resp = responses.get(matched_rid, {"choices": []})
            # Respect the n parameter: return only the first n choices
            req_n = json.get("n", len(full_resp.get("choices", [])))
            resp_data = dict(full_resp)
            resp_data["choices"] = full_resp.get("choices", [])[:req_n]
            mock_resp = MagicMock()
            mock_resp.status = 200

            async def _json(_d: dict = resp_data) -> dict:
                return _d

            async def _text(_d: dict = resp_data) -> str:
                return json_module.dumps(_d)

            mock_resp.json = _json
            mock_resp.text = _text
            return _AsyncCtx(mock_resp)

        import json as json_module

        class FakeSession:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.post = fake_post

            async def __aenter__(self) -> "FakeSession":
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

        class FakeConnector:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        with patch("pipeline.ops.evaluate.rollout.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = FakeSession
            mock_aiohttp.TCPConnector = FakeConnector
            mock_aiohttp.ClientTimeout = lambda total: total
            result = proc.process(records)

        return result

    def test_majority_correct_verification_structure(self) -> None:
        proc = self._make_processor()
        rec = CanonicalRecord(
            record_id="rec-001",
            question="Find all real solutions to $x^2 - 5x + 6 = 0$.",
            raw_dataset_answer="2, 3",
            training_phase="posttrain",
        )
        correct_answers = ["The roots are \\boxed{2, 3}"] * 6
        wrong_answers = ["\\boxed{1, 6}"] * 2
        all_answers = correct_answers + wrong_answers

        resp = _make_api_response(all_answers)
        result = self._run_with_mock_responses(proc, [rec], {"rec-001": resp})

        self.assertEqual(len(result.kept_records), 1)
        v = result.kept_records[0].verification
        self.assertIn("model", v)
        self.assertIn("num_samples", v)
        self.assertIn("majority_answer", v)
        self.assertIn("majority_count", v)
        self.assertIn("majority_matches_gt", v)
        self.assertIn("pass_ratio", v)
        self.assertIn("samples", v)
        self.assertEqual(v["num_samples"], 8)
        self.assertIsInstance(v["samples"], list)
        self.assertEqual(len(v["samples"]), 8)
        sample = v["samples"][0]
        self.assertIn("index", sample)
        self.assertIn("answer", sample)
        self.assertIn("solution", sample)
        self.assertIn("reasoning", sample)
        self.assertIn("success", sample)
        self.assertIn("stop_reason", sample)

    def test_api_payload_format(self) -> None:
        proc = self._make_processor(reasoning_effort="medium")
        rec = CanonicalRecord(
            record_id="rec-payload",
            question="What is 1+1?",
            raw_dataset_answer="2",
        )
        captured: list[dict[str, Any]] = []
        original_post = None

        def capturing_post(url: str, json: dict[str, Any], timeout: Any = None) -> _AsyncCtx:
            captured.append(json)
            resp = _make_api_response(["\\boxed{2}"] * 8)
            mock_resp = MagicMock()
            mock_resp.status = 200

            async def _json() -> dict:
                return resp

            async def _text() -> str:
                return "{}"

            mock_resp.json = _json
            mock_resp.text = _text
            return _AsyncCtx(mock_resp)

        class FakeSession:
            def __init__(self, *a: Any, **kw: Any) -> None:
                self.post = capturing_post
            async def __aenter__(self) -> "FakeSession":
                return self
            async def __aexit__(self, *a: Any) -> None:
                pass

        class FakeConnector:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

        with patch("pipeline.ops.evaluate.rollout.aiohttp") as m:
            m.ClientSession = FakeSession
            m.TCPConnector = FakeConnector
            m.ClientTimeout = lambda total: total
            proc.process([rec])

        self.assertTrue(len(captured) >= 1)
        payload = captured[0]
        self.assertEqual(payload["n"], 8)
        self.assertAlmostEqual(payload["temperature"], 0.7)
        self.assertAlmostEqual(payload["top_p"], 0.95)
        self.assertEqual(payload["top_k"], -1)
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["reasoning_effort"], "medium")
        self.assertIn("messages", payload)
        user_msg = payload["messages"][-1]["content"]
        self.assertIn("What is 1+1?", user_msg)
        self.assertIn("\\boxed{}", user_msg)

    def test_majority_wrong_flagged(self) -> None:
        proc = self._make_processor()
        rec = CanonicalRecord(
            record_id="rec-wrong",
            question="What is 2+2?",
            raw_dataset_answer="4",
        )
        resp = _make_api_response(["\\boxed{5}"] * 8)
        result = self._run_with_mock_responses(proc, [rec], {"rec-wrong": resp})

        v = result.kept_records[0].verification
        self.assertEqual(v["majority_answer"], "5")
        self.assertFalse(v["majority_matches_gt"])
        trace = result.kept_records[0].trace
        self.assertTrue(any(t.reason_code == "gt_suspect" for t in trace))

    def test_truncated_sample_marked(self) -> None:
        proc = self._make_processor()
        rec = CanonicalRecord(
            record_id="rec-trunc",
            question="Hard problem",
            raw_dataset_answer="42",
        )
        answers = ["\\boxed{42}"] * 7 + ["partial reasoning without boxed"]
        finish = ["stop"] * 7 + ["length"]
        resp = _make_api_response(answers, finish_reasons=finish)
        result = self._run_with_mock_responses(proc, [rec], {"rec-trunc": resp})

        v = result.kept_records[0].verification
        last_sample = v["samples"][-1]
        self.assertEqual(last_sample["stop_reason"], "truncated")
        self.assertTrue(last_sample["success"])

    def test_target_phase_filters_records(self) -> None:
        proc = self._make_processor(target_phase="posttrain")
        rec_post = CanonicalRecord(
            record_id="rec-post",
            question="What is 1+1?",
            raw_dataset_answer="2",
            training_phase="posttrain",
        )
        rec_mid = CanonicalRecord(
            record_id="rec-mid",
            question="What is 2+2?",
            raw_dataset_answer="4",
            training_phase="midtrain",
        )
        resp = _make_api_response(["\\boxed{2}"] * 8)
        result = self._run_with_mock_responses(
            proc, [rec_post, rec_mid], {"rec-post": resp}
        )

        self.assertEqual(len(result.kept_records), 2)
        post_rec = next(r for r in result.kept_records if r.record_id == "rec-post")
        mid_rec = next(r for r in result.kept_records if r.record_id == "rec-mid")
        self.assertIn("model", post_rec.verification)
        self.assertEqual(mid_rec.verification, {})

    def test_reasoning_content_separated(self) -> None:
        """When the server returns reasoning_content, reasoning and solution differ."""
        proc = self._make_processor(separate_reasoning=True)
        rec = CanonicalRecord(
            record_id="rec-sep",
            question="What is 1+1?",
            raw_dataset_answer="2",
        )
        solutions = ["The answer is \\boxed{2}"] * 8
        reasonings = ["Let me think step by step... 1+1=2"] * 8
        resp = _make_api_response(solutions, reasoning_contents=reasonings)
        result = self._run_with_mock_responses(proc, [rec], {"rec-sep": resp})

        v = result.kept_records[0].verification
        sample = v["samples"][0]
        self.assertEqual(sample["solution"], "The answer is \\boxed{2}")
        self.assertEqual(sample["reasoning"], "Let me think step by step... 1+1=2")
        self.assertNotEqual(sample["reasoning"], sample["solution"])

    def test_no_reasoning_content_fallback(self) -> None:
        """When server doesn't return reasoning_content, reasoning == solution."""
        proc = self._make_processor(separate_reasoning=False)
        rec = CanonicalRecord(
            record_id="rec-nosep",
            question="What is 1+1?",
            raw_dataset_answer="2",
        )
        solutions = ["Step by step... \\boxed{2}"] * 8
        resp = _make_api_response(solutions)
        result = self._run_with_mock_responses(proc, [rec], {"rec-nosep": resp})

        v = result.kept_records[0].verification
        sample = v["samples"][0]
        self.assertEqual(sample["reasoning"], sample["solution"])


# ========================================================================
# 8. split_n mode tests
# ========================================================================

class TestSplitNMode(unittest.TestCase):
    """Tests for split_n=True: n=8 split into 8 individual n=1 requests."""

    def _make_processor(self, **overrides: Any) -> RolloutCorrectnessProcessor:
        cfg: dict[str, Any] = {
            "api_base": "http://fake:8000/v1",
            "model": "test-model",
            "num_samples": 8,
            "concurrency": 16,
            "max_retries": 1,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": -1,
            "max_tokens": 4096,
            "split_n": True,
            "prompt_path": PROMPT_YAML_PATH,
        }
        cfg.update(overrides)
        return RolloutCorrectnessProcessor(config=cfg)

    def test_split_n_sends_individual_requests(self) -> None:
        """split_n=True should send 8 n=1 requests instead of 1 n=8."""
        proc = self._make_processor()
        rec = CanonicalRecord(
            record_id="rec-split",
            question="What is 2+2?",
            raw_dataset_answer="4",
        )
        captured: list[dict[str, Any]] = []

        def capturing_post(url: str, json: dict[str, Any], timeout: Any = None) -> _AsyncCtx:
            captured.append(dict(json))
            resp = _make_api_response(["\\boxed{4}"])
            mock_resp = MagicMock()
            mock_resp.status = 200

            async def _json() -> dict:
                return resp

            async def _text() -> str:
                return "{}"

            mock_resp.json = _json
            mock_resp.text = _text
            return _AsyncCtx(mock_resp)

        class FakeSession:
            def __init__(self, *a: Any, **kw: Any) -> None:
                self.post = capturing_post
            async def __aenter__(self) -> "FakeSession":
                return self
            async def __aexit__(self, *a: Any) -> None:
                pass

        class FakeConnector:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

        with patch("pipeline.ops.evaluate.rollout.aiohttp") as m:
            m.ClientSession = FakeSession
            m.TCPConnector = FakeConnector
            m.ClientTimeout = lambda total: total
            result = proc.process([rec])

        # Should have sent 8 individual requests
        self.assertEqual(len(captured), 8)
        for payload in captured:
            self.assertEqual(payload["n"], 1)

        # Result should still have 8 samples aggregated
        v = result.kept_records[0].verification
        self.assertEqual(v["num_samples"], 8)
        self.assertEqual(len(v["samples"]), 8)
        self.assertTrue(v["majority_matches_gt"])
        self.assertEqual(v["majority_answer"], "4")

    def test_split_n_false_sends_single_request(self) -> None:
        """split_n=False should send 1 n=8 request (original behavior)."""
        proc = self._make_processor(split_n=False)
        rec = CanonicalRecord(
            record_id="rec-nosplit",
            question="What is 3+3?",
            raw_dataset_answer="6",
        )
        captured: list[dict[str, Any]] = []

        def capturing_post(url: str, json: dict[str, Any], timeout: Any = None) -> _AsyncCtx:
            captured.append(dict(json))
            resp = _make_api_response(["\\boxed{6}"] * 8)
            mock_resp = MagicMock()
            mock_resp.status = 200

            async def _json() -> dict:
                return resp

            async def _text() -> str:
                return "{}"

            mock_resp.json = _json
            mock_resp.text = _text
            return _AsyncCtx(mock_resp)

        class FakeSession:
            def __init__(self, *a: Any, **kw: Any) -> None:
                self.post = capturing_post
            async def __aenter__(self) -> "FakeSession":
                return self
            async def __aexit__(self, *a: Any) -> None:
                pass

        class FakeConnector:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

        with patch("pipeline.ops.evaluate.rollout.aiohttp") as m:
            m.ClientSession = FakeSession
            m.TCPConnector = FakeConnector
            m.ClientTimeout = lambda total: total
            proc.process([rec])

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["n"], 8)

    def test_split_n_partial_failure(self) -> None:
        """When some split requests fail, remaining samples are still used."""
        proc = self._make_processor(num_samples=4)
        rec = CanonicalRecord(
            record_id="rec-partial",
            question="What is 5+5?",
            raw_dataset_answer="10",
        )
        call_count = 0

        def flaky_post(url: str, json: dict[str, Any], timeout: Any = None) -> _AsyncCtx:
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            # Fail the first request, succeed the rest
            if call_count == 1:
                mock_resp.status = 500
                async def _text() -> str:
                    return "Internal Server Error"
                mock_resp.text = _text
            else:
                resp = _make_api_response(["\\boxed{10}"])
                mock_resp.status = 200
                async def _json(_r: dict = resp) -> dict:
                    return _r
                async def _text2(_r: dict = resp) -> str:
                    return "{}"
                mock_resp.json = _json
                mock_resp.text = _text2
            return _AsyncCtx(mock_resp)

        class FakeSession:
            def __init__(self, *a: Any, **kw: Any) -> None:
                self.post = flaky_post
            async def __aenter__(self) -> "FakeSession":
                return self
            async def __aexit__(self, *a: Any) -> None:
                pass

        class FakeConnector:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

        with patch("pipeline.ops.evaluate.rollout.aiohttp") as m:
            m.ClientSession = FakeSession
            m.TCPConnector = FakeConnector
            m.ClientTimeout = lambda total: total
            result = proc.process([rec])

        v = result.kept_records[0].verification
        # Should have 3 successful samples (4 total minus 1 failed)
        self.assertEqual(v["num_samples"], 3)
        self.assertEqual(len(v["samples"]), 3)
        self.assertTrue(v["majority_matches_gt"])

    def test_split_n_multiple_records(self) -> None:
        """split_n correctly aggregates samples across multiple records."""
        proc = self._make_processor(num_samples=4)
        rec1 = CanonicalRecord(
            record_id="rec-m1",
            question="What is 1+1?",
            raw_dataset_answer="2",
        )
        rec2 = CanonicalRecord(
            record_id="rec-m2",
            question="What is 3+3?",
            raw_dataset_answer="6",
        )

        def smart_post(url: str, json: dict[str, Any], timeout: Any = None) -> _AsyncCtx:
            question = json["messages"][-1]["content"]
            if "1+1" in question:
                resp = _make_api_response(["\\boxed{2}"])
            else:
                resp = _make_api_response(["\\boxed{6}"])
            mock_resp = MagicMock()
            mock_resp.status = 200

            async def _json(_r: dict = resp) -> dict:
                return _r

            async def _text() -> str:
                return "{}"

            mock_resp.json = _json
            mock_resp.text = _text
            return _AsyncCtx(mock_resp)

        class FakeSession:
            def __init__(self, *a: Any, **kw: Any) -> None:
                self.post = smart_post
            async def __aenter__(self) -> "FakeSession":
                return self
            async def __aexit__(self, *a: Any) -> None:
                pass

        class FakeConnector:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

        with patch("pipeline.ops.evaluate.rollout.aiohttp") as m:
            m.ClientSession = FakeSession
            m.TCPConnector = FakeConnector
            m.ClientTimeout = lambda total: total
            result = proc.process([rec1, rec2])

        self.assertEqual(len(result.kept_records), 2)
        for rec in result.kept_records:
            v = rec.verification
            self.assertEqual(v["num_samples"], 4)
            self.assertTrue(v["majority_matches_gt"])


if __name__ == "__main__":
    unittest.main()

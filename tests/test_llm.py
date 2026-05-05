from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.ops.llm.backends import SGLangClient, VLLMClient, build_inference_client
from pipeline.ops.llm.client import BackendConfig, InferenceRequest, InferenceResponse


class FakeSGLangClient(SGLangClient):
    def _infer_once(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(text="pong", raw={"ok": True})


class LLMTests(unittest.TestCase):
    def test_router_builds_correct_client(self) -> None:
        config = BackendConfig(backend="sglang", base_url="http://localhost:30000", model="demo")
        client = build_inference_client(config)
        self.assertIsInstance(client, SGLangClient)

        config = BackendConfig(backend="vllm", base_url="http://localhost:8000", model="demo")
        client = build_inference_client(config)
        self.assertIsInstance(client, VLLMClient)

    def test_client_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = BackendConfig(
                backend="sglang",
                base_url="http://localhost:30000",
                model="demo",
                cache_dir=tmp_dir,
            )
            client = FakeSGLangClient(config)
            request = InferenceRequest(model="demo", prompt="ping")
            first = client.infer(request)
            second = client.infer(request)
            self.assertFalse(first.cached)
            self.assertTrue(second.cached)


if __name__ == "__main__":
    unittest.main()

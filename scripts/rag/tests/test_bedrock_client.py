from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bedrock_client import AwsCliBedrockClient


class BedrockClientTests(unittest.TestCase):
    def test_rerank_parses_json_response(self) -> None:
        captured: dict[str, object] = {}

        def runner(cmd, *, timeout_sec, retries, retry_backoff_sec):
            captured["cmd"] = cmd
            captured["timeout_sec"] = timeout_sec
            captured["retries"] = retries
            captured["retry_backoff_sec"] = retry_backoff_sec
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout='{"results":[{"index":0,"relevanceScore":0.9}]}',
                stderr="",
            )

        client = AwsCliBedrockClient(
            region="ap-northeast-1",
            profile="rag",
            timeout_sec=45,
            retries=1,
            retry_backoff_sec=1.0,
            runner=runner,
        )
        body = client.rerank(payload={"sources": [], "queries": []})
        self.assertIn("results", body)
        self.assertEqual(body["results"][0]["index"], 0)
        self.assertIn("bedrock-agent-runtime", captured["cmd"])

    def test_converse_returns_text_response(self) -> None:
        def runner(cmd, *, timeout_sec, retries, retry_backoff_sec):
            _ = timeout_sec
            _ = retries
            _ = retry_backoff_sec
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="answer text\n",
                stderr="",
            )

        client = AwsCliBedrockClient(
            region="ap-northeast-1",
            profile="rag",
            timeout_sec=45,
            retries=1,
            retry_backoff_sec=1.0,
            runner=runner,
        )
        answer = client.converse(
            model_id="dummy.model",
            payload={"messages": [{"role": "user", "content": [{"text": "q"}]}]},
        )
        self.assertEqual(answer, "answer text")

    def test_rerank_raises_when_json_is_invalid(self) -> None:
        def runner(cmd, *, timeout_sec, retries, retry_backoff_sec):
            _ = timeout_sec
            _ = retries
            _ = retry_backoff_sec
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="{invalid json",
                stderr="",
            )

        client = AwsCliBedrockClient(
            region="ap-northeast-1",
            profile="rag",
            timeout_sec=45,
            retries=1,
            retry_backoff_sec=1.0,
            runner=runner,
        )
        with self.assertRaises(RuntimeError):
            client.rerank(payload={"sources": [], "queries": []})


if __name__ == "__main__":
    unittest.main()

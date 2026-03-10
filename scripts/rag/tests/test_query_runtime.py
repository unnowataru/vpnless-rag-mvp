from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.query_runtime as query_runtime
from core.retriever_contract import RetrievalHit


class _CapturingBedrockClient:
    rerank_payload: dict[str, object] | None = None
    converse_payload: dict[str, object] | None = None

    def __init__(self, *, region, profile, timeout_sec, retries, retry_backoff_sec) -> None:
        _ = region
        _ = profile
        _ = timeout_sec
        _ = retries
        _ = retry_backoff_sec

    def rerank(self, *, payload):
        type(self).rerank_payload = payload
        return {"results": []}

    def converse(self, *, model_id, payload):
        _ = model_id
        type(self).converse_payload = payload
        return "stub-answer"


class QueryRuntimeSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_client = query_runtime.AwsCliBedrockClient
        _CapturingBedrockClient.rerank_payload = None
        _CapturingBedrockClient.converse_payload = None
        query_runtime.AwsCliBedrockClient = _CapturingBedrockClient

    def tearDown(self) -> None:
        query_runtime.AwsCliBedrockClient = self._orig_client

    def test_call_bedrock_sanitizes_question_before_converse(self) -> None:
        answer = query_runtime.call_bedrock(
            question="連絡先は foo.bar@example.com と 03-1234-5678 です",
            evidence="[1] score(...) doc=doc\n根拠テキスト\n",
            system_prompt="system",
            max_tokens=128,
            region="ap-northeast-1",
            profile="rag",
            model_id="dummy.model",
            timeout_sec=5,
            retries=0,
            retry_backoff_sec=0.1,
        )

        self.assertEqual(answer, "stub-answer")
        payload = _CapturingBedrockClient.converse_payload
        self.assertIsNotNone(payload)
        prompt = payload["messages"][0]["content"][0]["text"]
        self.assertIn("[EMAIL]", prompt)
        self.assertIn("[TEL]", prompt)
        self.assertNotIn("foo.bar@example.com", prompt)
        self.assertNotIn("03-1234-5678", prompt)

    def test_rerank_hits_sanitizes_question_and_sources_before_rerank(self) -> None:
        hit = RetrievalHit(
            chunk_id="DOC_POLICY_001:p1:c0:o0:h001",
            score=0.91,
            text_snippet="規程本文 foo.bar@example.com 03-1234-5678",
            doc_meta={"doc": "DOC_POLICY_001", "doc_id": "DOC_POLICY_001", "page": 1, "chunk": 0},
            section_path=(),
            labels=("policy",),
            vector_score=0.91,
            rerank_score=None,
            metadata_index=0,
            extra={},
        )

        reranked = query_runtime.rerank_hits(
            question="問い合わせ先 foo.bar@example.com 03-1234-5678",
            hits=[hit],
            metadata=[{"text": "根拠 foo.bar@example.com 03-1234-5678"}],
            region="ap-northeast-1",
            profile="rag",
            rerank_model="amazon.rerank-v1:0",
            rerank_topn=1,
            timeout_sec=5,
            retries=0,
            retry_backoff_sec=0.1,
        )

        self.assertEqual(reranked, [hit])
        payload = _CapturingBedrockClient.rerank_payload
        self.assertIsNotNone(payload)
        query_text = payload["queries"][0]["textQuery"]["text"]
        source_text = payload["sources"][0]["inlineDocumentSource"]["textDocument"]["text"]
        self.assertEqual(query_text, "問い合わせ先 [EMAIL] [TEL]")
        self.assertEqual(source_text, "根拠 [EMAIL] [TEL]")


if __name__ == "__main__":
    unittest.main()

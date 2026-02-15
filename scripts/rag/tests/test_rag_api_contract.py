from __future__ import annotations

import json
import sys
import threading
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

# Keep test import light in CI environment where numpy/sentence-transformers
# are not installed.
if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.ndarray = object
    sys.modules["numpy"] = numpy_stub
if "sentence_transformers" not in sys.modules:
    st_stub = types.ModuleType("sentence_transformers")

    class _DummySentenceTransformer:  # pragma: no cover - import stub only
        def __init__(self, *args, **kwargs):
            pass

    st_stub.SentenceTransformer = _DummySentenceTransformer
    sys.modules["sentence_transformers"] = st_stub

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.retriever_contract import RetrievalHit
import rag_api_server


class _DummyDiagnostics:
    def __init__(self, hits: list[RetrievalHit], stats: dict[str, object]) -> None:
        self.hits = hits
        self.stats = stats


class _DummyLocalRetriever:
    def __init__(self, hits: list[RetrievalHit] | None = None) -> None:
        self._hits = hits or []

    def search_with_diagnostics(self, query_text: str, top_k: int, filters: dict[str, object]):
        _ = query_text
        _ = top_k
        _ = filters
        stats = {
            "hits_before_filter": len(self._hits),
            "hits_after_filter": len(self._hits),
            "hits_after_rerank": len(self._hits),
            "filter_pass_rate": 1.0 if self._hits else 0.0,
            "zero_hit": len(self._hits) == 0,
        }
        return _DummyDiagnostics(self._hits, stats)


class _DummyFallbackRetriever:
    def __init__(self, *, error: str | None = None) -> None:
        self._error = error

    def search_with_fallback(self, query_text: str, top_k: int, filters: dict[str, object]):
        _ = query_text
        _ = top_k
        _ = filters
        if self._error:
            raise RuntimeError(self._error)
        return SimpleNamespace(
            hits=[],
            backend_used="dummy",
            fallback_triggered=False,
            error=None,
        )


def _build_context(
    *,
    local_hits: list[RetrievalHit] | None = None,
    vast_error: str | None = None,
    allow_unscoped_default: bool = False,
) -> rag_api_server.AppContext:
    runtime_config = SimpleNamespace(
        default_retrieval_filters={},
        temporal_rules=(),
        answer_profiles=(SimpleNamespace(key="cost"),),
        answer_profile_to_model={"cost": "dummy.model"},
    )
    return rag_api_server.AppContext(
        local_retriever=_DummyLocalRetriever(local_hits),
        vast_retriever=_DummyFallbackRetriever(error=vast_error),
        external_retriever=_DummyFallbackRetriever(),
        metadata=[],
        runtime_config=runtime_config,
        system_prompt="test prompt",
        region="ap-northeast-1",
        profile="rag",
        default_topk=5,
        default_rerank=False,
        rerank_model="amazon.rerank-v1:0",
        rerank_topn=0,
        max_context_chars=12000,
        max_tokens=128,
        aws_timeout_sec=5,
        aws_retries=0,
        aws_retry_backoff_sec=0.1,
        allow_unscoped_default=allow_unscoped_default,
        auto_scope_max_docs=4,
        audit_log_dir=None,
    )


class RagApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_bedrock = rag_api_server.call_bedrock
        rag_api_server.call_bedrock = lambda **kwargs: "stub-answer"
        rag_api_server.RagRequestHandler.log_message = lambda *_args: None

    def tearDown(self) -> None:
        rag_api_server.call_bedrock = self._orig_bedrock

    def _with_server(self, ctx: rag_api_server.AppContext):
        server = rag_api_server.RagHTTPServer(("127.0.0.1", 0), rag_api_server.RagRequestHandler)
        server.context = ctx
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _request(
        self,
        *,
        server: rag_api_server.RagHTTPServer,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        raw: bytes | None = None,
    ) -> tuple[int, dict[str, object]]:
        url = f"http://127.0.0.1:{server.server_port}{path}"
        if raw is not None:
            data = raw
        elif payload is None:
            data = None
        else:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            method=method,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            return exc.code, json.loads(body) if body else {}

    def test_health(self) -> None:
        server, thread = self._with_server(_build_context())
        try:
            status, body = self._request(server=server, method="GET", path="/health")
            self.assertEqual(status, 200)
            self.assertEqual(body.get("status"), "ok")
            self.assertEqual(body.get("service"), "rag-api")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_search_bad_request_when_query_missing(self) -> None:
        server, thread = self._with_server(_build_context())
        try:
            status, body = self._request(server=server, method="POST", path="/search", payload={})
            self.assertEqual(status, 400)
            self.assertIn("query_text is required", str(body.get("error")))
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_search_fail_closed_when_scope_unresolved(self) -> None:
        server, thread = self._with_server(_build_context(allow_unscoped_default=False))
        try:
            status, body = self._request(
                server=server,
                method="POST",
                path="/search",
                payload={"query_text": "scopeが決められない質問", "top_k": 3},
            )
            self.assertEqual(status, 502)
            self.assertIn("Failed to resolve retrieval scope", str(body.get("error")))
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_search_backend_error_returns_502(self) -> None:
        server, thread = self._with_server(_build_context(vast_error="vast backend unavailable"))
        try:
            status, body = self._request(
                server=server,
                method="POST",
                path="/search",
                payload={
                    "query_text": "dummy",
                    "retriever_backend": "vast",
                    "allow_unscoped": True,
                },
            )
            self.assertEqual(status, 502)
            self.assertIn("vast backend unavailable", str(body.get("error")))
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_qa_bad_request_when_question_missing(self) -> None:
        server, thread = self._with_server(_build_context())
        try:
            status, body = self._request(server=server, method="POST", path="/qa", payload={})
            self.assertEqual(status, 400)
            self.assertIn("question is required", str(body.get("error")))
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_qa_success_contract_shape(self) -> None:
        hit = RetrievalHit(
            chunk_id="DOC_POLICY_001:p1:c0:o0:h001",
            score=0.91,
            text_snippet="規程の要点",
            doc_meta={"doc": "DOC_POLICY_001", "doc_id": "DOC_POLICY_001", "page": 1, "chunk": 0},
            section_path=(),
            labels=("policy",),
            vector_score=0.91,
            rerank_score=None,
            metadata_index=0,
            extra={},
        )
        server, thread = self._with_server(_build_context(local_hits=[hit], allow_unscoped_default=True))
        try:
            status, body = self._request(
                server=server,
                method="POST",
                path="/qa",
                payload={
                    "question": "規程の要点は？",
                    "top_k": 5,
                    "answer_profile": "cost",
                    "rerank": False,
                    "allow_unscoped": True,
                },
            )
            self.assertEqual(status, 200)
            self.assertEqual(body.get("status"), "success")
            self.assertEqual(body.get("answer"), "stub-answer")
            self.assertEqual(body.get("model_id"), "dummy.model")
            self.assertEqual(body.get("scope_source"), "allow_unscoped")
            self.assertIsInstance(body.get("retrieval_stats"), dict)
            self.assertIsInstance(body.get("evidence_entries"), list)
            self.assertGreaterEqual(len(body.get("evidence_entries", [])), 2)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_search_invalid_json_returns_400(self) -> None:
        server, thread = self._with_server(_build_context())
        try:
            status, body = self._request(
                server=server,
                method="POST",
                path="/search",
                raw=b"{invalid-json",
            )
            self.assertEqual(status, 400)
            self.assertIn("Invalid JSON body", str(body.get("error")))
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


if __name__ == "__main__":
    unittest.main()

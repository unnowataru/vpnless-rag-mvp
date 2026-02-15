from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.retriever_contract import RetrievalHit
from core.retriever_contract import RetrieverBackendError
from core.retriever_fallback import FallbackRetriever


class _FailingBackendRetriever:
    def search(self, query_text: str, top_k: int, filters: dict | None = None) -> list[RetrievalHit]:
        _ = query_text
        _ = top_k
        _ = filters
        raise RetrieverBackendError("backend unavailable")


class _BadRequestRetriever:
    def search(self, query_text: str, top_k: int, filters: dict | None = None) -> list[RetrievalHit]:
        _ = query_text
        _ = top_k
        _ = filters
        raise ValueError("invalid filter")


class _SuccessRetriever:
    def search(self, query_text: str, top_k: int, filters: dict | None = None) -> list[RetrievalHit]:
        _ = query_text
        _ = top_k
        _ = filters
        return [
            RetrievalHit(
                chunk_id="doc:p1:c0:o0:h123",
                score=1.0,
                text_snippet="snippet",
                doc_meta={"doc_id": "doc", "doc": "doc", "page": 1, "chunk": 0},
            )
        ]


class FallbackRetrieverTests(unittest.TestCase):
    def test_fallback_runs_on_backend_error(self) -> None:
        retriever = FallbackRetriever(
            primary_name="vast",
            primary=_FailingBackendRetriever(),
            fallback_name="local",
            fallback=_SuccessRetriever(),
        )
        result = retriever.search_with_fallback(query_text="q", top_k=3, filters={})
        self.assertTrue(result.fallback_triggered)
        self.assertEqual(result.backend_used, "local")
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.error, "backend unavailable")

    def test_fallback_does_not_swallow_non_backend_error(self) -> None:
        retriever = FallbackRetriever(
            primary_name="vast",
            primary=_BadRequestRetriever(),
            fallback_name="local",
            fallback=_SuccessRetriever(),
        )
        with self.assertRaises(ValueError):
            retriever.search_with_fallback(query_text="q", top_k=3, filters={})


if __name__ == "__main__":
    unittest.main()

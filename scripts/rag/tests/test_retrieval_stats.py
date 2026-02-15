from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.retrieval_stats import attach_backend_metadata
from core.retrieval_stats import build_retrieval_stats


class RetrievalStatsTests(unittest.TestCase):
    def test_build_retrieval_stats_values(self) -> None:
        stats = build_retrieval_stats(10, 4, 3)
        self.assertEqual(stats["hits_before_filter"], 10)
        self.assertEqual(stats["hits_after_filter"], 4)
        self.assertEqual(stats["hits_after_rerank"], 3)
        self.assertEqual(stats["filter_pass_rate"], 0.4)
        self.assertFalse(stats["zero_hit"])

    def test_build_retrieval_stats_zero_base(self) -> None:
        stats = build_retrieval_stats(0, 0, 0)
        self.assertEqual(stats["filter_pass_rate"], 0.0)
        self.assertTrue(stats["zero_hit"])

    def test_attach_backend_metadata_keeps_stats_and_adds_keys(self) -> None:
        base = build_retrieval_stats(2, 1, 1)
        merged = attach_backend_metadata(
            base,
            backend_used="vast",
            fallback_triggered=True,
            fallback_error="backend unavailable",
            fallback_error_type="backend_error",
        )
        self.assertEqual(merged["hits_before_filter"], 2)
        self.assertEqual(merged["retriever_backend_used"], "vast")
        self.assertTrue(merged["fallback_triggered"])
        self.assertEqual(merged["fallback_error"], "backend unavailable")
        self.assertEqual(merged["fallback_error_type"], "backend_error")


if __name__ == "__main__":
    unittest.main()

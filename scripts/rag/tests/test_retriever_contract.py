from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.retriever_contract import build_hit_from_row
from core.retriever_contract import normalize_search_score
from core.retriever_contract import validate_filters


class RetrieverContractTests(unittest.TestCase):
    def test_validate_filters_allows_known_keys(self) -> None:
        filters = validate_filters(
            {
                "doc_id": "DOC_HANDBOOK",
                "label": ["hr", "policy"],
                "updated_at": {"gte": "2025-01-01", "lte": "2025-12-31"},
                "dept": "hr",
                "confidentiality": "internal",
            }
        )
        self.assertEqual(filters["doc_id"], "DOC_HANDBOOK")

    def test_validate_filters_rejects_unknown_key(self) -> None:
        with self.assertRaises(ValueError):
            validate_filters({"tenant": "x"})

    def test_build_hit_from_row_generates_stable_chunk_id(self) -> None:
        row = {
            "doc": "DOC_SAMPLE",
            "page": 3,
            "chunk": 7,
            "text": "This is a sample chunk body.",
        }
        hit_a = build_hit_from_row(row=row, score=0.8, fallback_index=100)
        hit_b = build_hit_from_row(row=row, score=0.4, fallback_index=999)
        self.assertEqual(hit_a.chunk_id, hit_b.chunk_id)
        self.assertTrue(hit_a.chunk_id.startswith("DOC_SAMPLE:p3:c7:"))

    def test_build_hit_from_row_uses_existing_chunk_id(self) -> None:
        row = {
            "chunk_id": "fixed-id-001",
            "doc": "DOC_SAMPLE",
            "text": "sample",
        }
        hit = build_hit_from_row(row=row, score=0.9)
        self.assertEqual(hit.chunk_id, "fixed-id-001")

    def test_normalize_search_score_distance(self) -> None:
        self.assertAlmostEqual(normalize_search_score(0.0, kind="distance"), 1.0)
        self.assertLess(normalize_search_score(5.0, kind="distance"), 0.2)


if __name__ == "__main__":
    unittest.main()

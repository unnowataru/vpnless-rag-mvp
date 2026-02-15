from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.scope_resolver import extract_scope_terms
from core.scope_resolver import infer_doc_id_scope_filters


class ScopeResolverTests(unittest.TestCase):
    def test_extract_scope_terms_includes_prefix_and_alias(self) -> None:
        terms = extract_scope_terms("出張申請について知りたい")
        self.assertIn("出張申請", terms)
        self.assertIn("出張", terms)

        traffic_terms = extract_scope_terms("交通費について知りたい")
        self.assertIn("旅費", traffic_terms)

    def test_infer_doc_id_scope_filters(self) -> None:
        metadata = [
            {"doc": "DOC_TRAVEL_TAG_A", "text": "..."},  # expected match
            {"doc": "DOC_TRAVEL_TAG_B", "text": "..."},  # expected match
            {"doc": "DOC_POLICY_GENERIC", "text": "..."},
        ]
        filters = infer_doc_id_scope_filters("交通費について知りたい", metadata, max_docs=3)
        self.assertIn("doc_id", filters)
        self.assertGreaterEqual(len(filters["doc_id"]), 1)
        self.assertTrue(any("旅費" in doc for doc in filters["doc_id"]))

    def test_infer_doc_id_scope_filters_empty_when_no_match(self) -> None:
        metadata = [{"doc": "DOC_POLICY_GENERIC", "text": "..."}]
        filters = infer_doc_id_scope_filters("量子コンピュータの話", metadata, max_docs=3)
        self.assertEqual(filters, {})


if __name__ == "__main__":
    unittest.main()

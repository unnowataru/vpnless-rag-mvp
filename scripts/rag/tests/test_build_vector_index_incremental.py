from __future__ import annotations

import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_vector_index_incremental import merge_doc_rows
from build_vector_index_incremental import parse_deleted_doc_ids


def _row(doc_id: str, text: str, page: int, chunk: int) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "doc": doc_id,
        "text": text,
        "page": page,
        "chunk": chunk,
        "start_offset": 0,
    }


class BuildVectorIndexIncrementalTests(unittest.TestCase):
    def test_parse_deleted_doc_ids(self) -> None:
        deleted = parse_deleted_doc_ids("DOC_A, DOC_B, ,DOC_C")
        self.assertEqual(deleted, {"DOC_A", "DOC_B", "DOC_C"})

    def test_merge_doc_rows_tracks_new_changed_and_deleted(self) -> None:
        existing = {
            "DOC_A": [_row("DOC_A", "old text", 1, 0)],
            "DOC_B": [_row("DOC_B", "keep or delete", 1, 0)],
        }
        new = {
            "DOC_A": [_row("DOC_A", "new text", 1, 0)],
            "DOC_C": [_row("DOC_C", "brand new", 1, 0)],
        }
        merged, stats = merge_doc_rows(existing, new, {"DOC_B"})
        merged_doc_ids = {str(row["doc_id"]) for row in merged}

        self.assertEqual(merged_doc_ids, {"DOC_A", "DOC_C"})
        self.assertEqual(stats["new_docs"], 1)
        self.assertEqual(stats["changed_docs"], 1)
        self.assertEqual(stats["unchanged_docs"], 0)

    def test_merge_doc_rows_tracks_unchanged(self) -> None:
        existing = {"DOC_A": [_row("DOC_A", "same text", 1, 0)]}
        new = {"DOC_A": [_row("DOC_A", "same text", 1, 0)]}
        merged, stats = merge_doc_rows(existing, new, set())

        self.assertEqual(len(merged), 1)
        self.assertEqual(stats["new_docs"], 0)
        self.assertEqual(stats["changed_docs"], 0)
        self.assertEqual(stats["unchanged_docs"], 1)


if __name__ == "__main__":
    unittest.main()

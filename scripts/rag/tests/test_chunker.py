from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.chunker import infer_chunk_type
from core.chunker import split_chunks


class ChunkerTests(unittest.TestCase):
    def test_split_chunks_keeps_section_path_from_heading(self) -> None:
        text = (
            "第1条 総則\n"
            "この規程は会社の運用に関する基本事項を定める。\n\n"
            "第2条 対象\n"
            "本制度の対象者は全社員とする。"
        )
        chunks = split_chunks(text, chunk_size=80, overlap=0, min_chars=10)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("section_path", chunks[0])
        self.assertTrue(chunks[0]["section_path"])

    def test_split_chunks_applies_overlap_tail(self) -> None:
        text = "A " * 120 + "\n\n" + "B " * 120
        chunks = split_chunks(text, chunk_size=180, overlap=20, min_chars=20)
        self.assertGreaterEqual(len(chunks), 2)
        second_text = str(chunks[1]["text"])
        self.assertGreater(len(second_text), 0)

    def test_infer_chunk_type_detects_table_like(self) -> None:
        text = "項目\t値\nA\t100\nB\t200\nC\t300"
        self.assertIn(infer_chunk_type(text), {"table", "table_like"})


if __name__ == "__main__":
    unittest.main()

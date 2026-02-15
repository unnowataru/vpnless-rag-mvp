from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pdf_extractor import iter_pdfs


class PdfExtractorTests(unittest.TestCase):
    def test_iter_pdfs_filters_pdf_suffix_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.pdf").write_text("x", encoding="utf-8")
            (root / "b.PDF").write_text("x", encoding="utf-8")
            (root / "c.txt").write_text("x", encoding="utf-8")
            files = iter_pdfs(root, "*")
            names = [f.name for f in files]
            self.assertEqual(names, ["a.pdf", "b.PDF"])


if __name__ == "__main__":
    unittest.main()

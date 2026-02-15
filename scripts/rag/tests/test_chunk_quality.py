from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.chunk_quality import PageExtractionCandidate
from core.chunk_quality import compute_text_quality_metrics
from core.chunk_quality import detect_repeated_margin_lines
from core.chunk_quality import remove_margin_noise
from core.chunk_quality import select_best_candidate


class ChunkQualityTests(unittest.TestCase):
    def test_compute_text_quality_metrics_non_empty(self) -> None:
        metrics = compute_text_quality_metrics("line1\nline2\nline2")
        self.assertGreater(metrics["char_count"], 0.0)
        self.assertGreaterEqual(metrics["score"], 0.0)

    def test_select_best_candidate_picks_highest_score(self) -> None:
        candidates = [
            PageExtractionCandidate(engine="a", text="x", score=0.2, metrics={"score": 0.2}),
            PageExtractionCandidate(engine="b", text="y", score=0.7, metrics={"score": 0.7}),
        ]
        selected = select_best_candidate(candidates)
        self.assertEqual(selected.engine, "b")

    def test_detect_and_remove_repeated_margin_lines(self) -> None:
        pages = [
            "HEADER\n本文A\nFOOTER",
            "HEADER\n本文B\nFOOTER",
            "HEADER\n本文C\nFOOTER",
        ]
        repeated = detect_repeated_margin_lines(pages, max_lines=1, threshold_ratio=0.6)
        self.assertIn("HEADER", repeated)
        self.assertIn("FOOTER", repeated)
        cleaned = remove_margin_noise(pages[0], repeated)
        self.assertNotIn("HEADER", cleaned)
        self.assertNotIn("FOOTER", cleaned)


if __name__ == "__main__":
    unittest.main()

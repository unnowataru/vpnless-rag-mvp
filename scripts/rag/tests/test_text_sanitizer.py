from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.text_sanitizer import sanitize


class TextSanitizerTests(unittest.TestCase):
    def test_sanitize_masks_email_and_tel(self) -> None:
        text = "contact: foo.bar@example.com / 03-1234-5678"
        self.assertEqual(sanitize(text), "contact: [EMAIL] / [TEL]")

    def test_sanitize_leaves_unrelated_text(self) -> None:
        text = "plain text only"
        self.assertEqual(sanitize(text), text)


if __name__ == "__main__":
    unittest.main()

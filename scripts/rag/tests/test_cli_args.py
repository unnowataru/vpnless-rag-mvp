from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cli_args import validate_args


def make_args(**overrides: object) -> SimpleNamespace:
    base = {
        "topk": 5,
        "snippet_max_chars": 1200,
        "auto_scope_max_docs": 6,
        "aws_timeout_sec": 45,
        "aws_retries": 1,
        "aws_retry_backoff_sec": 1.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class CliArgsValidationTests(unittest.TestCase):
    def test_validate_args_accepts_valid_input(self) -> None:
        validate_args(make_args())

    def test_validate_args_rejects_invalid_values(self) -> None:
        bad_cases = [
            ("topk", 0, "--topk must be greater than 0."),
            ("snippet_max_chars", 0, "--snippet-max-chars must be greater than 0."),
            ("auto_scope_max_docs", 0, "--auto-scope-max-docs must be greater than 0."),
            ("aws_timeout_sec", 0, "--aws-timeout-sec must be greater than 0."),
            ("aws_retries", -1, "--aws-retries must be 0 or greater."),
            ("aws_retry_backoff_sec", 0.0, "--aws-retry-backoff-sec must be greater than 0."),
        ]
        for key, value, expected in bad_cases:
            with self.subTest(key=key):
                with self.assertRaises(SystemExit) as ctx:
                    validate_args(make_args(**{key: value}))
                self.assertEqual(str(ctx.exception), expected)


if __name__ == "__main__":
    unittest.main()

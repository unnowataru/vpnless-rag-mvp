from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.app_config import build_app_config_from_args
from core.app_config import validate_app_config


class AppConfigTests(unittest.TestCase):
    def test_build_from_cli_args_uses_allow_unscoped(self) -> None:
        args = SimpleNamespace(
            topk=5,
            rerank=True,
            rerank_model="amazon.rerank-v1:0",
            rerank_topn=0,
            max_context_chars=12000,
            max_tokens=512,
            snippet_max_chars=1200,
            region="ap-northeast-1",
            profile="rag",
            aws_timeout_sec=45,
            aws_retries=1,
            aws_retry_backoff_sec=1.0,
            auto_scope_max_docs=6,
            allow_unscoped=True,
        )
        config = build_app_config_from_args(args)
        self.assertTrue(config.allow_unscoped)
        self.assertIsNone(config.port)

    def test_build_from_api_args_uses_allow_unscoped_default_and_port(self) -> None:
        args = SimpleNamespace(
            topk=5,
            rerank=False,
            rerank_model="amazon.rerank-v1:0",
            rerank_topn=0,
            max_context_chars=12000,
            max_tokens=512,
            snippet_max_chars=1200,
            region="ap-northeast-1",
            profile="rag",
            aws_timeout_sec=45,
            aws_retries=1,
            aws_retry_backoff_sec=1.0,
            auto_scope_max_docs=6,
            allow_unscoped_default=True,
            port=8000,
        )
        config = build_app_config_from_args(args)
        self.assertTrue(config.allow_unscoped)
        self.assertEqual(config.port, 8000)

    def test_validate_requires_positive_port_when_requested(self) -> None:
        args = SimpleNamespace(
            topk=5,
            rerank=False,
            rerank_model="amazon.rerank-v1:0",
            rerank_topn=0,
            max_context_chars=12000,
            max_tokens=512,
            snippet_max_chars=1200,
            region="ap-northeast-1",
            profile="rag",
            aws_timeout_sec=45,
            aws_retries=1,
            aws_retry_backoff_sec=1.0,
            auto_scope_max_docs=6,
            allow_unscoped_default=False,
            port=0,
        )
        config = build_app_config_from_args(args)
        with self.assertRaises(SystemExit) as ctx:
            validate_app_config(config, require_port=True)
        self.assertEqual(str(ctx.exception), "--port must be greater than 0.")


if __name__ == "__main__":
    unittest.main()

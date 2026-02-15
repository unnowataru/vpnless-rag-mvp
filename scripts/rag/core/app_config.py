"""Shared app configuration defaults and validation for CLI/API."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

DEFAULT_RUNTIME_CONFIG_FILE = "scripts/rag/config/runtime_config.json"
DEFAULT_REGION = "ap-northeast-1"
DEFAULT_PROFILE = "rag"
DEFAULT_RERANK_MODEL = "amazon.rerank-v1:0"


@dataclass(frozen=True)
class AppConfig:
    topk: int
    rerank: bool
    rerank_model: str
    rerank_topn: int
    max_context_chars: int
    max_tokens: int
    snippet_max_chars: int
    region: str
    profile: str
    aws_timeout_sec: int
    aws_retries: int
    aws_retry_backoff_sec: float
    auto_scope_max_docs: int
    allow_unscoped: bool
    port: int | None = None


def build_app_config_from_args(args: argparse.Namespace) -> AppConfig:
    allow_unscoped = bool(
        getattr(
            args,
            "allow_unscoped",
            getattr(args, "allow_unscoped_default", False),
        )
    )
    port_value = getattr(args, "port", None)
    return AppConfig(
        topk=int(args.topk),
        rerank=bool(args.rerank),
        rerank_model=str(args.rerank_model),
        rerank_topn=int(args.rerank_topn),
        max_context_chars=int(args.max_context_chars),
        max_tokens=int(args.max_tokens),
        snippet_max_chars=int(args.snippet_max_chars),
        region=str(args.region),
        profile=str(args.profile),
        aws_timeout_sec=int(args.aws_timeout_sec),
        aws_retries=int(args.aws_retries),
        aws_retry_backoff_sec=float(args.aws_retry_backoff_sec),
        auto_scope_max_docs=int(args.auto_scope_max_docs),
        allow_unscoped=allow_unscoped,
        port=int(port_value) if port_value is not None else None,
    )


def validate_app_config(config: AppConfig, *, require_port: bool) -> None:
    if config.topk <= 0:
        raise SystemExit("--topk must be greater than 0.")
    if config.snippet_max_chars <= 0:
        raise SystemExit("--snippet-max-chars must be greater than 0.")
    if config.max_context_chars <= 0:
        raise SystemExit("--max-context-chars must be greater than 0.")
    if config.max_tokens <= 0:
        raise SystemExit("--max-tokens must be greater than 0.")
    if config.auto_scope_max_docs <= 0:
        raise SystemExit("--auto-scope-max-docs must be greater than 0.")
    if config.aws_timeout_sec <= 0:
        raise SystemExit("--aws-timeout-sec must be greater than 0.")
    if config.aws_retries < 0:
        raise SystemExit("--aws-retries must be 0 or greater.")
    if config.aws_retry_backoff_sec <= 0:
        raise SystemExit("--aws-retry-backoff-sec must be greater than 0.")
    if require_port:
        if config.port is None:
            raise SystemExit("--port is required.")
        if config.port <= 0:
            raise SystemExit("--port must be greater than 0.")

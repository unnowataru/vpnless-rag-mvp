"""Argument parsing and process bootstrap for RAG API server."""

from __future__ import annotations

import argparse
from pathlib import Path

from api_endpoints import AppContext
from api_transport import RagHTTPServer
from api_transport import RagRequestHandler
from core.app_config import DEFAULT_PROFILE
from core.app_config import DEFAULT_REGION
from core.app_config import DEFAULT_RERANK_MODEL
from core.app_config import DEFAULT_RUNTIME_CONFIG_FILE
from core.app_config import build_app_config_from_args
from core.app_config import validate_app_config
from core.bootstrap import build_backend_retrievers
from core.bootstrap import build_local_retriever
from core.bootstrap import load_runtime_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--runtime-config-file", default=DEFAULT_RUNTIME_CONFIG_FILE)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--rerank-topn", type=int, default=0)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--snippet-max-chars", type=int, default=1200)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--vast-endpoint", default="")
    parser.add_argument("--vast-collection", default="default")
    parser.add_argument("--external-endpoint", default="")
    parser.add_argument("--external-provider", default="netapp")
    parser.add_argument(
        "--local-fallback-on-retriever-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--aws-timeout-sec", type=int, default=45)
    parser.add_argument("--aws-retries", type=int, default=1)
    parser.add_argument("--aws-retry-backoff-sec", type=float, default=1.0)
    parser.add_argument("--allow-unscoped-default", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-scope-max-docs", type=int, default=6)
    parser.add_argument(
        "--cors-allow-origin",
        default="*",
        help="CORS Access-Control-Allow-Origin value for API responses (PoC default: *).",
    )
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument("--audit-log-dir", default=None)
    return parser.parse_args()


def build_context(args: argparse.Namespace) -> tuple[AppContext, int]:
    app_config = build_app_config_from_args(args)
    validate_app_config(app_config, require_port=True)

    index_dir = Path(args.index_dir)
    artifacts = load_runtime_artifacts(
        index_dir=index_dir,
        runtime_config_file=args.runtime_config_file,
        system_prompt_file=args.system_prompt_file,
    )
    local_retriever = build_local_retriever(
        index_dir=index_dir,
        manifest=artifacts.manifest,
        metadata=artifacts.metadata,
        snippet_chars=app_config.snippet_max_chars,
    )
    vast_retriever, external_retriever = build_backend_retrievers(
        local_retriever=local_retriever,
        vast_endpoint=args.vast_endpoint,
        vast_collection=args.vast_collection,
        external_endpoint=args.external_endpoint,
        external_provider=args.external_provider,
        timeout_sec=app_config.aws_timeout_sec,
        local_fallback_on_retriever_error=args.local_fallback_on_retriever_error,
    )

    context = AppContext(
        local_retriever=local_retriever,
        vast_retriever=vast_retriever,
        external_retriever=external_retriever,
        metadata=artifacts.metadata,
        runtime_config=artifacts.runtime_config,
        system_prompt=artifacts.system_prompt,
        region=app_config.region,
        profile=app_config.profile,
        default_topk=app_config.topk,
        default_rerank=app_config.rerank,
        rerank_model=app_config.rerank_model,
        rerank_topn=app_config.rerank_topn,
        max_context_chars=app_config.max_context_chars,
        max_tokens=app_config.max_tokens,
        aws_timeout_sec=app_config.aws_timeout_sec,
        aws_retries=app_config.aws_retries,
        aws_retry_backoff_sec=app_config.aws_retry_backoff_sec,
        allow_unscoped_default=app_config.allow_unscoped,
        auto_scope_max_docs=app_config.auto_scope_max_docs,
        audit_log_dir=args.audit_log_dir,
        vast_endpoint=args.vast_endpoint,
        external_endpoint=args.external_endpoint,
        local_fallback_on_retriever_error=args.local_fallback_on_retriever_error,
        cors_allow_origin=args.cors_allow_origin,
    )
    server_port = app_config.port if app_config.port is not None else args.port
    return context, server_port


def main() -> None:
    args = parse_args()
    context, server_port = build_context(args)
    server = RagHTTPServer((args.host, server_port), RagRequestHandler)
    server.context = context
    print(f"RAG API server listening on http://{args.host}:{server_port}")
    server.serve_forever()

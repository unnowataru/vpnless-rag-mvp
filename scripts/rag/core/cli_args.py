"""CLI argument parser and validation for rag_vector_cli."""

from __future__ import annotations

import argparse

DEFAULT_REGION = "ap-northeast-1"
DEFAULT_PROFILE = "rag"
DEFAULT_RERANK_MODEL = "amazon.rerank-v1:0"
DEFAULT_RUNTIME_CONFIG_FILE = "scripts/rag/config/runtime_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True, help="Path to vector index directory")
    parser.add_argument(
        "--runtime-config-file",
        default=DEFAULT_RUNTIME_CONFIG_FILE,
        help="Path to runtime config JSON (profiles + temporal rules + default prompt path).",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument(
        "--rerank-topn",
        type=int,
        default=0,
        help="How many reranked results to request. 0 means all vector hits.",
    )
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument(
        "--snippet-max-chars",
        type=int,
        default=1200,
        help="Max chars per hit snippet sent to LLM (separate from full chunk storage).",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument(
        "--retriever-backend",
        choices=["local", "vast", "external"],
        default="local",
        help="Retriever backend contract target. vast/external uses local fallback by default.",
    )
    parser.add_argument(
        "--vast-endpoint",
        default="",
        help="VAST endpoint (for retriever-backend=vast).",
    )
    parser.add_argument(
        "--vast-collection",
        default="default",
        help="VAST collection/table identifier (for retriever-backend=vast).",
    )
    parser.add_argument(
        "--external-endpoint",
        default="",
        help="External retriever endpoint (for retriever-backend=external).",
    )
    parser.add_argument(
        "--external-provider",
        default="netapp",
        help="External retriever provider label.",
    )
    parser.add_argument(
        "--local-fallback-on-retriever-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to local retriever when non-local backend fails.",
    )
    parser.add_argument(
        "--aws-timeout-sec",
        type=int,
        default=45,
        help="Timeout seconds for AWS CLI calls (rerank/converse).",
    )
    parser.add_argument(
        "--aws-retries",
        type=int,
        default=1,
        help="Retry count for AWS CLI calls after first attempt.",
    )
    parser.add_argument(
        "--aws-retry-backoff-sec",
        type=float,
        default=1.0,
        help="Base backoff seconds for AWS CLI retries.",
    )
    parser.add_argument(
        "--fail-on-generation-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, generation errors raise non-zero exit instead of fallback answer.",
    )
    parser.add_argument(
        "--answer-profile",
        default=None,
        help="Answer profile key defined in runtime config.",
    )
    parser.add_argument(
        "--bedrock-model",
        default=None,
        help="Override Bedrock model ID directly. If set, --answer-profile is ignored.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Path to a custom system prompt text file.",
    )
    parser.add_argument(
        "--audit-log-dir",
        default=None,
        help="Directory to persist per-query audit logs as JSON.",
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Optional request ID for audit tracing. If omitted, auto-generated.",
    )
    parser.add_argument(
        "--filters-json",
        default=None,
        help=(
            "Optional retrieval filters JSON. Allowed keys: "
            "doc_id,label,updated_at,dept,confidentiality,customer,product,doc_type,retention."
        ),
    )
    parser.add_argument(
        "--auto-scope-max-docs",
        type=int,
        default=6,
        help="Max doc_id candidates for auto scope inference from question text.",
    )
    parser.add_argument(
        "--allow-unscoped",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow empty retrieval scope when auto/default scope cannot be resolved.",
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run in interactive multi-turn mode. Type 'exit' or 'quit' to finish.",
    )
    parser.add_argument("question", nargs="*")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.topk <= 0:
        raise SystemExit("--topk must be greater than 0.")
    if args.snippet_max_chars <= 0:
        raise SystemExit("--snippet-max-chars must be greater than 0.")
    if args.auto_scope_max_docs <= 0:
        raise SystemExit("--auto-scope-max-docs must be greater than 0.")
    if args.aws_timeout_sec <= 0:
        raise SystemExit("--aws-timeout-sec must be greater than 0.")
    if args.aws_retries < 0:
        raise SystemExit("--aws-retries must be 0 or greater.")
    if args.aws_retry_backoff_sec <= 0:
        raise SystemExit("--aws-retry-backoff-sec must be greater than 0.")

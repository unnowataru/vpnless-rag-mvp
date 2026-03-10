"""Endpoint business logic shared by HTTP transport handlers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from typing import Any

from core.local_retriever import LocalVectorRetriever
from core.qa_flow import run_qa_flow
from core.query_runtime import resolve_retrieval_filters
from core.query_runtime import select_default_answer_profile
from core.retriever_contract import RetrievalHit
from core.retriever_contract import serialize_hit
from core.retriever_contract import validate_filters
from core.retriever_fallback import FallbackRetriever
from core.retrieval_stats import attach_backend_metadata
from core.retrieval_stats import build_retrieval_stats

MAX_JSON_BODY_BYTES = 1_048_576


@dataclass(frozen=True)
class AppContext:
    local_retriever: LocalVectorRetriever
    vast_retriever: FallbackRetriever
    external_retriever: FallbackRetriever
    metadata: list[dict[str, Any]]
    runtime_config: Any
    system_prompt: str
    region: str
    profile: str
    default_topk: int
    default_rerank: bool
    rerank_model: str
    rerank_topn: int
    max_context_chars: int
    max_tokens: int
    aws_timeout_sec: int
    aws_retries: int
    aws_retry_backoff_sec: float
    allow_unscoped_default: bool
    auto_scope_max_docs: int
    audit_log_dir: str | None
    vast_endpoint: str = ""
    external_endpoint: str = ""
    local_fallback_on_retriever_error: bool = True
    cors_allow_origin: str = "*"


@dataclass(frozen=True)
class QaRequestOptions:
    top_k: int
    backend: str
    explicit_filters: dict[str, Any]
    rerank_enabled: bool
    allow_unscoped: bool
    auto_scope_max_docs: int
    answer_profile: str
    bedrock_model_override: str | None
    request_id_seed: str | None


@dataclass(frozen=True)
class ScopeOptions:
    explicit_filters: dict[str, Any]
    allow_unscoped: bool
    auto_scope_max_docs: int


def _json_error(message: str, *, status: int, details: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    body: dict[str, Any] = {"error": message}
    if details:
        body["details"] = details
    return status, body


class BodyTooLargeError(ValueError):
    """Raised when request body exceeds the accepted size limit."""


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw_len = handler.headers.get("Content-Length")
    if raw_len is None:
        raise ValueError("Missing Content-Length header.")
    try:
        length = int(raw_len)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length header.") from exc
    if length < 0:
        raise ValueError("Invalid Content-Length header.")
    if length > MAX_JSON_BODY_BYTES:
        raise BodyTooLargeError(
            f"JSON body must be <= {MAX_JSON_BODY_BYTES} bytes."
        )
    payload = handler.rfile.read(length)
    try:
        loaded = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("JSON body must be an object.")
    return loaded


def _backend_from_request(payload: dict[str, Any]) -> str:
    backend = str(payload.get("retriever_backend", "local")).strip().lower()
    if backend not in {"local", "vast", "external"}:
        raise ValueError("retriever_backend must be one of: local, vast, external.")
    return backend


def _int_from_request(payload: dict[str, Any], key: str, default: int, minimum: int) -> int:
    raw = payload.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer.") from exc
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}.")
    return value


def _bool_from_request(payload: dict[str, Any], key: str, default: bool) -> bool:
    raw = payload.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"{key} must be a boolean.")


def _parse_scope_options(ctx: AppContext, payload: dict[str, Any]) -> ScopeOptions:
    filters_raw = payload.get("filters")
    if filters_raw is None:
        explicit_filters: dict[str, Any] = {}
    else:
        if not isinstance(filters_raw, dict):
            raise ValueError("filters must be an object.")
        explicit_filters = validate_filters(filters_raw)
    allow_unscoped = _bool_from_request(payload, "allow_unscoped", ctx.allow_unscoped_default)
    auto_scope_max_docs = _int_from_request(payload, "auto_scope_max_docs", ctx.auto_scope_max_docs, minimum=1)
    return ScopeOptions(
        explicit_filters=explicit_filters,
        allow_unscoped=allow_unscoped,
        auto_scope_max_docs=auto_scope_max_docs,
    )


def _parse_qa_options(ctx: AppContext, payload: dict[str, Any]) -> QaRequestOptions:
    top_k = _int_from_request(payload, "top_k", ctx.default_topk, minimum=1)
    backend = _backend_from_request(payload)
    scope_options = _parse_scope_options(ctx, payload)
    rerank_enabled = _bool_from_request(payload, "rerank", ctx.default_rerank)
    answer_profile = str(payload.get("answer_profile") or "").strip() or select_default_answer_profile(
        ctx.runtime_config.answer_profiles
    )
    bedrock_model_override = str(payload.get("bedrock_model", "")).strip() or None
    request_id_seed = str(payload.get("request_id", "")).strip() or None
    return QaRequestOptions(
        top_k=top_k,
        backend=backend,
        explicit_filters=scope_options.explicit_filters,
        rerank_enabled=rerank_enabled,
        allow_unscoped=scope_options.allow_unscoped,
        auto_scope_max_docs=scope_options.auto_scope_max_docs,
        answer_profile=answer_profile,
        bedrock_model_override=bedrock_model_override,
        request_id_seed=request_id_seed,
    )


def _run_qa_request(
    ctx: AppContext,
    *,
    question: str,
    options: QaRequestOptions,
):
    def search_fn(query_text: str, top_k_inner: int, filters: dict[str, Any]):
        return _search(
            ctx,
            query_text=query_text,
            top_k=top_k_inner,
            filters=filters,
            backend=options.backend,
        )

    return run_qa_flow(
        question=question,
        request_id_seed=options.request_id_seed,
        metadata=ctx.metadata,
        temporal_rules=ctx.runtime_config.temporal_rules,
        search_fn=search_fn,
        top_k=options.top_k,
        explicit_filters=options.explicit_filters,
        runtime_default_filters=ctx.runtime_config.default_retrieval_filters,
        auto_scope_max_docs=options.auto_scope_max_docs,
        allow_unscoped=options.allow_unscoped,
        rerank_enabled=options.rerank_enabled,
        region=ctx.region,
        profile=ctx.profile,
        rerank_model=ctx.rerank_model,
        rerank_topn=ctx.rerank_topn,
        timeout_sec=ctx.aws_timeout_sec,
        retries=ctx.aws_retries,
        retry_backoff_sec=ctx.aws_retry_backoff_sec,
        max_context_chars=ctx.max_context_chars,
        max_tokens=ctx.max_tokens,
        answer_profile=options.answer_profile,
        bedrock_model_override=options.bedrock_model_override,
        answer_profile_to_model=ctx.runtime_config.answer_profile_to_model,
        system_prompt=ctx.system_prompt,
    )


def _openai_error_body(message: str, *, err_type: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type}}


def _resolve_filters_and_scope(
    ctx: AppContext,
    *,
    query_text: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    scope_options = _parse_scope_options(ctx, payload)
    return resolve_retrieval_filters(
        question=query_text,
        explicit_filters=scope_options.explicit_filters,
        runtime_default_filters=ctx.runtime_config.default_retrieval_filters,
        metadata=ctx.metadata,
        auto_scope_max_docs=scope_options.auto_scope_max_docs,
        allow_unscoped=scope_options.allow_unscoped,
    )


def _search(
    ctx: AppContext,
    *,
    query_text: str,
    top_k: int,
    filters: dict[str, Any],
    backend: str,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    if backend == "local":
        diagnostics = ctx.local_retriever.search_with_diagnostics(
            query_text=query_text,
            top_k=top_k,
            filters=filters,
        )
        stats = attach_backend_metadata(
            dict(diagnostics.stats),
            backend_used="local",
            fallback_triggered=False,
            fallback_error=None,
            fallback_error_type=None,
        )
        return diagnostics.hits, stats

    retriever = ctx.vast_retriever if backend == "vast" else ctx.external_retriever
    result = retriever.search_with_fallback(
        query_text=query_text,
        top_k=top_k,
        filters=filters,
    )
    stats = attach_backend_metadata(
        build_retrieval_stats(
            total_hits_before_filter=len(result.hits),
            total_hits_after_filter=len(result.hits),
            total_hits_after_rerank=len(result.hits),
        ),
        backend_used=result.backend_used,
        fallback_triggered=result.fallback_triggered,
        fallback_error=result.error,
        fallback_error_type=result.error_type,
    )
    return result.hits, stats


def _serialize_hits(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        row = serialize_hit(hit)
        row["rank"] = rank
        rows.append(row)
    return rows


def _backend_status(endpoint: str) -> dict[str, Any]:
    configured = bool(endpoint.strip())
    if configured:
        return {
            "configured": True,
            "status": "configured",
            "reason": None,
        }
    return {
        "configured": False,
        "status": "not_configured",
        "reason": "missing_endpoint",
    }

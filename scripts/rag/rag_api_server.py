#!/usr/bin/env python3
"""HTTP API wrapper for local/vast/external RAG retrieval and QA."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from core.app_config import DEFAULT_PROFILE
from core.app_config import DEFAULT_REGION
from core.app_config import DEFAULT_RERANK_MODEL
from core.app_config import DEFAULT_RUNTIME_CONFIG_FILE
from core.app_config import build_app_config_from_args
from core.app_config import validate_app_config
from core.audit import current_times as core_current_times
from core.audit import make_request_id as core_make_request_id
from core.audit import system_prompt_sha256 as core_system_prompt_sha256
from core.audit import write_audit_log as core_write_audit_log
from core.local_retriever import LocalVectorRetriever
from core.local_retriever import load_manifest
from core.local_retriever import load_metadata
from core.prompt_builder import build_evidence as core_build_evidence
from core.prompt_builder import build_runtime_evidence_block as core_build_runtime_evidence_block
from core.retriever_contract import RetrievalHit
from core.retriever_contract import serialize_hit
from core.retriever_contract import validate_filters
from core.retriever_external import ExternalRetriever
from core.retriever_external import ExternalRetrieverConfig
from core.retriever_fallback import FallbackRetriever
from core.query_runtime import build_rule_based_answer
from core.query_runtime import call_bedrock
from core.query_runtime import load_runtime_config
from core.query_runtime import load_system_prompt
from core.query_runtime import rerank_hits
from core.query_runtime import resolve_bedrock_model
from core.query_runtime import resolve_retrieval_filters
from core.query_runtime import sanitize
from core.query_runtime import select_default_answer_profile
from core.retriever_vast import VastRetriever
from core.retriever_vast import VastRetrieverConfig

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency: sentence-transformers. "
        "Install with: pip install -r scripts/rag/requirements.txt"
    ) from exc


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


def _json_error(message: str, *, status: int, details: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    body: dict[str, Any] = {"error": message}
    if details:
        body["details"] = details
    return status, body


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw_len = handler.headers.get("Content-Length")
    if raw_len is None:
        raise ValueError("Missing Content-Length header.")
    try:
        length = int(raw_len)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length header.") from exc
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


def _resolve_filters_and_scope(
    ctx: AppContext,
    *,
    query_text: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    filters_raw = payload.get("filters")
    explicit_filters: dict[str, Any]
    if filters_raw is None:
        explicit_filters = {}
    else:
        if not isinstance(filters_raw, dict):
            raise ValueError("filters must be an object.")
        explicit_filters = validate_filters(filters_raw)

    allow_unscoped = _bool_from_request(payload, "allow_unscoped", ctx.allow_unscoped_default)
    auto_scope_max_docs = _int_from_request(
        payload,
        "auto_scope_max_docs",
        ctx.auto_scope_max_docs,
        minimum=1,
    )
    return resolve_retrieval_filters(
        question=query_text,
        explicit_filters=explicit_filters,
        runtime_default_filters=ctx.runtime_config.default_retrieval_filters,
        metadata=ctx.metadata,
        auto_scope_max_docs=auto_scope_max_docs,
        allow_unscoped=allow_unscoped,
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
        stats = dict(diagnostics.stats)
        stats["retriever_backend_used"] = "local"
        stats["fallback_triggered"] = False
        stats["fallback_error"] = None
        stats["fallback_error_type"] = None
        return diagnostics.hits, stats

    retriever = ctx.vast_retriever if backend == "vast" else ctx.external_retriever
    result = retriever.search_with_fallback(
        query_text=query_text,
        top_k=top_k,
        filters=filters,
    )
    stats = {
        "hits_before_filter": len(result.hits),
        "hits_after_filter": len(result.hits),
        "hits_after_rerank": len(result.hits),
        "filter_pass_rate": 1.0 if result.hits else 0.0,
        "zero_hit": len(result.hits) == 0,
        "retriever_backend_used": result.backend_used,
        "fallback_triggered": result.fallback_triggered,
        "fallback_error": result.error,
        "fallback_error_type": result.error_type,
    }
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


class RagHTTPServer(ThreadingHTTPServer):
    context: AppContext


class RagRequestHandler(BaseHTTPRequestHandler):
    server: RagHTTPServer

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            ctx = self.server.context
            self._send(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "rag-api",
                    "backends": {
                        "local": {"configured": True, "status": "ready", "reason": None},
                        "vast": _backend_status(ctx.vast_endpoint),
                        "external": _backend_status(ctx.external_endpoint),
                    },
                    "fallback": {
                        "enabled": ctx.local_fallback_on_retriever_error,
                    },
                },
            )
            return
        self._send(*_json_error("Not found.", status=HTTPStatus.NOT_FOUND))

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = _read_json_body(self)
        except ValueError as exc:
            self._send(*_json_error(str(exc), status=HTTPStatus.BAD_REQUEST))
            return

        if self.path == "/search":
            self._handle_search(payload)
            return
        if self.path == "/qa":
            self._handle_qa(payload)
            return
        self._send(*_json_error("Not found.", status=HTTPStatus.NOT_FOUND))

    def _handle_search(self, payload: dict[str, Any]) -> None:
        ctx = self.server.context
        query_text = str(payload.get("query_text", "")).strip()
        if not query_text:
            self._send(*_json_error("query_text is required.", status=HTTPStatus.BAD_REQUEST))
            return

        try:
            top_k = _int_from_request(payload, "top_k", ctx.default_topk, minimum=1)
            backend = _backend_from_request(payload)
            filters, scope_source = _resolve_filters_and_scope(ctx, query_text=query_text, payload=payload)
            hits, retrieval_stats = _search(
                ctx,
                query_text=query_text,
                top_k=top_k,
                filters=filters,
                backend=backend,
            )
        except (ValueError, RuntimeError) as exc:
            status = HTTPStatus.BAD_REQUEST if isinstance(exc, ValueError) else HTTPStatus.BAD_GATEWAY
            self._send(*_json_error(str(exc), status=status))
            return

        response = {
            "query_text": query_text,
            "top_k": top_k,
            "scope_source": scope_source,
            "filters": filters,
            "retrieval_stats": retrieval_stats,
            "hits": _serialize_hits(hits),
        }
        self._send(HTTPStatus.OK, response)

    def _handle_qa(self, payload: dict[str, Any]) -> None:
        ctx = self.server.context
        question = str(payload.get("question") or payload.get("query_text") or "").strip()
        if not question:
            self._send(*_json_error("question is required.", status=HTTPStatus.BAD_REQUEST))
            return

        request_id = core_make_request_id(str(payload.get("request_id", "")).strip() or None)
        now_utc, now_jst = core_current_times()
        try:
            top_k = _int_from_request(payload, "top_k", ctx.default_topk, minimum=1)
            backend = _backend_from_request(payload)
            filters, scope_source = _resolve_filters_and_scope(ctx, query_text=question, payload=payload)
            hits, retrieval_stats = _search(
                ctx,
                query_text=question,
                top_k=top_k,
                filters=filters,
                backend=backend,
            )
        except (ValueError, RuntimeError) as exc:
            status = HTTPStatus.BAD_REQUEST if isinstance(exc, ValueError) else HTTPStatus.BAD_GATEWAY
            self._send(*_json_error(str(exc), status=status))
            return

        rerank_enabled = _bool_from_request(payload, "rerank", ctx.default_rerank)
        if rerank_enabled:
            try:
                hits = rerank_hits(
                    question=question,
                    hits=hits,
                    metadata=ctx.metadata,
                    region=ctx.region,
                    profile=ctx.profile,
                    rerank_model=ctx.rerank_model,
                    rerank_topn=ctx.rerank_topn,
                    timeout_sec=ctx.aws_timeout_sec,
                    retries=ctx.aws_retries,
                    retry_backoff_sec=ctx.aws_retry_backoff_sec,
                )
            except RuntimeError as exc:
                retrieval_stats["rerank_error"] = str(exc)
        retrieval_stats["hits_after_rerank"] = len(hits)

        rule_answer = build_rule_based_answer(
            question=question,
            metadata=ctx.metadata,
            today=now_jst.date(),
            temporal_rules=ctx.runtime_config.temporal_rules,
        )

        vector_start_rank = 3 if rule_answer else 2
        vector_evidence, vector_entries = core_build_evidence(
            hits,
            ctx.max_context_chars,
            start_rank=vector_start_rank,
        )
        runtime_block, runtime_entry = core_build_runtime_evidence_block(
            rank=1,
            request_id=request_id,
            now_utc=now_utc,
            now_jst=now_jst,
        )
        evidence_parts: list[str] = [runtime_block]
        evidence_entries: list[dict[str, Any]] = [runtime_entry]
        if rule_answer:
            evidence_parts.append(
                "[2] score(rule=1.00000) doc=local-temporal-helper page=- chunk=-\n"
                f"{sanitize(rule_answer)}\n"
            )
            evidence_entries.append(
                {
                    "rank": 2,
                    "source_type": "local_rule",
                    "doc": "local-temporal-helper",
                    "page": -1,
                    "chunk": -1,
                }
            )
        if vector_evidence:
            evidence_parts.append(vector_evidence)
            evidence_entries.extend(vector_entries)
        evidence_text = ("\n".join(evidence_parts))[: ctx.max_context_chars]

        has_substantive_evidence = bool(hits) or bool(rule_answer)
        answer_profile = str(payload.get("answer_profile") or "").strip() or select_default_answer_profile(
            ctx.runtime_config.answer_profiles
        )
        bedrock_model_override = str(payload.get("bedrock_model", "")).strip() or None
        try:
            bedrock_model = resolve_bedrock_model(
                answer_profile,
                bedrock_model_override,
                ctx.runtime_config.answer_profile_to_model,
            )
        except SystemExit as exc:
            self._send(*_json_error(str(exc), status=HTTPStatus.BAD_REQUEST))
            return

        status = "success"
        error_message: str | None = None
        if not evidence_text or not has_substantive_evidence:
            status = "insufficient"
            answer = "Evidence is insufficient."
        else:
            try:
                answer = call_bedrock(
                    question=question,
                    evidence=evidence_text,
                    system_prompt=ctx.system_prompt,
                    max_tokens=ctx.max_tokens,
                    region=ctx.region,
                    profile=ctx.profile,
                    model_id=bedrock_model,
                    timeout_sec=ctx.aws_timeout_sec,
                    retries=ctx.aws_retries,
                    retry_backoff_sec=ctx.aws_retry_backoff_sec,
                )
            except RuntimeError as exc:
                status = "failed"
                error_message = str(exc)
                answer = "Evidence is insufficient."

        response = {
            "request_id": request_id,
            "status": status,
            "question": question,
            "answer": answer,
            "error": error_message,
            "answer_profile": answer_profile,
            "model_id": bedrock_model,
            "scope_source": scope_source,
            "filters": filters,
            "retrieval_stats": retrieval_stats,
            "evidence_entries": evidence_entries,
        }

        if ctx.audit_log_dir:
            payload_row = {
                "request_id": request_id,
                "executed_at_utc": now_utc.isoformat(),
                "executed_at_jst": now_jst.isoformat(),
                "region": ctx.region,
                "profile": ctx.profile,
                "model_id": bedrock_model,
                "question": sanitize(question),
                "answer": answer,
                "status": status,
                "error": error_message,
                "system_prompt_sha256": core_system_prompt_sha256(ctx.system_prompt),
                "evidence_entries": evidence_entries,
                "topk": top_k,
                "rerank": rerank_enabled,
                "rerank_model": ctx.rerank_model,
                "rerank_topn": ctx.rerank_topn,
                "filters": filters,
                "scope_source": scope_source,
                "retrieval_stats": retrieval_stats,
            }
            audit_path = core_write_audit_log(ctx.audit_log_dir, request_id, payload_row)
            response["audit_log_path"] = str(audit_path)

        self._send(HTTPStatus.OK, response)


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
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument("--audit-log-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = build_app_config_from_args(args)
    validate_app_config(app_config, require_port=True)

    index_dir = Path(args.index_dir)
    manifest = load_manifest(index_dir)
    metadata = load_metadata(index_dir)
    model = SentenceTransformer(manifest["embedding_model"])

    runtime_config = load_runtime_config(args.runtime_config_file)
    system_prompt = load_system_prompt(args.system_prompt_file, runtime_config.default_system_prompt_file)
    local_retriever = LocalVectorRetriever(
        index_dir=index_dir,
        backend=manifest["backend"],
        metadata=metadata,
        model=model,
        query_prefix=manifest.get("query_prefix", ""),
        snippet_chars=app_config.snippet_max_chars,
    )
    vast_primary = VastRetriever(
        VastRetrieverConfig(
            endpoint=args.vast_endpoint,
            collection=args.vast_collection,
            timeout_sec=app_config.aws_timeout_sec,
        )
    )
    external_primary = ExternalRetriever(
        ExternalRetrieverConfig(
            endpoint=args.external_endpoint,
            provider=args.external_provider,
            timeout_sec=app_config.aws_timeout_sec,
        )
    )
    if args.local_fallback_on_retriever_error:
        vast_retriever = FallbackRetriever(
            primary_name="vast",
            primary=vast_primary,
            fallback_name="local",
            fallback=local_retriever,
        )
        external_retriever = FallbackRetriever(
            primary_name=args.external_provider,
            primary=external_primary,
            fallback_name="local",
            fallback=local_retriever,
        )
    else:
        vast_retriever = FallbackRetriever(
            primary_name="vast",
            primary=vast_primary,
            fallback_name="vast",
            fallback=vast_primary,
        )
        external_retriever = FallbackRetriever(
            primary_name=args.external_provider,
            primary=external_primary,
            fallback_name=args.external_provider,
            fallback=external_primary,
        )

    context = AppContext(
        local_retriever=local_retriever,
        vast_retriever=vast_retriever,
        external_retriever=external_retriever,
        metadata=metadata,
        runtime_config=runtime_config,
        system_prompt=system_prompt,
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
    )

    server_port = app_config.port if app_config.port is not None else args.port
    server = RagHTTPServer((args.host, server_port), RagRequestHandler)
    server.context = context
    print(f"RAG API server listening on http://{args.host}:{server_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

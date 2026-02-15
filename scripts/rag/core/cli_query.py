"""Query execution helpers for rag_vector_cli."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from core.audit import system_prompt_sha256 as core_system_prompt_sha256
from core.audit import write_audit_log as core_write_audit_log
from core.local_retriever import LocalVectorRetriever
from core.qa_flow import run_qa_flow
from core.query_runtime import AnswerProfileConfig
from core.query_runtime import TemporalMilestoneRule
from core.query_runtime import sanitize
from core.retriever_fallback import FallbackRetriever


def prompt_answer_profile(
    default_profile: str,
    answer_profiles: tuple[AnswerProfileConfig, ...],
) -> str:
    index_to_key = {str(i): profile.key for i, profile in enumerate(answer_profiles, start=1)}
    text_to_key = {profile.key.lower(): profile.key for profile in answer_profiles}

    while True:
        labels: list[str] = []
        for i, profile in enumerate(answer_profiles, start=1):
            desc = f" ({profile.description})" if profile.description else ""
            labels.append(f"[{i}] {profile.key}{desc}")
        print("Select answer profile: " + ", ".join(labels))
        options = "/".join([str(i) for i in range(1, len(answer_profiles) + 1)] + [p.key for p in answer_profiles])
        raw = input(f"Mode [{options}, Enter={default_profile}]> ").strip()
        if not raw:
            return default_profile
        selected = index_to_key.get(raw) or text_to_key.get(raw.lower())
        if selected is not None:
            return selected
        print(f"Invalid choice. Please enter one of: {options}.", file=sys.stderr)


def _build_audit_payload(
    *,
    request_id: str,
    now_utc: Any,
    now_jst: Any,
    args: argparse.Namespace,
    model_id: str,
    question: str,
    answer: str | None,
    status: str,
    error: str | None,
    system_prompt: str,
    evidence_entries: list[dict[str, Any]],
    applied_filters: dict[str, Any],
    scope_source: str,
    retrieval_stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "executed_at_utc": now_utc.isoformat(),
        "executed_at_jst": now_jst.isoformat(),
        "region": args.region,
        "profile": args.profile,
        "model_id": model_id,
        "question": sanitize(question),
        "answer": answer,
        "status": status,
        "error": error,
        "system_prompt_file": args.system_prompt_file,
        "system_prompt_sha256": core_system_prompt_sha256(system_prompt),
        "evidence_entries": evidence_entries,
        "index_dir": str(args.index_dir),
        "topk": args.topk,
        "rerank": args.rerank,
        "rerank_model": args.rerank_model,
        "rerank_topn": args.rerank_topn,
        "filters": applied_filters,
        "scope_source": scope_source,
        "retrieval_stats": retrieval_stats,
    }


def run_single_query(
    question: str,
    args: argparse.Namespace,
    metadata: list[dict[str, Any]],
    answer_profile: str,
    system_prompt: str,
    temporal_rules: tuple[TemporalMilestoneRule, ...],
    answer_profile_to_model: dict[str, str],
    local_retriever: LocalVectorRetriever,
    external_fallback_retriever: FallbackRetriever | None,
) -> None:
    def _search(query_text: str, top_k: int, filters: dict[str, Any]):
        if external_fallback_retriever is None:
            diagnostics = local_retriever.search_with_diagnostics(
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

        fallback_result = external_fallback_retriever.search_with_fallback(
            query_text=query_text,
            top_k=top_k,
            filters=filters,
        )
        stats = {
            "hits_before_filter": len(fallback_result.hits),
            "hits_after_filter": len(fallback_result.hits),
            "hits_after_rerank": len(fallback_result.hits),
            "filter_pass_rate": 1.0 if fallback_result.hits else 0.0,
            "zero_hit": len(fallback_result.hits) == 0,
            "retriever_backend_used": fallback_result.backend_used,
            "fallback_triggered": fallback_result.fallback_triggered,
            "fallback_error": fallback_result.error,
            "fallback_error_type": fallback_result.error_type,
        }
        return fallback_result.hits, stats

    try:
        result = run_qa_flow(
            question=question,
            request_id_seed=args.request_id,
            metadata=metadata,
            temporal_rules=temporal_rules,
            search_fn=_search,
            top_k=args.topk,
            explicit_filters=args.explicit_retrieval_filters,
            runtime_default_filters=args.runtime_default_retrieval_filters,
            auto_scope_max_docs=args.auto_scope_max_docs,
            allow_unscoped=args.allow_unscoped,
            rerank_enabled=args.rerank,
            region=args.region,
            profile=args.profile,
            rerank_model=args.rerank_model,
            rerank_topn=args.rerank_topn,
            timeout_sec=args.aws_timeout_sec,
            retries=args.aws_retries,
            retry_backoff_sec=args.aws_retry_backoff_sec,
            max_context_chars=args.max_context_chars,
            max_tokens=args.max_tokens,
            answer_profile=answer_profile,
            bedrock_model_override=args.bedrock_model,
            answer_profile_to_model=answer_profile_to_model,
            system_prompt=system_prompt,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"[INFO] retrieval_scope_source={result.scope_source} filters="
        f"{json.dumps(result.filters, ensure_ascii=False)}",
        file=sys.stderr,
    )
    print(
        "[INFO] retrieval_stats="
        + json.dumps(result.retrieval_stats, ensure_ascii=False),
        file=sys.stderr,
    )
    print("=== TOPK EVIDENCE ===")
    print(result.evidence_text if result.evidence_text else "(no hits)")
    print("=== BEDROCK ANSWER ===")
    if result.status == "failed" and args.fail_on_generation_error:
        raise RuntimeError(result.error or "Bedrock generation failed.")
    print(result.answer)

    if args.audit_log_dir:
        payload = _build_audit_payload(
            request_id=result.request_id,
            now_utc=result.now_utc,
            now_jst=result.now_jst,
            args=args,
            model_id=result.model_id,
            question=question,
            answer=result.answer if result.status != "failed" else None,
            status=result.status,
            error=result.error,
            system_prompt=system_prompt,
            evidence_entries=result.evidence_entries,
            applied_filters=result.filters,
            scope_source=result.scope_source,
            retrieval_stats=result.retrieval_stats,
        )
        audit_path = core_write_audit_log(args.audit_log_dir, result.request_id, payload)
        print(f"[INFO] Audit log written: {audit_path}", file=sys.stderr)

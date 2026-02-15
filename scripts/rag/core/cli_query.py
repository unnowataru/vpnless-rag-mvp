"""Query execution helpers for rag_vector_cli."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from core.audit import current_times as core_current_times
from core.audit import make_request_id as core_make_request_id
from core.audit import system_prompt_sha256 as core_system_prompt_sha256
from core.audit import write_audit_log as core_write_audit_log
from core.local_retriever import LocalVectorRetriever
from core.prompt_builder import build_evidence as core_build_evidence
from core.prompt_builder import build_runtime_evidence_block as core_build_runtime_evidence_block
from core.query_runtime import AnswerProfileConfig
from core.query_runtime import TemporalMilestoneRule
from core.query_runtime import build_rule_based_answer
from core.query_runtime import call_bedrock
from core.query_runtime import rerank_hits
from core.query_runtime import resolve_bedrock_model
from core.query_runtime import resolve_retrieval_filters
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
    request_id = core_make_request_id(args.request_id)
    now_utc, now_jst = core_current_times()
    rule_answer = build_rule_based_answer(
        question=question,
        metadata=metadata,
        today=now_jst.date(),
        temporal_rules=temporal_rules,
    )
    try:
        applied_filters, scope_source = resolve_retrieval_filters(
            question=question,
            explicit_filters=args.explicit_retrieval_filters,
            runtime_default_filters=args.runtime_default_retrieval_filters,
            metadata=metadata,
            auto_scope_max_docs=args.auto_scope_max_docs,
            allow_unscoped=args.allow_unscoped,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"[INFO] retrieval_scope_source={scope_source} filters="
        f"{json.dumps(applied_filters, ensure_ascii=False)}",
        file=sys.stderr,
    )

    if external_fallback_retriever is None:
        diagnostics = local_retriever.search_with_diagnostics(
            query_text=question,
            top_k=args.topk,
            filters=applied_filters,
        )
        contract_hits = diagnostics.hits
        retrieval_stats = dict(diagnostics.stats)
        retrieval_stats["retriever_backend_used"] = "local"
        retrieval_stats["fallback_triggered"] = False
        retrieval_stats["fallback_error"] = None
    else:
        fallback_result = external_fallback_retriever.search_with_fallback(
            query_text=question,
            top_k=args.topk,
            filters=applied_filters,
        )
        contract_hits = fallback_result.hits
        retrieval_stats = {
            "hits_before_filter": len(contract_hits),
            "hits_after_filter": len(contract_hits),
            "hits_after_rerank": len(contract_hits),
            "filter_pass_rate": 1.0 if contract_hits else 0.0,
            "zero_hit": len(contract_hits) == 0,
            "retriever_backend_used": fallback_result.backend_used,
            "fallback_triggered": fallback_result.fallback_triggered,
            "fallback_error": fallback_result.error,
        }
    if args.rerank:
        try:
            contract_hits = rerank_hits(
                question=question,
                hits=contract_hits,
                metadata=metadata,
                region=args.region,
                profile=args.profile,
                rerank_model=args.rerank_model,
                rerank_topn=args.rerank_topn,
                timeout_sec=args.aws_timeout_sec,
                retries=args.aws_retries,
                retry_backoff_sec=args.aws_retry_backoff_sec,
            )
        except RuntimeError as exc:
            print(f"[WARN] Rerank failed. Falling back to vector-only ranking: {exc}", file=sys.stderr)
    retrieval_stats["hits_after_rerank"] = len(contract_hits)
    print(
        "[INFO] retrieval_stats="
        + json.dumps(retrieval_stats, ensure_ascii=False),
        file=sys.stderr,
    )
    vector_start_rank = 3 if rule_answer else 2
    vector_evidence, vector_entries = core_build_evidence(
        contract_hits,
        args.max_context_chars,
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
        rule_rank = 2
        evidence_parts.append(
            f"[{rule_rank}] score(rule=1.00000) doc=local-temporal-helper page=- chunk=-\n"
            f"{sanitize(rule_answer)}\n"
        )
        evidence_entries.append(
            {
                "rank": rule_rank,
                "source_type": "local_rule",
                "doc": "local-temporal-helper",
                "page": -1,
                "chunk": -1,
            }
        )
    if vector_evidence:
        evidence_parts.append(vector_evidence)
        evidence_entries.extend(vector_entries)
    evidence = ("\n".join(evidence_parts))[: args.max_context_chars]

    print("=== TOPK EVIDENCE ===")
    print(evidence if evidence else "(no hits)")
    print("=== BEDROCK ANSWER ===")
    has_substantive_evidence = bool(contract_hits) or bool(rule_answer)
    if not evidence or not has_substantive_evidence:
        answer = "Evidence is insufficient."
        print(answer)
        if args.audit_log_dir:
            payload = _build_audit_payload(
                request_id=request_id,
                now_utc=now_utc,
                now_jst=now_jst,
                args=args,
                model_id=resolve_bedrock_model(answer_profile, args.bedrock_model, answer_profile_to_model),
                question=question,
                answer=answer,
                status="insufficient",
                error=None,
                system_prompt=system_prompt,
                evidence_entries=evidence_entries,
                applied_filters=applied_filters,
                scope_source=scope_source,
                retrieval_stats=retrieval_stats,
            )
            audit_path = core_write_audit_log(args.audit_log_dir, request_id, payload)
            print(f"[INFO] Audit log written: {audit_path}", file=sys.stderr)
        return

    bedrock_model = resolve_bedrock_model(
        answer_profile,
        args.bedrock_model,
        answer_profile_to_model,
    )
    answer = ""
    error_message: str | None = None
    try:
        answer = call_bedrock(
            question=question,
            evidence=evidence,
            system_prompt=system_prompt,
            max_tokens=args.max_tokens,
            region=args.region,
            profile=args.profile,
            model_id=bedrock_model,
            timeout_sec=args.aws_timeout_sec,
            retries=args.aws_retries,
            retry_backoff_sec=args.aws_retry_backoff_sec,
        )
    except RuntimeError as exc:
        error_message = str(exc)
        if args.audit_log_dir:
            payload = _build_audit_payload(
                request_id=request_id,
                now_utc=now_utc,
                now_jst=now_jst,
                args=args,
                model_id=bedrock_model,
                question=question,
                answer=None,
                status="failed",
                error=error_message,
                system_prompt=system_prompt,
                evidence_entries=evidence_entries,
                applied_filters=applied_filters,
                scope_source=scope_source,
                retrieval_stats=retrieval_stats,
            )
            audit_path = core_write_audit_log(args.audit_log_dir, request_id, payload)
            print(f"[INFO] Audit log written: {audit_path}", file=sys.stderr)
        if args.fail_on_generation_error:
            raise
        answer = "Evidence is insufficient."
        print(answer)
        return

    print(answer)
    if args.audit_log_dir:
        payload = _build_audit_payload(
            request_id=request_id,
            now_utc=now_utc,
            now_jst=now_jst,
            args=args,
            model_id=bedrock_model,
            question=question,
            answer=answer,
            status="success",
            error=error_message,
            system_prompt=system_prompt,
            evidence_entries=evidence_entries,
            applied_filters=applied_filters,
            scope_source=scope_source,
            retrieval_stats=retrieval_stats,
        )
        audit_path = core_write_audit_log(args.audit_log_dir, request_id, payload)
        print(f"[INFO] Audit log written: {audit_path}", file=sys.stderr)

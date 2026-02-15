"""Shared QA execution flow used by CLI and API entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import Callable

from core.audit import current_times as core_current_times
from core.audit import make_request_id as core_make_request_id
from core.prompt_builder import build_evidence as core_build_evidence
from core.prompt_builder import build_runtime_evidence_block as core_build_runtime_evidence_block
from core.query_runtime import TemporalMilestoneRule
from core.query_runtime import build_rule_based_answer
from core.query_runtime import call_bedrock
from core.query_runtime import rerank_hits
from core.query_runtime import resolve_bedrock_model
from core.query_runtime import resolve_retrieval_filters
from core.query_runtime import sanitize
from core.retriever_contract import RetrievalHit

SearchFn = Callable[[str, int, dict[str, Any]], tuple[list[RetrievalHit], dict[str, Any]]]


@dataclass(frozen=True)
class QaExecutionResult:
    request_id: str
    now_utc: datetime
    now_jst: datetime
    model_id: str
    status: str
    answer: str
    error: str | None
    scope_source: str
    filters: dict[str, Any]
    retrieval_stats: dict[str, Any]
    evidence_text: str
    evidence_entries: list[dict[str, Any]]
    hits: list[RetrievalHit]


def run_qa_flow(
    *,
    question: str,
    request_id_seed: str | None,
    metadata: list[dict[str, Any]],
    temporal_rules: tuple[TemporalMilestoneRule, ...],
    search_fn: SearchFn,
    top_k: int,
    explicit_filters: dict[str, Any],
    runtime_default_filters: dict[str, Any],
    auto_scope_max_docs: int,
    allow_unscoped: bool,
    rerank_enabled: bool,
    region: str,
    profile: str,
    rerank_model: str,
    rerank_topn: int,
    timeout_sec: int,
    retries: int,
    retry_backoff_sec: float,
    max_context_chars: int,
    max_tokens: int,
    answer_profile: str,
    bedrock_model_override: str | None,
    answer_profile_to_model: dict[str, str],
    system_prompt: str,
) -> QaExecutionResult:
    request_id = core_make_request_id(request_id_seed)
    now_utc, now_jst = core_current_times()

    rule_answer = build_rule_based_answer(
        question=question,
        metadata=metadata,
        today=now_jst.date(),
        temporal_rules=temporal_rules,
    )
    filters, scope_source = resolve_retrieval_filters(
        question=question,
        explicit_filters=explicit_filters,
        runtime_default_filters=runtime_default_filters,
        metadata=metadata,
        auto_scope_max_docs=auto_scope_max_docs,
        allow_unscoped=allow_unscoped,
    )

    hits, retrieval_stats = search_fn(question, top_k, filters)
    if rerank_enabled:
        try:
            hits = rerank_hits(
                question=question,
                hits=hits,
                metadata=metadata,
                region=region,
                profile=profile,
                rerank_model=rerank_model,
                rerank_topn=rerank_topn,
                timeout_sec=timeout_sec,
                retries=retries,
                retry_backoff_sec=retry_backoff_sec,
            )
        except RuntimeError as exc:
            retrieval_stats["rerank_error"] = str(exc)
    retrieval_stats["hits_after_rerank"] = len(hits)

    vector_start_rank = 3 if rule_answer else 2
    vector_evidence, vector_entries = core_build_evidence(
        hits,
        max_context_chars,
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
    evidence_text = ("\n".join(evidence_parts))[:max_context_chars]

    model_id = resolve_bedrock_model(
        answer_profile,
        bedrock_model_override,
        answer_profile_to_model,
    )
    has_substantive_evidence = bool(hits) or bool(rule_answer)
    if not evidence_text or not has_substantive_evidence:
        return QaExecutionResult(
            request_id=request_id,
            now_utc=now_utc,
            now_jst=now_jst,
            model_id=model_id,
            status="insufficient",
            answer="Evidence is insufficient.",
            error=None,
            scope_source=scope_source,
            filters=filters,
            retrieval_stats=retrieval_stats,
            evidence_text=evidence_text,
            evidence_entries=evidence_entries,
            hits=hits,
        )

    try:
        answer = call_bedrock(
            question=question,
            evidence=evidence_text,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            region=region,
            profile=profile,
            model_id=model_id,
            timeout_sec=timeout_sec,
            retries=retries,
            retry_backoff_sec=retry_backoff_sec,
        )
        return QaExecutionResult(
            request_id=request_id,
            now_utc=now_utc,
            now_jst=now_jst,
            model_id=model_id,
            status="success",
            answer=answer,
            error=None,
            scope_source=scope_source,
            filters=filters,
            retrieval_stats=retrieval_stats,
            evidence_text=evidence_text,
            evidence_entries=evidence_entries,
            hits=hits,
        )
    except RuntimeError as exc:
        return QaExecutionResult(
            request_id=request_id,
            now_utc=now_utc,
            now_jst=now_jst,
            model_id=model_id,
            status="failed",
            answer="Evidence is insufficient.",
            error=str(exc),
            scope_source=scope_source,
            filters=filters,
            retrieval_stats=retrieval_stats,
            evidence_text=evidence_text,
            evidence_entries=evidence_entries,
            hits=hits,
        )

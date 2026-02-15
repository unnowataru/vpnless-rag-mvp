#!/usr/bin/env python3
"""Vector-based RAG CLI: local vector search + Bedrock Converse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from core.audit import current_times as core_current_times
from core.audit import make_request_id as core_make_request_id
from core.audit import system_prompt_sha256 as core_system_prompt_sha256
from core.audit import write_audit_log as core_write_audit_log
from core.bedrock_client import run_aws_cli
from core.local_retriever import LocalVectorRetriever
from core.prompt_builder import build_evidence as core_build_evidence
from core.prompt_builder import build_runtime_evidence_block as core_build_runtime_evidence_block
from core.retriever_external import ExternalRetriever
from core.retriever_external import ExternalRetrieverConfig
from core.retriever_fallback import FallbackRetriever
from core.scope_resolver import infer_doc_id_scope_filters
from core.retriever_vast import VastRetriever
from core.retriever_vast import VastRetrieverConfig
from core.retriever_contract import RetrievalHit
from core.retriever_contract import build_hit_from_row
from core.retriever_contract import serialize_hit
from core.retriever_contract import validate_filters

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency: sentence-transformers. "
        "Install with: pip install -r scripts/rag/requirements.txt"
    ) from exc


DEFAULT_REGION = "ap-northeast-1"
DEFAULT_PROFILE = "rag"
DEFAULT_RERANK_MODEL = "amazon.rerank-v1:0"
DEFAULT_RUNTIME_CONFIG_FILE = "scripts/rag/config/runtime_config.json"
JST = timezone(timedelta(hours=9), "JST")
FALLBACK_SYSTEM_PROMPT = (
    "You must answer ONLY using the Evidence blocks.\n"
    "Output in Japanese.\n"
    "Rules:\n"
    "1) Cite evidence by block number like [1], [2]. Every factual claim MUST have a citation.\n"
    "2) Do not assume missing conditions. If multiple interpretations exist, present branches clearly.\n"
    "3) Never stop at only a clarification question. Provide a useful best-effort answer first.\n"
    "4) If key conditions are missing, show case-by-case answers and then optionally ask ONE clarification question.\n"
    "5) Say exactly 'Evidence is insufficient.' only when evidence has no explicit answer.\n"
    "6) Interpret relative dates (e.g., today) using runtime-context evidence if present.\n"
    "7) Do not quote evidence verbatim. Summarize.\n"
)

EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
TEL_RE = re.compile(r"\b\d{2,4}-\d{2,4}-\d{3,4}\b")
JOIN_DATE_RE = re.compile(r"([0-9０-９]{4})\s*年\s*([0-9０-９]{1,2})\s*月\s*([0-9０-９]{1,2})\s*日")
SERVICE_YEARS_WITH_DAYS_RE = re.compile(r"([0-9０-９]{1,2})\s*年\s*([0-9０-９]{1,2})\s*日")
SERVICE_YEARS_WITH_KEYWORD_RE = re.compile(r"(?:勤続|在籍)\s*([0-9０-９]{1,2})\s*年")


@dataclass(frozen=True)
class TemporalMilestoneRule:
    name: str
    question_keywords: tuple[str, ...]
    source_keywords: tuple[str, ...]
    anchor_month: int = 4
    anchor_day: int = 1
    exclusion_note: str | None = None


@dataclass(frozen=True)
class AnswerProfileConfig:
    key: str
    model_id: str
    description: str = ""


@dataclass(frozen=True)
class RuntimeConfig:
    answer_profiles: tuple[AnswerProfileConfig, ...]
    answer_profile_to_model: dict[str, str]
    temporal_rules: tuple[TemporalMilestoneRule, ...]
    default_system_prompt_file: Path | None
    default_retrieval_filters: dict[str, Any]


def sanitize(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = TEL_RE.sub("[TEL]", text)
    return text


def normalize_digits(text: str) -> str:
    table = str.maketrans("０１２３４５６７８９", "0123456789")
    return text.translate(table)


def parse_join_date(question: str) -> date | None:
    normalized = normalize_digits(question)
    match = JOIN_DATE_RE.search(normalized)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def full_years_between(start: date, end: date) -> int:
    years = end.year - start.year
    if (end.month, end.day) < (start.month, start.day):
        years -= 1
    return max(0, years)


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def extract_service_milestones(text: str) -> set[int]:
    years_with_days = {int(y) for y, _days in SERVICE_YEARS_WITH_DAYS_RE.findall(text)}
    if years_with_days:
        return {year for year in years_with_days if 1 <= year <= 40}

    years_with_keyword = {int(y) for y in SERVICE_YEARS_WITH_KEYWORD_RE.findall(text)}
    return {year for year in years_with_keyword if 1 <= year <= 40}


def extract_rule_milestones(
    metadata: list[dict[str, Any]], source_keywords: tuple[str, ...]
) -> list[int]:
    candidates: list[str] = []
    for row in metadata:
        doc = normalize_digits(str(row.get("doc", "")))
        text = normalize_digits(str(row.get("text", "")))
        if contains_any(doc, source_keywords) or contains_any(text, source_keywords):
            candidates.append(text)
    if not candidates:
        return []

    service_years: set[int] = set()
    for text in candidates:
        service_years.update(extract_service_milestones(text))
    return sorted(service_years)


def compute_next_milestone_window(
    joined_on: date,
    milestones: list[int],
    today: date,
    anchor_month: int,
    anchor_day: int,
) -> tuple[int, date, date] | None:
    if not milestones:
        return None
    current_fy_year = (
        today.year if today >= date(today.year, anchor_month, anchor_day) else today.year - 1
    )
    current_anchor = date(current_fy_year, anchor_month, anchor_day)
    current_years = full_years_between(joined_on, current_anchor)

    next_milestone = next((m for m in milestones if m > current_years), None)
    if next_milestone is None:
        return None

    for year in range(current_fy_year, current_fy_year + 80):
        anchor = date(year, anchor_month, anchor_day)
        if full_years_between(joined_on, anchor) >= next_milestone:
            start = anchor
            next_anchor = date(year + 1, anchor_month, anchor_day)
            end = next_anchor - timedelta(days=1)
            return next_milestone, start, end
    return None


def should_apply_temporal_rule(question: str, question_keywords: tuple[str, ...]) -> bool:
    q = normalize_digits(question)
    if not contains_any(q, question_keywords):
        return False
    if not any(token in q for token in ("いつ", "次", "つぎ")):
        return False
    return True


def build_temporal_rule_answer(
    question: str,
    metadata: list[dict[str, Any]],
    today: date,
    rule: TemporalMilestoneRule,
) -> str | None:
    if not should_apply_temporal_rule(question, rule.question_keywords):
        return None
    joined_on = parse_join_date(question)
    if joined_on is None:
        return None

    milestones = extract_rule_milestones(metadata, source_keywords=rule.source_keywords)
    result = compute_next_milestone_window(
        joined_on=joined_on,
        milestones=milestones,
        today=today,
        anchor_month=rule.anchor_month,
        anchor_day=rule.anchor_day,
    )
    if result is None:
        return None

    milestone, window_start, window_end = result
    lines = [
        f"本日（{today.year}年{today.month}月{today.day}日）時点の計算です。\n"
        f"入社日 {joined_on.year}年{joined_on.month}月{joined_on.day}日 の場合、"
        f"次の{rule.name}対象は勤続{milestone}年の年度です。\n"
        f"取得期間の目安は {window_start.year}年{window_start.month}月{window_start.day}日 "
        f"〜 {window_end.year}年{window_end.month}月{window_end.day}日 です。"
    ]
    if rule.exclusion_note:
        lines.append(f"注: {rule.exclusion_note}")
    return "\n".join(lines)


def build_rule_based_answer(
    question: str,
    metadata: list[dict[str, Any]],
    today: date,
    temporal_rules: tuple[TemporalMilestoneRule, ...],
) -> str | None:
    for rule in temporal_rules:
        answer = build_temporal_rule_answer(
            question=question,
            metadata=metadata,
            today=today,
            rule=rule,
        )
        if answer is not None:
            return answer
    return None


def make_request_id(provided: str | None) -> str:
    if provided:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", provided).strip("._-")
        if safe:
            return safe[:128]
    return uuid.uuid4().hex


def current_times() -> tuple[datetime, datetime]:
    now_utc = datetime.now(timezone.utc)
    now_jst = now_utc.astimezone(JST)
    return now_utc, now_jst


def system_prompt_sha256(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def load_runtime_config(path: str) -> RuntimeConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"--runtime-config-file not found: {config_path}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))

    profiles_raw = raw.get("answer_profiles")
    if not isinstance(profiles_raw, list) or not profiles_raw:
        raise SystemExit(f"Invalid answer_profiles in runtime config: {config_path}")

    answer_profiles: list[AnswerProfileConfig] = []
    answer_profile_to_model: dict[str, str] = {}
    for row in profiles_raw:
        if not isinstance(row, dict):
            raise SystemExit(f"Invalid answer_profiles entry in runtime config: {config_path}")
        key = str(row.get("key", "")).strip()
        model_id = str(row.get("model_id", "")).strip()
        if not key or not model_id:
            raise SystemExit(f"answer_profiles entry requires key/model_id: {config_path}")

        env_override = str(row.get("env_override", "")).strip()
        if env_override:
            override_value = os.getenv(env_override, "").strip()
            if override_value:
                model_id = override_value

        description = str(row.get("description", "")).strip()
        if key in answer_profile_to_model:
            raise SystemExit(f"Duplicate answer profile key '{key}' in: {config_path}")
        answer_profiles.append(AnswerProfileConfig(key=key, model_id=model_id, description=description))
        answer_profile_to_model[key] = model_id

    temporal_rules_raw = raw.get("temporal_rules", [])
    if not isinstance(temporal_rules_raw, list):
        raise SystemExit(f"Invalid temporal_rules in runtime config: {config_path}")

    temporal_rules: list[TemporalMilestoneRule] = []
    for row in temporal_rules_raw:
        if not isinstance(row, dict):
            raise SystemExit(f"Invalid temporal_rules entry in runtime config: {config_path}")
        name = str(row.get("name", "")).strip()
        question_keywords = tuple(
            str(item).strip() for item in row.get("question_keywords", []) if str(item).strip()
        )
        source_keywords = tuple(
            str(item).strip() for item in row.get("source_keywords", []) if str(item).strip()
        )
        if not name or not question_keywords or not source_keywords:
            raise SystemExit(f"temporal_rules entry requires name/question_keywords/source_keywords: {config_path}")
        temporal_rules.append(
            TemporalMilestoneRule(
                name=name,
                question_keywords=question_keywords,
                source_keywords=source_keywords,
                anchor_month=int(row.get("anchor_month", 4)),
                anchor_day=int(row.get("anchor_day", 1)),
                exclusion_note=str(row.get("exclusion_note", "")).strip() or None,
            )
        )

    default_prompt_raw = str(raw.get("default_system_prompt_file", "")).strip()
    default_prompt_path: Path | None = None
    if default_prompt_raw:
        candidate = Path(default_prompt_raw).expanduser()
        if not candidate.is_absolute():
            candidate = (config_path.parent / candidate).resolve()
        default_prompt_path = candidate

    default_retrieval_filters_raw = raw.get("default_retrieval_filters")
    if default_retrieval_filters_raw is None:
        default_retrieval_filters: dict[str, Any] = {}
    else:
        if not isinstance(default_retrieval_filters_raw, dict):
            raise SystemExit(f"default_retrieval_filters must be an object: {config_path}")
        try:
            default_retrieval_filters = validate_filters(default_retrieval_filters_raw)
        except ValueError as exc:
            raise SystemExit(f"Invalid default_retrieval_filters in runtime config: {exc}") from exc

    return RuntimeConfig(
        answer_profiles=tuple(answer_profiles),
        answer_profile_to_model=answer_profile_to_model,
        temporal_rules=tuple(temporal_rules),
        default_system_prompt_file=default_prompt_path,
        default_retrieval_filters=default_retrieval_filters,
    )


def load_system_prompt(path: str | None, default_path: Path | None) -> str:
    prompt_path: Path | None = None
    if path is not None:
        prompt_path = Path(path).expanduser().resolve()
    elif default_path is not None:
        prompt_path = default_path
    else:
        return FALLBACK_SYSTEM_PROMPT

    if not prompt_path.exists():
        raise SystemExit(f"--system-prompt-file not found: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"--system-prompt-file is empty: {prompt_path}")
    return text


def load_manifest(index_dir: Path) -> dict[str, Any]:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Build index first with scripts/rag/build_vector_index.py."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_metadata(index_dir: Path) -> list[dict[str, Any]]:
    metadata_path = index_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}.")
    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_filters_json(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--filters-json must be valid JSON: {exc}") from exc
    try:
        return validate_filters(loaded)
    except ValueError as exc:
        raise SystemExit(f"Invalid --filters-json: {exc}") from exc


def resolve_retrieval_filters(
    question: str,
    explicit_filters: dict[str, Any],
    runtime_default_filters: dict[str, Any],
    metadata: list[dict[str, Any]],
    auto_scope_max_docs: int,
    allow_unscoped: bool,
) -> tuple[dict[str, Any], str]:
    if explicit_filters:
        return explicit_filters, "cli_filters_json"

    auto_scope = infer_doc_id_scope_filters(question, metadata, max_docs=auto_scope_max_docs)
    if auto_scope:
        return auto_scope, "auto_doc_id_scope"

    if runtime_default_filters:
        return runtime_default_filters, "runtime_default_retrieval_filters"

    if allow_unscoped:
        return {}, "allow_unscoped"

    raise RuntimeError(
        "Failed to resolve retrieval scope. "
        "Provide --filters-json, configure default_retrieval_filters in runtime config, "
        "or use --allow-unscoped."
    )


def embed_query(model: SentenceTransformer, question: str, query_prefix: str) -> np.ndarray:
    query = f"{query_prefix}{question}" if query_prefix else question
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vec, dtype=np.float32)


def topk_search(
    qvec: np.ndarray, topk: int, index_dir: Path, backend: str
) -> tuple[np.ndarray, np.ndarray]:
    if backend == "faiss":
        if faiss is None:
            raise RuntimeError(
                "faiss index detected but faiss is not installed. "
                "Install with: pip install faiss-cpu"
            )
        index_path = index_dir / "vectors.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing {index_path}.")
        index = faiss.read_index(str(index_path))
        return index.search(qvec, topk)

    vectors_path = index_dir / "vectors.npy"
    if not vectors_path.exists():
        raise FileNotFoundError(f"Missing {vectors_path}.")

    matrix = np.load(vectors_path).astype(np.float32, copy=False)
    scores = matrix @ qvec[0]
    if len(scores) == 0:
        return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)

    topk = min(topk, len(scores))
    candidate_idx = np.argpartition(scores, -topk)[-topk:]
    sorted_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
    return scores[sorted_idx][None, :], sorted_idx[None, :]


def collect_hits(results_scores: np.ndarray, results_ids: np.ndarray) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for score, idx in zip(results_scores[0], results_ids[0]):
        int_idx = int(idx)
        if int_idx < 0:
            continue
        hits.append({"idx": int_idx, "vector_score": float(score)})
    return hits


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    if isinstance(value, tuple):
        return tuple(str(v) for v in value)
    if value is None:
        return ()
    return (str(value),)


def _matches_filter_value(actual: Any, expected: Any) -> bool:
    actual_values = {v.strip() for v in _as_tuple(actual) if v.strip()}
    if isinstance(expected, (list, tuple)):
        expected_values = {str(v).strip() for v in expected if str(v).strip()}
        return bool(actual_values & expected_values)
    text = str(expected).strip()
    if not text:
        return True
    return text in actual_values


def _matches_updated_at(actual: Any, expected: Any) -> bool:
    actual_text = str(actual or "").strip()
    if not actual_text:
        return False
    if isinstance(expected, dict):
        gte = str(expected.get("gte", "")).strip()
        lte = str(expected.get("lte", "")).strip()
        if gte and actual_text < gte:
            return False
        if lte and actual_text > lte:
            return False
        return True
    return _matches_filter_value(actual_text, expected)


def row_matches_filters(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if key == "doc_id":
            actual = row.get("doc_id") or row.get("doc")
        elif key == "label":
            labels = row.get("labels")
            actual = labels if labels else row.get("label")
        elif key == "updated_at":
            if not _matches_updated_at(row.get("updated_at"), expected):
                return False
            continue
        else:
            actual = row.get(key)
        if not _matches_filter_value(actual, expected):
            return False
    return True


def apply_filters_to_hits(
    hits: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    if not filters:
        return hits
    filtered: list[dict[str, Any]] = []
    for hit in hits:
        idx = int(hit["idx"])
        if idx < 0 or idx >= len(metadata):
            continue
        if row_matches_filters(metadata[idx], filters):
            filtered.append(hit)
    return filtered


def build_retrieval_stats(
    total_hits_before_filter: int,
    total_hits_after_filter: int,
    total_hits_after_rerank: int,
) -> dict[str, Any]:
    pass_rate = (
        float(total_hits_after_filter) / float(total_hits_before_filter)
        if total_hits_before_filter > 0
        else 0.0
    )
    return {
        "hits_before_filter": total_hits_before_filter,
        "hits_after_filter": total_hits_after_filter,
        "hits_after_rerank": total_hits_after_rerank,
        "filter_pass_rate": pass_rate,
        "zero_hit": total_hits_after_filter == 0,
    }


def to_contract_hits(
    hits: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    snippet_chars: int,
) -> list[RetrievalHit]:
    contract_hits: list[RetrievalHit] = []
    for hit in hits:
        idx = int(hit["idx"])
        if idx < 0 or idx >= len(metadata):
            continue
        row = metadata[idx]
        vector_score = hit.get("vector_score")
        rerank_score = hit.get("rerank_score")
        effective_score = rerank_score if rerank_score is not None else vector_score
        if effective_score is None:
            effective_score = 0.0
        contract_hits.append(
            build_hit_from_row(
                row=row,
                score=float(effective_score),
                snippet_chars=snippet_chars,
                fallback_index=idx,
                vector_score=float(vector_score) if vector_score is not None else None,
                rerank_score=float(rerank_score) if rerank_score is not None else None,
            )
        )
    return contract_hits


def to_rerank_model_arn(model_id_or_arn: str, region: str) -> str:
    if model_id_or_arn.startswith("arn:"):
        return model_id_or_arn
    return f"arn:aws:bedrock:{region}::foundation-model/{model_id_or_arn}"


def rerank_hits(
    question: str,
    hits: list[RetrievalHit],
    metadata: list[dict[str, Any]],
    region: str,
    profile: str,
    rerank_model: str,
    rerank_topn: int,
    timeout_sec: int,
    retries: int,
    retry_backoff_sec: float,
) -> list[RetrievalHit]:
    if not hits:
        return hits

    model_arn = to_rerank_model_arn(rerank_model, region)
    number_of_results = len(hits) if rerank_topn <= 0 else min(rerank_topn, len(hits))

    sources: list[dict[str, Any]] = []
    for hit in hits:
        idx = hit.metadata_index
        if idx is None or idx < 0 or idx >= len(metadata):
            row = {"text": hit.text_snippet}
        else:
            row = metadata[idx]
        text = sanitize(str(row.get("text", "")).replace("\n", " ")).strip()
        if not text:
            text = "(empty)"
        sources.append(
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {"text": text[:32000]},
                },
            }
        )

    payload = {
        "queries": [{"type": "TEXT", "textQuery": {"text": question}}],
        "rerankingConfiguration": {
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": model_arn},
                "numberOfResults": number_of_results,
            },
        },
        "sources": sources,
    }

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        temp_path = tmp.name

    try:
        cmd = [
            "aws",
            "bedrock-agent-runtime",
            "rerank",
            "--region",
            region,
            "--profile",
            profile,
            "--cli-input-json",
            f"file://{temp_path}",
            "--no-cli-pager",
            "--output",
            "json",
        ]
        result = run_aws_cli(
            cmd,
            timeout_sec=timeout_sec,
            retries=retries,
            retry_backoff_sec=retry_backoff_sec,
        )

        body = json.loads(result.stdout) if result.stdout.strip() else {}
        rerank_results = body.get("results", [])
        if not rerank_results:
            return hits

        ordered: list[RetrievalHit] = []
        seen_source_indexes: set[int] = set()
        for item in rerank_results:
            source_index = item.get("index")
            if not isinstance(source_index, int):
                continue
            if source_index < 0 or source_index >= len(hits):
                continue
            relevance = float(item.get("relevanceScore", 0.0))
            merged = replace(
                hits[source_index],
                score=relevance,
                rerank_score=relevance,
            )
            ordered.append(merged)
            seen_source_indexes.add(source_index)

        for source_index, hit in enumerate(hits):
            if source_index not in seen_source_indexes:
                ordered.append(hit)
        return ordered
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def build_evidence(
    hits: list[RetrievalHit],
    max_context_chars: int,
    start_rank: int = 1,
) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    entries: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=start_rank):
        doc = hit.doc_meta.get("doc")
        page = hit.doc_meta.get("page")
        chunk = hit.doc_meta.get("chunk")
        section = "/".join(hit.section_path) if hit.section_path else "-"
        labels = ",".join(hit.labels) if hit.labels else "-"

        score_str = f"score={hit.score:.5f}"
        if hit.vector_score is not None:
            score_str += f" vector={hit.vector_score:.5f}"
        if hit.rerank_score is not None:
            score_str += f" rerank={hit.rerank_score:.5f}"
        parts.append(
            f"[{rank}] score({score_str}) doc={doc} "
            f"page={page} chunk={chunk} chunk_id={hit.chunk_id} "
            f"section={section} labels={labels}\n"
            f"{sanitize(hit.text_snippet)}\n"
        )
        entry = serialize_hit(hit)
        entry["rank"] = rank
        entry["source_type"] = "vector"
        entries.append(entry)
    return ("\n".join(parts))[:max_context_chars], entries


def build_runtime_evidence_block(
    rank: int, request_id: str, now_utc: datetime, now_jst: datetime
) -> tuple[str, dict[str, Any]]:
    block = (
        f"[{rank}] score(runtime=1.00000) doc=runtime-context page=- chunk=-\n"
        f"request_id={request_id} "
        f"executed_at_utc={now_utc.isoformat()} "
        f"executed_at_jst={now_jst.isoformat()} "
        f"today_jst={now_jst.date().isoformat()} "
        "relative_date_reference=today_jst\n"
    )
    entry = {
        "rank": rank,
        "source_type": "runtime_context",
        "request_id": request_id,
        "executed_at_utc": now_utc.isoformat(),
        "executed_at_jst": now_jst.isoformat(),
        "today_jst": now_jst.date().isoformat(),
    }
    return block, entry


def write_audit_log(log_dir: str, request_id: str, payload: dict[str, Any]) -> Path:
    root = Path(log_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    stamp = payload.get("executed_at_utc", "").replace(":", "").replace("-", "")
    safe_stamp = re.sub(r"[^0-9TZ\.]", "", str(stamp)) or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = root / f"{safe_stamp}_{request_id}.json"
    if out.exists():
        out = root / f"{safe_stamp}_{request_id}_{uuid.uuid4().hex[:8]}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def call_bedrock(
    question: str,
    evidence: str,
    system_prompt: str,
    max_tokens: int,
    region: str,
    profile: str,
    model_id: str,
    timeout_sec: int,
    retries: int,
    retry_backoff_sec: float,
) -> str:
    prompt = f"{system_prompt.rstrip()}\n\nQuestion:\n{question}\n\nEvidence:\n{evidence}"

    payload = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0},
    }

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        temp_path = tmp.name

    try:
        cmd = [
            "aws",
            "bedrock-runtime",
            "converse",
            "--region",
            region,
            "--profile",
            profile,
            "--model-id",
            model_id,
            "--cli-input-json",
            f"file://{temp_path}",
            "--no-cli-pager",
            "--query",
            "output.message.content[0].text",
            "--output",
            "text",
        ]
        result = run_aws_cli(
            cmd,
            timeout_sec=timeout_sec,
            retries=retries,
            retry_backoff_sec=retry_backoff_sec,
        )
        return result.stdout.strip()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def resolve_bedrock_model(
    answer_profile: str,
    bedrock_model: str | None,
    answer_profile_to_model: dict[str, str],
) -> str:
    if bedrock_model:
        return bedrock_model
    model_id = answer_profile_to_model.get(answer_profile)
    if model_id is None:
        raise SystemExit(f"Unsupported --answer-profile: {answer_profile}")
    return model_id


def select_default_answer_profile(answer_profiles: tuple[AnswerProfileConfig, ...]) -> str:
    if not answer_profiles:
        raise SystemExit("No answer profiles configured in runtime config.")
    for profile in answer_profiles:
        if profile.key == "cost":
            return profile.key
    return answer_profiles[0].key


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


def run_single_query(
    question: str,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    metadata: list[dict[str, Any]],
    model: SentenceTransformer,
    query_prefix: str,
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
            payload = {
                "request_id": request_id,
                "executed_at_utc": now_utc.isoformat(),
                "executed_at_jst": now_jst.isoformat(),
                "region": args.region,
                "profile": args.profile,
                "model_id": resolve_bedrock_model(
                    answer_profile,
                    args.bedrock_model,
                    answer_profile_to_model,
                ),
                "question": sanitize(question),
                "answer": answer,
                "status": "insufficient",
                "error": None,
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
            payload = {
                "request_id": request_id,
                "executed_at_utc": now_utc.isoformat(),
                "executed_at_jst": now_jst.isoformat(),
                "region": args.region,
                "profile": args.profile,
                "model_id": bedrock_model,
                "question": sanitize(question),
                "answer": None,
                "status": "failed",
                "error": error_message,
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
            audit_path = core_write_audit_log(args.audit_log_dir, request_id, payload)
            print(f"[INFO] Audit log written: {audit_path}", file=sys.stderr)
        if args.fail_on_generation_error:
            raise
        answer = "Evidence is insufficient."
        print(answer)
        return

    print(answer)
    if args.audit_log_dir:
        payload = {
            "request_id": request_id,
            "executed_at_utc": now_utc.isoformat(),
            "executed_at_jst": now_jst.isoformat(),
            "region": args.region,
            "profile": args.profile,
            "model_id": bedrock_model,
            "question": sanitize(question),
            "answer": answer,
            "status": "success",
            "error": error_message,
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
        audit_path = core_write_audit_log(args.audit_log_dir, request_id, payload)
        print(f"[INFO] Audit log written: {audit_path}", file=sys.stderr)


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


def main() -> None:
    args = parse_args()
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
    args.explicit_retrieval_filters = parse_filters_json(args.filters_json)

    index_dir = Path(args.index_dir)

    manifest = load_manifest(index_dir)
    metadata = load_metadata(index_dir)
    model_name = manifest["embedding_model"]
    query_prefix = manifest.get("query_prefix", "")
    runtime_config = load_runtime_config(args.runtime_config_file)
    args.runtime_default_retrieval_filters = runtime_config.default_retrieval_filters

    model = SentenceTransformer(model_name)
    system_prompt = load_system_prompt(args.system_prompt_file, runtime_config.default_system_prompt_file)
    local_retriever = LocalVectorRetriever(
        index_dir=index_dir,
        backend=manifest["backend"],
        metadata=metadata,
        model=model,
        query_prefix=query_prefix,
        snippet_chars=args.snippet_max_chars,
    )
    external_fallback_retriever: FallbackRetriever | None = None
    if args.retriever_backend == "vast":
        primary = VastRetriever(
            VastRetrieverConfig(
                endpoint=args.vast_endpoint,
                collection=args.vast_collection,
                timeout_sec=args.aws_timeout_sec,
            )
        )
        if args.local_fallback_on_retriever_error:
            external_fallback_retriever = FallbackRetriever(
                primary_name="vast",
                primary=primary,
                fallback_name="local",
                fallback=local_retriever,
            )
        else:
            external_fallback_retriever = FallbackRetriever(
                primary_name="vast",
                primary=primary,
                fallback_name="vast",
                fallback=primary,
            )
    elif args.retriever_backend == "external":
        primary = ExternalRetriever(
            ExternalRetrieverConfig(
                endpoint=args.external_endpoint,
                provider=args.external_provider,
                timeout_sec=args.aws_timeout_sec,
            )
        )
        if args.local_fallback_on_retriever_error:
            external_fallback_retriever = FallbackRetriever(
                primary_name=args.external_provider,
                primary=primary,
                fallback_name="local",
                fallback=local_retriever,
            )
        else:
            external_fallback_retriever = FallbackRetriever(
                primary_name=args.external_provider,
                primary=primary,
                fallback_name=args.external_provider,
                fallback=primary,
            )

    configured_default = select_default_answer_profile(runtime_config.answer_profiles)
    selected_profile = args.answer_profile or configured_default
    if selected_profile not in runtime_config.answer_profile_to_model:
        allowed = ", ".join(runtime_config.answer_profile_to_model.keys())
        raise SystemExit(f"Unsupported --answer-profile: {selected_profile}. Allowed: {allowed}")

    if args.interactive:
        if args.bedrock_model:
            print(f"[INFO] --bedrock-model is set. answer profile prompt is skipped: {args.bedrock_model}")
        else:
            selected_profile = prompt_answer_profile(configured_default, runtime_config.answer_profiles)
            selected_model = resolve_bedrock_model(
                selected_profile,
                None,
                runtime_config.answer_profile_to_model,
            )
            print(f"[INFO] Using answer profile '{selected_profile}' ({selected_model})")
        if args.system_prompt_file:
            print(f"[INFO] Using custom system prompt: {args.system_prompt_file}")
        print("Interactive mode started. Type 'exit' or 'quit' to finish.")
        while True:
            try:
                question = input("Q> ").strip()
            except EOFError:
                print()
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            run_single_query(
                question=question,
                args=args,
                manifest=manifest,
                metadata=metadata,
                model=model,
                query_prefix=query_prefix,
                answer_profile=selected_profile,
                system_prompt=system_prompt,
                temporal_rules=runtime_config.temporal_rules,
                answer_profile_to_model=runtime_config.answer_profile_to_model,
                local_retriever=local_retriever,
                external_fallback_retriever=external_fallback_retriever,
            )
            print()
        return

    question = " ".join(args.question).strip() or input("Q> ").strip()
    if not question:
        raise SystemExit("Question is required.")
    run_single_query(
        question=question,
        args=args,
        manifest=manifest,
        metadata=metadata,
        model=model,
        query_prefix=query_prefix,
        answer_profile=selected_profile,
        system_prompt=system_prompt,
        temporal_rules=runtime_config.temporal_rules,
        answer_profile_to_model=runtime_config.answer_profile_to_model,
        local_retriever=local_retriever,
        external_fallback_retriever=external_fallback_retriever,
    )


if __name__ == "__main__":
    main()

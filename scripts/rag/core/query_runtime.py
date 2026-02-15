"""Shared runtime/query helpers for RAG CLI and API entrypoints."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .bedrock_client import AwsCliBedrockClient
from .retriever_contract import RetrievalHit
from .retriever_contract import validate_filters
from .scope_resolver import infer_doc_id_scope_filters

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
    bedrock_client = AwsCliBedrockClient(
        region=region,
        profile=profile,
        timeout_sec=timeout_sec,
        retries=retries,
        retry_backoff_sec=retry_backoff_sec,
    )
    body = bedrock_client.rerank(payload=payload)
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
    bedrock_client = AwsCliBedrockClient(
        region=region,
        profile=profile,
        timeout_sec=timeout_sec,
        retries=retries,
        retry_backoff_sec=retry_backoff_sec,
    )
    return bedrock_client.converse(model_id=model_id, payload=payload)


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

#!/usr/bin/env python3
"""Vector-based RAG CLI: local vector search + Bedrock Converse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

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

    return RuntimeConfig(
        answer_profiles=tuple(answer_profiles),
        answer_profile_to_model=answer_profile_to_model,
        temporal_rules=tuple(temporal_rules),
        default_system_prompt_file=default_prompt_path,
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


def to_rerank_model_arn(model_id_or_arn: str, region: str) -> str:
    if model_id_or_arn.startswith("arn:"):
        return model_id_or_arn
    return f"arn:aws:bedrock:{region}::foundation-model/{model_id_or_arn}"


def rerank_hits(
    question: str,
    hits: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    region: str,
    profile: str,
    rerank_model: str,
    rerank_topn: int,
) -> list[dict[str, Any]]:
    if not hits:
        return hits

    model_arn = to_rerank_model_arn(rerank_model, region)
    number_of_results = len(hits) if rerank_topn <= 0 else min(rerank_topn, len(hits))

    sources: list[dict[str, Any]] = []
    for hit in hits:
        row = metadata[hit["idx"]]
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
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "AWS rerank call failed.")

        body = json.loads(result.stdout) if result.stdout.strip() else {}
        rerank_results = body.get("results", [])
        if not rerank_results:
            return hits

        ordered: list[dict[str, Any]] = []
        seen_source_indexes: set[int] = set()
        for item in rerank_results:
            source_index = item.get("index")
            if not isinstance(source_index, int):
                continue
            if source_index < 0 or source_index >= len(hits):
                continue
            merged = dict(hits[source_index])
            merged["rerank_score"] = float(item.get("relevanceScore", 0.0))
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
    hits: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    max_context_chars: int,
    start_rank: int = 1,
) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    entries: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=start_rank):
        idx = hit["idx"]
        if idx < 0 or idx >= len(metadata):
            continue
        row = metadata[idx]
        snippet = sanitize(str(row.get("text", "")).replace("\n", " "))[:1200]
        score_str = f"vector={hit['vector_score']:.5f}"
        if "rerank_score" in hit:
            score_str += f" rerank={hit['rerank_score']:.5f}"
        parts.append(
            f"[{rank}] score({score_str}) doc={row.get('doc')} "
            f"page={row.get('page')} chunk={row.get('chunk')}\n{snippet}\n"
        )
        entries.append(
            {
                "rank": rank,
                "source_type": "vector",
                "doc": row.get("doc"),
                "page": row.get("page"),
                "chunk": row.get("chunk"),
                "vector_score": hit.get("vector_score"),
                "rerank_score": hit.get("rerank_score"),
            }
        )
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
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "AWS CLI call failed.")
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
) -> None:
    request_id = make_request_id(args.request_id)
    now_utc, now_jst = current_times()
    rule_answer = build_rule_based_answer(
        question=question,
        metadata=metadata,
        today=now_jst.date(),
        temporal_rules=temporal_rules,
    )

    qvec = embed_query(model, question, query_prefix)

    scores, ids = topk_search(qvec, args.topk, Path(args.index_dir), manifest["backend"])
    hits = collect_hits(scores, ids)
    if args.rerank:
        try:
            hits = rerank_hits(
                question=question,
                hits=hits,
                metadata=metadata,
                region=args.region,
                profile=args.profile,
                rerank_model=args.rerank_model,
                rerank_topn=args.rerank_topn,
            )
        except RuntimeError as exc:
            print(f"[WARN] Rerank failed. Falling back to vector-only ranking: {exc}", file=sys.stderr)
    vector_start_rank = 3 if rule_answer else 2
    vector_evidence, vector_entries = build_evidence(
        hits,
        metadata,
        args.max_context_chars,
        start_rank=vector_start_rank,
    )

    runtime_block, runtime_entry = build_runtime_evidence_block(
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
    has_substantive_evidence = bool(vector_entries) or bool(rule_answer)
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
                "system_prompt_sha256": system_prompt_sha256(system_prompt),
                "evidence_entries": evidence_entries,
                "index_dir": str(args.index_dir),
                "topk": args.topk,
                "rerank": args.rerank,
                "rerank_model": args.rerank_model,
                "rerank_topn": args.rerank_topn,
            }
            audit_path = write_audit_log(args.audit_log_dir, request_id, payload)
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
                "system_prompt_sha256": system_prompt_sha256(system_prompt),
                "evidence_entries": evidence_entries,
                "index_dir": str(args.index_dir),
                "topk": args.topk,
                "rerank": args.rerank,
                "rerank_model": args.rerank_model,
                "rerank_topn": args.rerank_topn,
            }
            audit_path = write_audit_log(args.audit_log_dir, request_id, payload)
            print(f"[INFO] Audit log written: {audit_path}", file=sys.stderr)
        raise

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
            "system_prompt_sha256": system_prompt_sha256(system_prompt),
            "evidence_entries": evidence_entries,
            "index_dir": str(args.index_dir),
            "topk": args.topk,
            "rerank": args.rerank,
            "rerank_model": args.rerank_model,
            "rerank_topn": args.rerank_topn,
        }
        audit_path = write_audit_log(args.audit_log_dir, request_id, payload)
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
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
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

    index_dir = Path(args.index_dir)

    manifest = load_manifest(index_dir)
    metadata = load_metadata(index_dir)
    model_name = manifest["embedding_model"]
    query_prefix = manifest.get("query_prefix", "")
    runtime_config = load_runtime_config(args.runtime_config_file)

    model = SentenceTransformer(model_name)
    system_prompt = load_system_prompt(args.system_prompt_file, runtime_config.default_system_prompt_file)

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
    )


if __name__ == "__main__":
    main()

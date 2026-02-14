#!/usr/bin/env python3
"""Vector-based RAG CLI: local vector search + Bedrock Converse."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
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
ANSWER_PROFILE_TO_MODEL = {
    "cost": "google.gemma-3-4b-it",
    "high": "google.gemma-3-27b-it",
}

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


TEMPORAL_MILESTONE_RULES: tuple[TemporalMilestoneRule, ...] = (
    TemporalMilestoneRule(
        name="リフレッシュ休暇",
        question_keywords=("リフレッシュ休暇", "リフレッシュ"),
        source_keywords=("リフレッシュ休暇制度", "リフレッシュ休暇"),
        anchor_month=4,
        anchor_day=1,
        exclusion_note="過去5年間の休職通算1年以上などの除外条件は、この計算では未反映です。",
    ),
    TemporalMilestoneRule(
        name="永年勤続",
        question_keywords=("永年勤続", "勤続表彰", "慰労金"),
        source_keywords=("永年勤続", "勤続表彰", "慰労金"),
        anchor_month=4,
        anchor_day=1,
    ),
)


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


def build_rule_based_answer(question: str, metadata: list[dict[str, Any]], today: date) -> str | None:
    for rule in TEMPORAL_MILESTONE_RULES:
        answer = build_temporal_rule_answer(
            question=question,
            metadata=metadata,
            today=today,
            rule=rule,
        )
        if answer is not None:
            return answer
    return None


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
) -> str:
    parts: list[str] = []
    for rank, hit in enumerate(hits, start=1):
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
    return ("\n".join(parts))[:max_context_chars]


def call_bedrock(
    question: str,
    evidence: str,
    max_tokens: int,
    region: str,
    profile: str,
    model_id: str,
) -> str:
    prompt = (
        "You must answer ONLY using the Evidence blocks.\n"
        "Output in Japanese.\n"
        "Rules:\n"
        "1) Cite evidence by block number like [1], [2]. Every factual claim MUST have a citation.\n"
        "2) If the question is ambiguous in scope, ask ONE clarification question and "
        "show candidate answers with citations in one line.\n"
        "3) Say exactly 'Evidence is insufficient.' only when evidence has no explicit answer.\n"
        "4) Do not quote evidence verbatim. Summarize.\n\n"
        f"Question:\n{question}\n\nEvidence:\n{evidence}"
    )

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


def resolve_bedrock_model(answer_profile: str, bedrock_model: str | None) -> str:
    if bedrock_model:
        return bedrock_model
    try:
        return ANSWER_PROFILE_TO_MODEL[answer_profile]
    except KeyError as exc:
        raise SystemExit(f"Unsupported --answer-profile: {answer_profile}") from exc


def prompt_answer_profile(default_profile: str) -> str:
    choices = {
        "1": "cost",
        "2": "high",
        "cost": "cost",
        "high": "high",
    }
    while True:
        print("Select answer profile: [1] cost (lower cost), [2] high (higher quality)")
        raw = input(f"Mode [1/2/cost/high, Enter={default_profile}]> ").strip().lower()
        if not raw:
            return default_profile
        selected = choices.get(raw)
        if selected is not None:
            return selected
        print("Invalid choice. Please enter 1, 2, cost, or high.", file=sys.stderr)


def run_single_query(
    question: str,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    metadata: list[dict[str, Any]],
    model: SentenceTransformer,
    query_prefix: str,
    answer_profile: str,
) -> None:
    rule_answer = build_rule_based_answer(question=question, metadata=metadata, today=date.today())
    if rule_answer is not None:
        print("=== RULE-BASED ANSWER ===")
        print(rule_answer)
        return

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
    evidence = build_evidence(hits, metadata, args.max_context_chars)

    print("=== TOPK EVIDENCE ===")
    print(evidence if evidence else "(no hits)")
    print("=== BEDROCK ANSWER ===")
    if not evidence:
        print("Evidence is insufficient.")
        return

    bedrock_model = resolve_bedrock_model(answer_profile, args.bedrock_model)
    answer = call_bedrock(
        question=question,
        evidence=evidence,
        max_tokens=args.max_tokens,
        region=args.region,
        profile=args.profile,
        model_id=bedrock_model,
    )
    print(answer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True, help="Path to vector index directory")
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
        choices=["cost", "high"],
        default="cost",
        help=(
            "Select answer model profile. "
            "cost=google.gemma-3-4b-it, high=google.gemma-3-27b-it."
        ),
    )
    parser.add_argument(
        "--bedrock-model",
        default=None,
        help="Override Bedrock model ID directly. If set, --answer-profile is ignored.",
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

    model = SentenceTransformer(model_name)

    selected_profile = args.answer_profile
    if args.interactive:
        if args.bedrock_model:
            print(f"[INFO] --bedrock-model is set. answer profile prompt is skipped: {args.bedrock_model}")
        else:
            selected_profile = prompt_answer_profile(args.answer_profile)
            selected_model = resolve_bedrock_model(selected_profile, None)
            print(f"[INFO] Using answer profile '{selected_profile}' ({selected_model})")
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
    )


if __name__ == "__main__":
    main()

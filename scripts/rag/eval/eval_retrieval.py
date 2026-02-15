#!/usr/bin/env python3
"""Evaluate retrieval quality against a golden question set."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_SPLIT_RE = re.compile(r"[\s\t\r\n、。,.!！?？/\\()（）【】「」『』:：;；\[\]{}]+")


def tokenize(text: str) -> list[str]:
    raw = str(text).strip().lower()
    if not raw:
        return []
    words = [part for part in TOKEN_SPLIT_RE.split(raw) if part]
    # Add 2-char ngrams to better support Japanese term matching.
    normalized = re.sub(r"\s+", "", raw)
    bigrams = [normalized[i : i + 2] for i in range(len(normalized) - 1)]
    return words + bigrams


def weighted_overlap_score(query_tokens: list[str], text_tokens: list[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    q_count = Counter(query_tokens)
    t_count = Counter(text_tokens)
    overlap = 0.0
    for token, q_freq in q_count.items():
        t_freq = t_count.get(token, 0)
        if t_freq <= 0:
            continue
        token_weight = 1.5 if len(token) >= 3 else 1.0
        overlap += token_weight * min(q_freq, t_freq)
    norm = math.sqrt(sum(v * v for v in q_count.values())) * math.sqrt(sum(v * v for v in t_count.values()))
    if norm <= 0:
        return 0.0
    return overlap / norm


@dataclass(frozen=True)
class ChunkRow:
    doc_id: str
    page: int
    text: str
    labels: tuple[str, ...]


def load_chunks(path: Path) -> list[ChunkRow]:
    rows: list[ChunkRow] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        doc_id = str(obj.get("doc_id") or obj.get("doc") or "").strip()
        if not doc_id:
            raise ValueError(f"Missing doc_id/doc at line {i} in {path}")
        labels_raw = obj.get("labels")
        if isinstance(labels_raw, list):
            labels = tuple(str(v).strip() for v in labels_raw if str(v).strip())
        else:
            label = str(obj.get("label", "")).strip()
            labels = (label,) if label else ()
        rows.append(
            ChunkRow(
                doc_id=doc_id,
                page=int(obj.get("page", -1)),
                text=str(obj.get("text", "")),
                labels=labels,
            )
        )
    if not rows:
        raise ValueError(f"No chunks found in {path}")
    return rows


def load_golden(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not str(obj.get("id", "")).strip():
            raise ValueError(f"golden row {i} missing id")
        if not str(obj.get("question", "")).strip():
            raise ValueError(f"golden row {i} missing question")
        expected_doc_ids = obj.get("expected_doc_ids", [])
        if not isinstance(expected_doc_ids, list) or not expected_doc_ids:
            raise ValueError(f"golden row {i} must have non-empty expected_doc_ids")
        rows.append(obj)
    if not rows:
        raise ValueError(f"No golden rows found in {path}")
    return rows


def filter_chunk(row: ChunkRow, filters: dict[str, Any]) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if key == "doc_id":
            expected_values = expected if isinstance(expected, list) else [expected]
            if row.doc_id not in {str(v) for v in expected_values}:
                return False
            continue
        if key in {"label", "labels"}:
            expected_values = expected if isinstance(expected, list) else [expected]
            expected_set = {str(v).strip() for v in expected_values if str(v).strip()}
            if not expected_set:
                continue
            if not (set(row.labels) & expected_set):
                return False
            continue
    return True


def search_chunks(
    question: str,
    chunks: list[ChunkRow],
    *,
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[tuple[float, ChunkRow]]:
    active_filters = filters or {}
    query_tokens = tokenize(question)
    scored: list[tuple[float, ChunkRow]] = []
    for row in chunks:
        if not filter_chunk(row, active_filters):
            continue
        score = weighted_overlap_score(query_tokens, tokenize(row.text + " " + row.doc_id))
        if score <= 0:
            continue
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def dcg(relevances: list[int]) -> float:
    value = 0.0
    for idx, rel in enumerate(relevances, start=1):
        if rel <= 0:
            continue
        value += (2**rel - 1) / math.log2(idx + 1)
    return value


def evaluate(
    chunks: list[ChunkRow],
    golden_rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    recalls: list[float] = []
    rr_values: list[float] = []
    ndcg_values: list[float] = []
    failures: list[dict[str, Any]] = []

    for row in golden_rows:
        qid = str(row["id"])
        question = str(row["question"])
        expected_doc_ids = {str(v) for v in row["expected_doc_ids"]}
        filters = row.get("filters", {})
        if not isinstance(filters, dict):
            filters = {}
        hits = search_chunks(question, chunks, top_k=top_k, filters=filters)
        ranked_doc_ids = [hit.doc_id for _score, hit in hits]
        relevances = [1 if doc_id in expected_doc_ids else 0 for doc_id in ranked_doc_ids]
        expected_count = len(expected_doc_ids)
        found = len(set(ranked_doc_ids) & expected_doc_ids)
        recall = float(found) / float(expected_count) if expected_count > 0 else 0.0
        recalls.append(recall)

        rr = 0.0
        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id in expected_doc_ids:
                rr = 1.0 / float(rank)
                break
        rr_values.append(rr)

        ideal_rels = sorted(relevances, reverse=True)
        actual_dcg = dcg(relevances)
        ideal_dcg = dcg(ideal_rels)
        ndcg_values.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

        if recall <= 0.0:
            failures.append(
                {
                    "id": qid,
                    "question": question,
                    "expected_doc_ids": sorted(expected_doc_ids),
                    "ranked_doc_ids": ranked_doc_ids,
                    "filters": filters,
                }
            )

    chunk_lengths = [len(row.text) for row in chunks]
    unique_text_count = len({row.text for row in chunks})
    duplicate_rate = 1.0 - (unique_text_count / float(len(chunks)))

    return {
        "num_questions": len(golden_rows),
        "num_chunks": len(chunks),
        "retrieval": {
            "Recall@K": sum(recalls) / float(len(recalls)),
            "MRR": sum(rr_values) / float(len(rr_values)),
            "NDCG@K": sum(ndcg_values) / float(len(ndcg_values)),
        },
        "chunk": {
            "avg_len": sum(chunk_lengths) / float(len(chunk_lengths)),
            "var_len": (
                sum((length - (sum(chunk_lengths) / float(len(chunk_lengths)))) ** 2 for length in chunk_lengths)
                / float(len(chunk_lengths))
            ),
            "duplicate_rate": duplicate_rate,
        },
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="chunks/metadata JSONL path")
    parser.add_argument("--golden", required=True, help="golden questions JSONL path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=0.60)
    parser.add_argument("--min-mrr", type=float, default=0.40)
    parser.add_argument("--min-ndcg", type=float, default=0.50)
    parser.add_argument("--max-duplicate-rate", type=float, default=0.50)
    parser.add_argument("--out", default=None, help="optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")

    chunks = load_chunks(Path(args.chunks))
    golden = load_golden(Path(args.golden))
    report = evaluate(chunks, golden, top_k=args.top_k)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))

    retrieval = report["retrieval"]
    chunk = report["chunk"]
    gate_errors: list[str] = []
    if retrieval["Recall@K"] < args.min_recall:
        gate_errors.append(f"Recall@K {retrieval['Recall@K']:.4f} < {args.min_recall:.4f}")
    if retrieval["MRR"] < args.min_mrr:
        gate_errors.append(f"MRR {retrieval['MRR']:.4f} < {args.min_mrr:.4f}")
    if retrieval["NDCG@K"] < args.min_ndcg:
        gate_errors.append(f"NDCG@K {retrieval['NDCG@K']:.4f} < {args.min_ndcg:.4f}")
    if chunk["duplicate_rate"] > args.max_duplicate_rate:
        gate_errors.append(
            f"duplicate_rate {chunk['duplicate_rate']:.4f} > {args.max_duplicate_rate:.4f}"
        )

    if gate_errors:
        raise SystemExit("retrieval regression gate failed: " + "; ".join(gate_errors))


if __name__ == "__main__":
    main()

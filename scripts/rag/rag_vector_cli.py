#!/usr/bin/env python3
"""Vector-based RAG CLI: local vector search + Bedrock Converse."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
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


DEFAULT_BEDROCK_MODEL = "google.gemma-3-4b-it"
DEFAULT_REGION = "ap-northeast-1"
DEFAULT_PROFILE = "rag"

EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
TEL_RE = re.compile(r"\b\d{2,4}-\d{2,4}-\d{3,4}\b")


def sanitize(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = TEL_RE.sub("[TEL]", text)
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


def build_evidence(
    results_scores: np.ndarray,
    results_ids: np.ndarray,
    metadata: list[dict[str, Any]],
    max_context_chars: int,
) -> str:
    parts: list[str] = []
    for rank, (score, idx) in enumerate(zip(results_scores[0], results_ids[0]), start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        row = metadata[idx]
        snippet = sanitize(str(row.get("text", "")).replace("\n", " "))[:1200]
        parts.append(
            f"[{rank}] score={float(score):.5f} doc={row.get('doc')} "
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True, help="Path to vector index directory")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--bedrock-model", default=DEFAULT_BEDROCK_MODEL)
    parser.add_argument("question", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise SystemExit("--topk must be greater than 0.")

    index_dir = Path(args.index_dir)

    question = " ".join(args.question).strip() or input("Q> ").strip()
    if not question:
        raise SystemExit("Question is required.")

    manifest = load_manifest(index_dir)
    metadata = load_metadata(index_dir)
    model_name = manifest["embedding_model"]
    query_prefix = manifest.get("query_prefix", "")

    model = SentenceTransformer(model_name)
    qvec = embed_query(model, question, query_prefix)

    scores, ids = topk_search(qvec, args.topk, index_dir, manifest["backend"])
    evidence = build_evidence(scores, ids, metadata, args.max_context_chars)

    print("=== TOPK EVIDENCE ===")
    print(evidence if evidence else "(no hits)")
    print("=== BEDROCK ANSWER ===")
    if not evidence:
        print("Evidence is insufficient.")
        return

    answer = call_bedrock(
        question=question,
        evidence=evidence,
        max_tokens=args.max_tokens,
        region=args.region,
        profile=args.profile,
        model_id=args.bedrock_model,
    )
    print(answer)


if __name__ == "__main__":
    main()

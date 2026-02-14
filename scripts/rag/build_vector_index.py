#!/usr/bin/env python3
"""Build local vector index from chunks.jsonl."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def detect_prefix(model_name: str) -> tuple[str, str]:
    lowered = model_name.lower()
    if "e5" in lowered:
        return "query: ", "passage: "
    return "", ""


def read_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i}: {exc}") from exc
            text = str(obj.get("text", "")).strip()
            if not text:
                continue
            obj["text"] = text
            rows.append(obj)
    if not rows:
        raise ValueError(f"No valid chunk text found in {path}")
    return rows


def write_metadata(rows: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--index-dir", required=True, help="Directory to write index artifacts")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--backend",
        choices=["faiss", "numpy"],
        default="faiss",
        help="faiss is faster; numpy is fallback without faiss package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0.")

    chunks_path = Path(args.chunks)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    rows = read_chunks(chunks_path)
    query_prefix, passage_prefix = detect_prefix(args.embedding_model)
    passages = [f"{passage_prefix}{row['text']}" if passage_prefix else row["text"] for row in rows]

    model = SentenceTransformer(args.embedding_model)
    vectors = model.encode(
        passages,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    backend = args.backend
    if backend == "faiss":
        if faiss is None:
            raise SystemExit("faiss backend selected but faiss is not installed. Use --backend numpy.")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(index_dir / "vectors.faiss"))
    else:
        np.save(index_dir / "vectors.npy", vectors)

    write_metadata(rows, index_dir / "metadata.jsonl")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunks_path": str(chunks_path.resolve()),
        "num_chunks": len(rows),
        "dim": int(vectors.shape[1]),
        "backend": backend,
        "embedding_model": args.embedding_model,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Indexed {len(rows)} chunks into {index_dir}")
    print(f"Backend: {backend} / Model: {args.embedding_model}")


if __name__ == "__main__":
    main()

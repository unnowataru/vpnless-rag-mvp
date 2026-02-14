#!/usr/bin/env python3
"""Extract text from PDFs and write chunks.jsonl for vector indexing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency: pypdf. Install with: pip install -r scripts/rag/requirements.txt"
    ) from exc


SENTENCE_BREAKS = "\n。！？.!?"


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_chunks(text: str, chunk_size: int, overlap: int, min_chars: int) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    n = len(clean)
    while start < n:
        hard_end = min(start + chunk_size, n)
        end = hard_end

        if hard_end < n:
            window_start = start + int(chunk_size * 0.6)
            window = clean[window_start:hard_end]
            last_break = -1
            for c in SENTENCE_BREAKS:
                pos = window.rfind(c)
                if pos > last_break:
                    last_break = pos
            if last_break >= 0:
                end = window_start + last_break + 1

        piece = clean[start:end].strip()
        if len(piece) >= min_chars:
            chunks.append(piece)

        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        start = next_start

    return chunks


def iter_pdfs(pdf_dir: Path, pattern: str) -> list[Path]:
    files = sorted(p for p in pdf_dir.glob(pattern) if p.is_file())
    return [p for p in files if p.suffix.lower() == ".pdf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True, help="Directory containing source PDFs")
    parser.add_argument("--out", required=True, help="Output chunks.jsonl path")
    parser.add_argument("--glob", default="*.pdf", help="Glob pattern under --pdf-dir")
    parser.add_argument("--chunk-size", type=int, default=900, help="Characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Overlap characters")
    parser.add_argument("--min-chars", type=int, default=80, help="Drop very short chunks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than 0.")
    if args.chunk_overlap < 0:
        raise SystemExit("--chunk-overlap must be 0 or greater.")
    if args.chunk_overlap >= args.chunk_size:
        raise SystemExit("--chunk-overlap must be smaller than --chunk-size.")
    if args.min_chars <= 0:
        raise SystemExit("--min-chars must be greater than 0.")

    pdf_dir = Path(args.pdf_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")

    pdf_files = iter_pdfs(pdf_dir, args.glob)
    if not pdf_files:
        raise SystemExit(f"No PDFs found under {pdf_dir} with pattern {args.glob}")

    total_chunks = 0
    total_pages = 0
    with out_path.open("w", encoding="utf-8") as out:
        for pdf_path in pdf_files:
            reader = PdfReader(str(pdf_path))
            doc_chunk_idx = 0
            for page_idx, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                chunks = split_chunks(
                    raw,
                    chunk_size=args.chunk_size,
                    overlap=args.chunk_overlap,
                    min_chars=args.min_chars,
                )
                if chunks:
                    total_pages += 1
                for text in chunks:
                    row = {
                        "doc": pdf_path.name,
                        "page": page_idx,
                        "chunk": doc_chunk_idx,
                        "text": text,
                    }
                    out.write(json.dumps(row, ensure_ascii=False))
                    out.write("\n")
                    doc_chunk_idx += 1
                    total_chunks += 1

    if total_chunks == 0:
        raise SystemExit("No text chunks were extracted. PDFs may be image-only.")

    print(f"PDF files: {len(pdf_files)}")
    print(f"Pages with text: {total_pages}")
    print(f"Chunks written: {total_chunks}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract text from PDFs and write chunks.jsonl for vector indexing."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from core.chunk_normalizer import normalize_text
from core.chunk_quality import detect_repeated_margin_lines
from core.chunk_quality import remove_margin_noise
from core.chunk_quality import select_best_candidate
from core.chunker import infer_chunk_type
from core.chunker import split_chunks
from core.ingest_metadata import load_metadata_rules
from core.ingest_metadata import missing_required_metadata
from core.ingest_metadata import parse_default_metadata_json
from core.ingest_metadata import parse_required_metadata_fields
from core.ingest_metadata import resolve_document_metadata
from core.pdf_extractor import extract_page_candidates
from core.pdf_extractor import iter_pdfs

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency: pypdf. Install with: pip install -r scripts/rag/requirements.txt"
    ) from exc

try:
    import fitz  # type: ignore
except ImportError:
    fitz = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True, help="Directory containing source PDFs")
    parser.add_argument("--out", required=True, help="Output chunks.jsonl path")
    parser.add_argument("--glob", default="*.pdf", help="Glob pattern under --pdf-dir")
    parser.add_argument("--chunk-size", type=int, default=900, help="Characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Overlap characters")
    parser.add_argument("--min-chars", type=int, default=80, help="Drop very short chunks")
    parser.add_argument(
        "--scan-suspected-char-threshold",
        type=int,
        default=40,
        help="Mark page as scan-suspected when extracted chars are below this threshold.",
    )
    parser.add_argument(
        "--margin-lines",
        type=int,
        default=2,
        help="Top/bottom line count per page used for repeated header/footer removal.",
    )
    parser.add_argument(
        "--margin-threshold-ratio",
        type=float,
        default=0.6,
        help="Frequency ratio threshold to classify repeated margin lines as noise.",
    )
    parser.add_argument(
        "--metadata-rules-file",
        default=None,
        help="Optional JSON file with per-document metadata rules (regex pattern based).",
    )
    parser.add_argument(
        "--default-metadata-json",
        default=None,
        help=(
            "Optional default metadata JSON object. Supported keys: "
            "doc_id,dept,label,labels,updated_at,confidentiality,customer,product,doc_type,retention."
        ),
    )
    parser.add_argument(
        "--updated-at-source",
        choices=["mtime", "none"],
        default="mtime",
        help="How to populate updated_at when not set by defaults/rules.",
    )
    parser.add_argument(
        "--required-metadata-fields",
        default="doc_id,dept,labels,updated_at,confidentiality",
        help=(
            "Comma-separated required metadata fields. "
            "If any required field is missing for a document, ingestion fails."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than 0.")
    if args.chunk_overlap < 0:
        raise SystemExit("--chunk-overlap must be 0 or greater.")
    if args.chunk_overlap >= args.chunk_size:
        raise SystemExit("--chunk-overlap must be smaller than --chunk-size.")
    if args.min_chars <= 0:
        raise SystemExit("--min-chars must be greater than 0.")
    if args.scan_suspected_char_threshold < 0:
        raise SystemExit("--scan-suspected-char-threshold must be 0 or greater.")
    if args.margin_lines <= 0:
        raise SystemExit("--margin-lines must be greater than 0.")
    if args.margin_threshold_ratio <= 0 or args.margin_threshold_ratio > 1:
        raise SystemExit("--margin-threshold-ratio must be within (0, 1].")


def main() -> None:
    args = parse_args()
    validate_args(args)

    pdf_dir = Path(args.pdf_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")

    pdf_files = iter_pdfs(pdf_dir, args.glob)
    if not pdf_files:
        raise SystemExit(f"No PDFs found under {pdf_dir} with pattern {args.glob}")

    try:
        default_metadata = parse_default_metadata_json(args.default_metadata_json)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        metadata_rules = load_metadata_rules(args.metadata_rules_file)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    try:
        required_metadata_fields = parse_required_metadata_fields(args.required_metadata_fields)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    include_updated_at_from_mtime = args.updated_at_source == "mtime"

    total_chunks = 0
    total_pages = 0
    scan_suspected_pages = 0
    extractor_counter: Counter[str] = Counter()
    with out_path.open("w", encoding="utf-8") as out:
        for pdf_path in pdf_files:
            reader = PdfReader(str(pdf_path))
            pymupdf_doc = fitz.open(str(pdf_path)) if fitz is not None else None
            doc_metadata = resolve_document_metadata(
                pdf_path,
                pdf_root=pdf_dir,
                default_metadata=default_metadata,
                metadata_rules=metadata_rules,
                include_updated_at_from_mtime=include_updated_at_from_mtime,
            )
            missing = missing_required_metadata(doc_metadata, required_metadata_fields)
            if missing:
                raise SystemExit(
                    "Missing required metadata for "
                    f"{pdf_path.name}: {', '.join(missing)}. "
                    "Update metadata rules/default metadata before indexing."
                )
            page_candidates = []
            for page_zero_idx in range(len(reader.pages)):
                candidates = extract_page_candidates(
                    page_idx=page_zero_idx,
                    pypdf_reader=reader,
                    pymupdf_doc=pymupdf_doc,
                )
                page_candidates.append(select_best_candidate(candidates))

            repeated_margin_lines = detect_repeated_margin_lines(
                [candidate.text for candidate in page_candidates],
                max_lines=args.margin_lines,
                threshold_ratio=args.margin_threshold_ratio,
            )

            doc_chunk_idx = 0
            for page_idx, selected in enumerate(page_candidates, start=1):
                extractor_counter[selected.engine] += 1
                raw = remove_margin_noise(selected.text, repeated_margin_lines)
                raw = normalize_text(raw)
                scan_suspected = len(raw) < args.scan_suspected_char_threshold
                if scan_suspected:
                    scan_suspected_pages += 1
                chunks = split_chunks(
                    raw,
                    chunk_size=args.chunk_size,
                    overlap=args.chunk_overlap,
                    min_chars=args.min_chars,
                )
                if chunks:
                    total_pages += 1
                for chunk_piece in chunks:
                    chunk_text = str(chunk_piece["text"])
                    row = {
                        "doc": pdf_path.name,
                        "doc_id": doc_metadata.get("doc_id", pdf_path.name),
                        "page": page_idx,
                        "chunk": doc_chunk_idx,
                        "text": chunk_text,
                        "start_offset": int(chunk_piece.get("start_offset", 0)),
                        "section_path": chunk_piece.get("section_path", []),
                        "extract_engine": selected.engine,
                        "extract_score": float(selected.score),
                        "scan_suspected": scan_suspected,
                        "ocr": False,
                        "chunk_type": infer_chunk_type(chunk_text),
                    }
                    for key, value in doc_metadata.items():
                        if key == "doc_id":
                            continue
                        row[key] = value
                    out.write(json.dumps(row, ensure_ascii=False))
                    out.write("\n")
                    doc_chunk_idx += 1
                    total_chunks += 1
            if pymupdf_doc is not None:
                pymupdf_doc.close()

    if total_chunks == 0:
        raise SystemExit("No text chunks were extracted. PDFs may be image-only.")

    print(f"PDF files: {len(pdf_files)}")
    print(f"Pages with text: {total_pages}")
    print(f"Scan-suspected pages: {scan_suspected_pages}")
    if extractor_counter:
        print("Extractor usage: " + ", ".join(f"{k}={v}" for k, v in sorted(extractor_counter.items())))
    print(f"Chunks written: {total_chunks}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()

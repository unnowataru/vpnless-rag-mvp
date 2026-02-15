#!/usr/bin/env python3
"""Extract text from PDFs and write chunks.jsonl for vector indexing."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from core.ingest_metadata import load_metadata_rules
from core.ingest_metadata import missing_required_metadata
from core.ingest_metadata import parse_default_metadata_json
from core.ingest_metadata import parse_required_metadata_fields
from core.ingest_metadata import resolve_document_metadata

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


SENTENCE_BREAKS = "\n。！？.!?"
HEADING_RE = re.compile(r"^(第[\d一二三四五六七八九十]+[章条]|[0-9]+[.)]|[A-Za-z]-[0-9]{2}-[0-9]{2})")


@dataclass(frozen=True)
class PageExtractionCandidate:
    engine: str
    text: str
    score: float
    metrics: dict[str, float]


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # dehyphenate wrapped English words
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def merge_wrapped_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    merged: list[str] = []
    for line in lines:
        if not line:
            if merged and merged[-1] != "":
                merged.append("")
            continue
        if not merged or merged[-1] == "":
            merged.append(line)
            continue
        prev = merged[-1]
        if prev.endswith(tuple(SENTENCE_BREAKS)) or HEADING_RE.match(line):
            merged.append(line)
            continue
        merged[-1] = f"{prev} {line}".strip()
    return "\n".join(merged)


def paragraph_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def infer_chunk_type(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return "text"
    pipe_count = sum(line.count("|") for line in lines)
    tab_count = sum(line.count("\t") for line in lines)
    digit_heavy = sum(1 for line in lines if re.search(r"\d", line))
    code_like = sum(1 for line in lines if re.search(r"[{}();=<>\[\]]", line))
    if pipe_count >= max(4, len(lines)) or tab_count >= max(2, len(lines) // 2):
        return "table"
    if code_like >= max(3, len(lines) // 2):
        return "code_or_log"
    if digit_heavy >= max(3, len(lines) // 2) and len(lines) >= 3:
        return "table_like"
    return "text"


def compute_text_quality_metrics(text: str) -> dict[str, float]:
    total_chars = float(len(text))
    if total_chars <= 0:
        return {
            "char_count": 0.0,
            "printable_ratio": 0.0,
            "blank_line_ratio": 1.0,
            "duplicate_line_ratio": 1.0,
            "score": 0.0,
        }
    lines = [line for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line.strip()]
    printable = sum(1 for ch in text if ch.isprintable() and ch not in {"\x0c", "\x0b"})
    printable_ratio = printable / total_chars
    blank_line_ratio = (
        (len(lines) - len(non_empty_lines)) / float(len(lines))
        if lines
        else 1.0
    )
    duplicate_line_ratio = (
        1.0 - (len(set(non_empty_lines)) / float(len(non_empty_lines)))
        if non_empty_lines
        else 1.0
    )
    score = (
        min(total_chars / 1200.0, 1.0) * 0.45
        + printable_ratio * 0.35
        + (1.0 - min(blank_line_ratio, 0.8)) * 0.1
        + (1.0 - min(duplicate_line_ratio, 0.8)) * 0.1
    )
    return {
        "char_count": total_chars,
        "printable_ratio": printable_ratio,
        "blank_line_ratio": blank_line_ratio,
        "duplicate_line_ratio": duplicate_line_ratio,
        "score": score,
    }


def split_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chars: int,
    *,
    default_section: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    clean = normalize_text(text)
    if not clean:
        return []

    blocks = paragraph_blocks(merge_wrapped_lines(clean))
    chunks: list[dict[str, object]] = []
    current: list[str] = []
    current_len = 0
    cursor = 0
    chunk_start = 0
    section_path = list(default_section)

    for block in blocks:
        heading_candidate = block.splitlines()[0].strip()
        if HEADING_RE.match(heading_candidate) and len(heading_candidate) <= 80:
            section_path = [heading_candidate]

        block_len = len(block)
        if current and current_len + block_len + 2 > chunk_size:
            text_piece = "\n\n".join(current).strip()
            if len(text_piece) >= min_chars:
                chunks.append(
                    {
                        "text": text_piece,
                        "start_offset": chunk_start,
                        "section_path": list(section_path),
                    }
                )
            # retain tail overlap characters as next context seed
            if overlap > 0 and text_piece:
                tail = text_piece[-overlap:].strip()
                current = [tail] if tail else []
                current_len = len(tail)
                chunk_start = max(0, cursor - len(tail))
            else:
                current = []
                current_len = 0
                chunk_start = cursor

        if not current:
            chunk_start = cursor
        current.append(block)
        current_len += block_len + 2
        cursor += block_len + 2

    if current:
        text_piece = "\n\n".join(current).strip()
        if len(text_piece) >= min_chars:
            chunks.append(
                {
                    "text": text_piece,
                    "start_offset": chunk_start,
                    "section_path": list(section_path),
                }
            )

    return chunks


def iter_pdfs(pdf_dir: Path, pattern: str) -> list[Path]:
    files = sorted(p for p in pdf_dir.glob(pattern) if p.is_file())
    return [p for p in files if p.suffix.lower() == ".pdf"]


def extract_page_candidates(
    pdf_path: Path,
    page_idx: int,
    pypdf_reader: PdfReader,
    pymupdf_doc: "fitz.Document | None",
) -> list[PageExtractionCandidate]:
    candidates: list[PageExtractionCandidate] = []
    pypdf_text = pypdf_reader.pages[page_idx].extract_text() or ""
    metrics = compute_text_quality_metrics(pypdf_text)
    candidates.append(
        PageExtractionCandidate(
            engine="pypdf",
            text=pypdf_text,
            score=float(metrics["score"]),
            metrics=metrics,
        )
    )
    if pymupdf_doc is not None and page_idx < pymupdf_doc.page_count:
        page = pymupdf_doc.load_page(page_idx)
        pymupdf_text = page.get_text("text") or ""
        pymupdf_metrics = compute_text_quality_metrics(pymupdf_text)
        candidates.append(
            PageExtractionCandidate(
                engine="pymupdf",
                text=pymupdf_text,
                score=float(pymupdf_metrics["score"]),
                metrics=pymupdf_metrics,
            )
        )
    return candidates


def select_best_candidate(candidates: list[PageExtractionCandidate]) -> PageExtractionCandidate:
    if not candidates:
        return PageExtractionCandidate(engine="none", text="", score=0.0, metrics={"score": 0.0})
    return sorted(candidates, key=lambda item: item.score, reverse=True)[0]


def detect_repeated_margin_lines(page_texts: list[str], max_lines: int, threshold_ratio: float) -> set[str]:
    if not page_texts:
        return set()
    count_threshold = max(2, int(len(page_texts) * threshold_ratio))
    counter: Counter[str] = Counter()
    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        candidates = lines[:max_lines] + lines[-max_lines:]
        for line in candidates:
            if len(line) >= 4:
                counter[line] += 1
    return {line for line, c in counter.items() if c >= count_threshold}


def remove_margin_noise(text: str, repeated_lines: set[str]) -> str:
    if not text or not repeated_lines:
        return text
    cleaned_lines = [line for line in text.splitlines() if line.strip() not in repeated_lines]
    return "\n".join(cleaned_lines)


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
    if args.scan_suspected_char_threshold < 0:
        raise SystemExit("--scan-suspected-char-threshold must be 0 or greater.")
    if args.margin_lines <= 0:
        raise SystemExit("--margin-lines must be greater than 0.")
    if args.margin_threshold_ratio <= 0 or args.margin_threshold_ratio > 1:
        raise SystemExit("--margin-threshold-ratio must be within (0, 1].")

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
            page_candidates: list[PageExtractionCandidate] = []
            for page_zero_idx in range(len(reader.pages)):
                candidates = extract_page_candidates(
                    pdf_path=pdf_path,
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

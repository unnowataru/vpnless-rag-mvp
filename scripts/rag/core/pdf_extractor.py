"""PDF discovery and extraction candidate generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunk_quality import PageExtractionCandidate
from .chunk_quality import compute_text_quality_metrics


def iter_pdfs(pdf_dir: Path, pattern: str) -> list[Path]:
    files = sorted(p for p in pdf_dir.glob(pattern) if p.is_file())
    return [p for p in files if p.suffix.lower() == ".pdf"]


def extract_page_candidates(
    page_idx: int,
    pypdf_reader: Any,
    pymupdf_doc: Any | None,
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

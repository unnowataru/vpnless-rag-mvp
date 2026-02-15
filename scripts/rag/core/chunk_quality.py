"""Extraction quality scoring and margin-noise detection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class PageExtractionCandidate:
    engine: str
    text: str
    score: float
    metrics: dict[str, float]


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
    blank_line_ratio = (len(lines) - len(non_empty_lines)) / float(len(lines)) if lines else 1.0
    duplicate_line_ratio = (
        1.0 - (len(set(non_empty_lines)) / float(len(non_empty_lines))) if non_empty_lines else 1.0
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

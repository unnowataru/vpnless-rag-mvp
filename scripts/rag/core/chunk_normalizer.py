"""Text normalization utilities for PDF ingestion."""

from __future__ import annotations

import re

SENTENCE_BREAKS = "\n。！？.!?"


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # dehyphenate wrapped English words
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def merge_wrapped_lines(text: str, heading_matcher: re.Pattern[str]) -> str:
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
        if prev.endswith(tuple(SENTENCE_BREAKS)) or heading_matcher.match(line):
            merged.append(line)
            continue
        merged[-1] = f"{prev} {line}".strip()
    return "\n".join(merged)


def paragraph_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]

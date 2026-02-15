"""Chunk splitting logic for normalized document text."""

from __future__ import annotations

import re
from typing import Any

from .chunk_normalizer import merge_wrapped_lines
from .chunk_normalizer import normalize_text
from .chunk_normalizer import paragraph_blocks

HEADING_RE = re.compile(
    r"^(第[\d一二三四五六七八九十]+[章条]|[0-9]+[.)]|[A-Za-z]-[0-9]{2}-[0-9]{2})"
)


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


def split_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chars: int,
    *,
    default_section: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    clean = normalize_text(text)
    if not clean:
        return []

    blocks = paragraph_blocks(merge_wrapped_lines(clean, HEADING_RE))
    chunks: list[dict[str, Any]] = []
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

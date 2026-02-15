"""Automatic retrieval scope resolution helpers."""

from __future__ import annotations

import re
from typing import Any


SCOPE_STOP_PHRASES = (
    "について知りたい",
    "について",
    "に関して",
    "を知りたい",
    "を教えて",
    "教えてください",
    "教えて",
    "知りたい",
    "とは",
)

SCOPE_PARTICLE_SPLITS = ("について", "に関して", "とは", "を", "は", "が", "に", "で", "と", "の")

SCOPE_TERM_ALIASES = {
    "交通費": ("旅費",),
    "出張費": ("旅費",),
    "テレワーク": ("テレワーク", "ハイブリッドワーク"),
    "在宅勤務": ("テレワーク", "ハイブリッドワーク"),
}


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_scope_terms(question: str) -> list[str]:
    text = str(question).strip().lower()
    if not text:
        return []

    for phrase in SCOPE_STOP_PHRASES:
        text = text.replace(phrase, " ")

    chunks = [
        chunk.strip()
        for chunk in re.split(r"[ \t\r\n、。,.!！?？/\\()（）【】「」『』:：;；]+", text)
        if chunk.strip()
    ]

    terms: list[str] = []
    for chunk in chunks:
        candidates = [chunk]
        for sep in SCOPE_PARTICLE_SPLITS:
            if sep in chunk:
                candidates.extend(part for part in chunk.split(sep) if part)

        for candidate in candidates:
            candidate = candidate.strip()
            if len(candidate) < 2:
                continue
            terms.append(candidate)
            # Prefixes help map "出張申請" -> "出張".
            if len(candidate) >= 4:
                terms.append(candidate[:2])
                terms.append(candidate[:3])
            alias_terms = SCOPE_TERM_ALIASES.get(candidate)
            if alias_terms:
                terms.extend(alias_terms)

    return _dedupe([term for term in terms if len(term) >= 2])


def infer_doc_id_scope_filters(
    question: str,
    metadata: list[dict[str, Any]],
    *,
    max_docs: int = 8,
) -> dict[str, Any]:
    if max_docs <= 0:
        return {}

    terms = extract_scope_terms(question)
    if not terms:
        return {}

    unique_doc_ids: list[str] = []
    seen_docs: set[str] = set()
    for row in metadata:
        doc_id = str(row.get("doc_id") or row.get("doc") or "").strip()
        if not doc_id or doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        unique_doc_ids.append(doc_id)

    if not unique_doc_ids:
        return {}

    scored: list[tuple[int, str]] = []
    for doc_id in unique_doc_ids:
        lowered = doc_id.lower()
        score = 0
        for term in terms:
            if term.lower() in lowered:
                score += 2 if len(term) >= 4 else 1
        if score > 0:
            scored.append((score, doc_id))

    if not scored:
        return {}

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [doc_id for _score, doc_id in scored[:max_docs]]
    return {"doc_id": selected}

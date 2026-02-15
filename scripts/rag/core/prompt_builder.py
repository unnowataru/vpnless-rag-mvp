"""Evidence and prompt assembly helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .retriever_contract import RetrievalHit
from .retriever_contract import serialize_hit
from .text_sanitizer import sanitize as shared_sanitize


def sanitize(text: str) -> str:
    # Keep public import path stable while using a single shared implementation.
    return shared_sanitize(text)


def build_evidence(
    hits: list[RetrievalHit],
    max_context_chars: int,
    start_rank: int = 1,
) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    entries: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=start_rank):
        doc = hit.doc_meta.get("doc")
        page = hit.doc_meta.get("page")
        chunk = hit.doc_meta.get("chunk")
        section = "/".join(hit.section_path) if hit.section_path else "-"
        labels = ",".join(hit.labels) if hit.labels else "-"

        score_str = f"score={hit.score:.5f}"
        if hit.vector_score is not None:
            score_str += f" vector={hit.vector_score:.5f}"
        if hit.rerank_score is not None:
            score_str += f" rerank={hit.rerank_score:.5f}"
        parts.append(
            f"[{rank}] score({score_str}) doc={doc} "
            f"page={page} chunk={chunk} chunk_id={hit.chunk_id} "
            f"section={section} labels={labels}\n"
            f"{sanitize(hit.text_snippet)}\n"
        )
        entry = serialize_hit(hit)
        entry["rank"] = rank
        entry["source_type"] = "vector"
        entries.append(entry)
    return ("\n".join(parts))[:max_context_chars], entries


def build_runtime_evidence_block(
    rank: int,
    request_id: str,
    now_utc: datetime,
    now_jst: datetime,
) -> tuple[str, dict[str, Any]]:
    block = (
        f"[{rank}] score(runtime=1.00000) doc=runtime-context page=- chunk=-\n"
        f"request_id={request_id} "
        f"executed_at_utc={now_utc.isoformat()} "
        f"executed_at_jst={now_jst.isoformat()} "
        f"today_jst={now_jst.date().isoformat()} "
        "relative_date_reference=today_jst\n"
    )
    entry = {
        "rank": rank,
        "source_type": "runtime_context",
        "request_id": request_id,
        "executed_at_utc": now_utc.isoformat(),
        "executed_at_jst": now_jst.isoformat(),
        "today_jst": now_jst.date().isoformat(),
    }
    return block, entry

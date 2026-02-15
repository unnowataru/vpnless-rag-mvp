"""Retriever contract helpers for local/external vector stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Protocol

ALLOWED_FILTER_KEYS = frozenset(
    {
        "doc_id",
        "label",
        "updated_at",
        "dept",
        "confidentiality",
        "customer",
        "product",
        "doc_type",
        "retention",
    }
)


@dataclass(frozen=True)
class RetrievalHit:
    """Canonical search hit schema used across retrievers."""

    chunk_id: str
    score: float
    text_snippet: str
    doc_meta: dict[str, Any]
    section_path: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    vector_score: float | None = None
    rerank_score: float | None = None
    metadata_index: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    """Stable retriever interface for LocalFaiss/VAST/NetApp adapters."""

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        ...


class RetrieverBackendError(RuntimeError):
    """Errors that indicate backend retriever failure and allow fallback."""


def validate_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    if filters is None:
        return {}
    if not isinstance(filters, dict):
        raise ValueError("filters must be a JSON object")
    unknown = sorted(set(filters.keys()) - ALLOWED_FILTER_KEYS)
    if unknown:
        raise ValueError(
            "Unsupported filter keys: "
            + ", ".join(unknown)
            + ". Allowed keys: "
            + ", ".join(sorted(ALLOWED_FILTER_KEYS))
        )
    return {str(k): v for k, v in filters.items()}


def normalize_search_score(raw_score: float, kind: str = "similarity") -> float:
    score = float(raw_score)
    if kind == "similarity":
        return score
    if kind == "distance":
        return 1.0 / (1.0 + max(0.0, score))
    raise ValueError("kind must be 'similarity' or 'distance'")


def _coerce_tuple(values: Any) -> tuple[str, ...]:
    if isinstance(values, list):
        return tuple(str(v).strip() for v in values if str(v).strip())
    if isinstance(values, tuple):
        return tuple(str(v).strip() for v in values if str(v).strip())
    if isinstance(values, str):
        text = values.strip()
        return (text,) if text else ()
    return ()


def build_text_snippet(text: str, max_chars: int = 400) -> str:
    compact = " ".join(str(text).replace("\n", " ").split())
    return compact[:max_chars]


def build_chunk_id(row: dict[str, Any], fallback_index: int | None = None) -> str:
    existing = str(row.get("chunk_id", "")).strip()
    if existing:
        return existing

    doc_id = str(row.get("doc_id") or row.get("doc") or "unknown-doc")
    page = int(row.get("page", -1))
    chunk = int(row.get("chunk", fallback_index if fallback_index is not None else -1))
    start_offset = int(row.get("start_offset", 0))
    text_hash = sha256(str(row.get("text", "")).encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:p{page}:c{chunk}:o{start_offset}:h{text_hash}"


def build_hit_from_row(
    row: dict[str, Any],
    score: float,
    *,
    snippet_chars: int = 400,
    fallback_index: int | None = None,
    vector_score: float | None = None,
    rerank_score: float | None = None,
) -> RetrievalHit:
    doc = str(row.get("doc", "")).strip()
    doc_id = str(row.get("doc_id", "")).strip() or doc
    page = int(row.get("page", -1))
    chunk = int(row.get("chunk", fallback_index if fallback_index is not None else -1))
    section_path = _coerce_tuple(row.get("section_path"))

    labels = _coerce_tuple(row.get("labels"))
    if not labels:
        labels = _coerce_tuple(row.get("label"))

    return RetrievalHit(
        chunk_id=build_chunk_id(row, fallback_index=fallback_index),
        score=float(score),
        text_snippet=build_text_snippet(str(row.get("text", "")), max_chars=snippet_chars),
        doc_meta={
            "doc": doc,
            "doc_id": doc_id,
            "page": page,
            "chunk": chunk,
            "updated_at": row.get("updated_at"),
            "dept": row.get("dept"),
            "confidentiality": row.get("confidentiality"),
        },
        section_path=section_path,
        labels=labels,
        vector_score=vector_score,
        rerank_score=rerank_score,
        metadata_index=fallback_index,
    )


def serialize_hit(hit: RetrievalHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "score": hit.score,
        "text_snippet": hit.text_snippet,
        "doc_meta": hit.doc_meta,
        "section_path": list(hit.section_path),
        "labels": list(hit.labels),
        "vector_score": hit.vector_score,
        "rerank_score": hit.rerank_score,
        "metadata_index": hit.metadata_index,
        "extra": hit.extra,
    }

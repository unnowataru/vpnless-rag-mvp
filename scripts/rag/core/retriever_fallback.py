"""Fallback wrapper retriever that fails over to local backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .retriever_contract import RetrievalHit
from .retriever_contract import Retriever


@dataclass(frozen=True)
class FallbackSearchResult:
    hits: list[RetrievalHit]
    backend_used: str
    fallback_triggered: bool
    error: str | None


class FallbackRetriever:
    def __init__(
        self,
        *,
        primary_name: str,
        primary: Retriever,
        fallback_name: str,
        fallback: Retriever,
    ) -> None:
        self.primary_name = primary_name
        self.primary = primary
        self.fallback_name = fallback_name
        self.fallback = fallback

    def search_with_fallback(
        self,
        *,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> FallbackSearchResult:
        try:
            hits = self.primary.search(query_text=query_text, top_k=top_k, filters=filters)
            return FallbackSearchResult(
                hits=hits,
                backend_used=self.primary_name,
                fallback_triggered=False,
                error=None,
            )
        except Exception as exc:  # pragma: no cover - runtime fallback path
            hits = self.fallback.search(query_text=query_text, top_k=top_k, filters=filters)
            return FallbackSearchResult(
                hits=hits,
                backend_used=self.fallback_name,
                fallback_triggered=True,
                error=str(exc),
            )

"""VAST retriever adapter scaffold (placeholder for future production wiring)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .retriever_contract import RetrievalHit
from .retriever_contract import RetrieverBackendError
from .retriever_contract import Retriever
from .retriever_contract import normalize_search_score
from .retriever_contract import validate_filters


@dataclass(frozen=True)
class VastRetrieverConfig:
    endpoint: str
    collection: str
    timeout_sec: int = 10


class VastRetriever(Retriever):
    """Adapter contract for VAST SQL/vector search backends."""

    def __init__(self, config: VastRetrieverConfig) -> None:
        self.config = config

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        _ = query_text
        _ = top_k
        _ = validate_filters(filters)
        raise RetrieverBackendError(
            "VAST retriever is not configured in this environment. "
            "Set runtime adapter implementation and credentials first."
        )

    @staticmethod
    def normalize_vast_score(raw_score: float, score_kind: str) -> float:
        # VAST backends may return distance-style values. Normalize to larger-is-better.
        if score_kind == "distance":
            return normalize_search_score(raw_score, kind="distance")
        return normalize_search_score(raw_score, kind="similarity")

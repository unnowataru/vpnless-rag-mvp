"""External retriever adapter scaffold for NetApp/other managed services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .retriever_contract import RetrievalHit
from .retriever_contract import RetrieverBackendError
from .retriever_contract import Retriever
from .retriever_contract import normalize_search_score
from .retriever_contract import validate_filters


@dataclass(frozen=True)
class ExternalRetrieverConfig:
    endpoint: str
    provider: str
    timeout_sec: int = 10


class ExternalRetriever(Retriever):
    """Adapter contract for external search APIs (e.g. NetApp-side retriever)."""

    def __init__(self, config: ExternalRetrieverConfig) -> None:
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
            f"External retriever provider '{self.config.provider}' is not configured in this environment.",
            code="not_configured",
        )

    @staticmethod
    def normalize_external_score(raw_score: float, score_kind: str) -> float:
        if score_kind == "distance":
            return normalize_search_score(raw_score, kind="distance")
        return normalize_search_score(raw_score, kind="similarity")

"""Local vector retriever implementation (FAISS / NumPy backend)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .retriever_contract import RetrievalHit
from .retriever_contract import Retriever
from .retriever_contract import build_hit_from_row
from .retriever_contract import validate_filters

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency: sentence-transformers. "
        "Install with: pip install -r scripts/rag/requirements.txt"
    ) from exc


def load_manifest(index_dir: Path) -> dict[str, Any]:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}. Build index first.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_metadata(index_dir: Path) -> list[dict[str, Any]]:
    metadata_path = index_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}.")
    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _embed_query(model: SentenceTransformer, question: str, query_prefix: str) -> np.ndarray:
    query = f"{query_prefix}{question}" if query_prefix else question
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vec, dtype=np.float32)


def _topk_search(qvec: np.ndarray, topk: int, index_dir: Path, backend: str) -> tuple[np.ndarray, np.ndarray]:
    if backend == "faiss":
        if faiss is None:
            raise RuntimeError("faiss index detected but faiss is not installed.")
        index_path = index_dir / "vectors.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing {index_path}.")
        index = faiss.read_index(str(index_path))
        return index.search(qvec, topk)

    vectors_path = index_dir / "vectors.npy"
    if not vectors_path.exists():
        raise FileNotFoundError(f"Missing {vectors_path}.")
    matrix = np.load(vectors_path).astype(np.float32, copy=False)
    scores = matrix @ qvec[0]
    if len(scores) == 0:
        return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
    topk = min(topk, len(scores))
    candidate_idx = np.argpartition(scores, -topk)[-topk:]
    sorted_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
    return scores[sorted_idx][None, :], sorted_idx[None, :]


def _collect_hits(results_scores: np.ndarray, results_ids: np.ndarray) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for score, idx in zip(results_scores[0], results_ids[0]):
        int_idx = int(idx)
        if int_idx < 0:
            continue
        hits.append({"idx": int_idx, "vector_score": float(score)})
    return hits


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    if isinstance(value, tuple):
        return tuple(str(v) for v in value)
    if value is None:
        return ()
    return (str(value),)


def _matches_filter_value(actual: Any, expected: Any) -> bool:
    actual_values = {v.strip() for v in _as_tuple(actual) if v.strip()}
    if isinstance(expected, (list, tuple)):
        expected_values = {str(v).strip() for v in expected if str(v).strip()}
        return bool(actual_values & expected_values)
    text = str(expected).strip()
    if not text:
        return True
    return text in actual_values


def _matches_updated_at(actual: Any, expected: Any) -> bool:
    actual_text = str(actual or "").strip()
    if not actual_text:
        return False
    if isinstance(expected, dict):
        gte = str(expected.get("gte", "")).strip()
        lte = str(expected.get("lte", "")).strip()
        if gte and actual_text < gte:
            return False
        if lte and actual_text > lte:
            return False
        return True
    return _matches_filter_value(actual_text, expected)


def _row_matches_filters(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if key == "doc_id":
            actual = row.get("doc_id") or row.get("doc")
        elif key == "label":
            labels = row.get("labels")
            actual = labels if labels else row.get("label")
        elif key == "updated_at":
            if not _matches_updated_at(row.get("updated_at"), expected):
                return False
            continue
        else:
            actual = row.get(key)
        if not _matches_filter_value(actual, expected):
            return False
    return True


def build_retrieval_stats(
    total_hits_before_filter: int,
    total_hits_after_filter: int,
    total_hits_after_rerank: int,
) -> dict[str, Any]:
    pass_rate = (
        float(total_hits_after_filter) / float(total_hits_before_filter)
        if total_hits_before_filter > 0
        else 0.0
    )
    return {
        "hits_before_filter": total_hits_before_filter,
        "hits_after_filter": total_hits_after_filter,
        "hits_after_rerank": total_hits_after_rerank,
        "filter_pass_rate": pass_rate,
        "zero_hit": total_hits_after_filter == 0,
    }


@dataclass(frozen=True)
class SearchDiagnostics:
    hits: list[RetrievalHit]
    stats: dict[str, Any]


class LocalVectorRetriever(Retriever):
    def __init__(
        self,
        *,
        index_dir: Path,
        backend: str,
        metadata: list[dict[str, Any]],
        model: SentenceTransformer,
        query_prefix: str,
        snippet_chars: int,
    ) -> None:
        self.index_dir = index_dir
        self.backend = backend
        self.metadata = metadata
        self.model = model
        self.query_prefix = query_prefix
        self.snippet_chars = snippet_chars

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        return self.search_with_diagnostics(query_text=query_text, top_k=top_k, filters=filters).hits

    def search_with_diagnostics(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> SearchDiagnostics:
        validated_filters = validate_filters(filters)
        qvec = _embed_query(self.model, query_text, self.query_prefix)
        scores, ids = _topk_search(qvec, top_k, self.index_dir, self.backend)
        raw_hits = _collect_hits(scores, ids)
        before_count = len(raw_hits)

        filtered_hits: list[dict[str, Any]] = []
        for hit in raw_hits:
            idx = int(hit["idx"])
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = self.metadata[idx]
            if _row_matches_filters(row, validated_filters):
                filtered_hits.append(hit)
        after_count = len(filtered_hits)

        contract_hits: list[RetrievalHit] = []
        for hit in filtered_hits:
            idx = int(hit["idx"])
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = self.metadata[idx]
            vector_score = hit.get("vector_score")
            contract_hits.append(
                build_hit_from_row(
                    row=row,
                    score=float(vector_score if vector_score is not None else 0.0),
                    snippet_chars=self.snippet_chars,
                    fallback_index=idx,
                    vector_score=float(vector_score) if vector_score is not None else None,
                    rerank_score=None,
                )
            )

        stats = build_retrieval_stats(before_count, after_count, after_count)
        return SearchDiagnostics(hits=contract_hits, stats=stats)

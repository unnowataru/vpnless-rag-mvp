"""Helpers to build retrieval stats payloads consistently."""

from __future__ import annotations

from typing import Any


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


def attach_backend_metadata(
    stats: dict[str, Any],
    *,
    backend_used: str,
    fallback_triggered: bool,
    fallback_error: str | None,
    fallback_error_type: str | None,
) -> dict[str, Any]:
    merged = dict(stats)
    merged["retriever_backend_used"] = backend_used
    merged["fallback_triggered"] = fallback_triggered
    merged["fallback_error"] = fallback_error
    merged["fallback_error_type"] = fallback_error_type
    return merged

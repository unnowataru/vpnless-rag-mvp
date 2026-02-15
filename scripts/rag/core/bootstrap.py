"""Shared bootstrap helpers for CLI and API entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .local_retriever import LocalVectorRetriever
from .local_retriever import load_manifest
from .local_retriever import load_metadata
from .query_runtime import RuntimeConfig
from .query_runtime import load_runtime_config
from .query_runtime import load_system_prompt
from .retriever_external import ExternalRetriever
from .retriever_external import ExternalRetrieverConfig
from .retriever_fallback import FallbackRetriever
from .retriever_vast import VastRetriever
from .retriever_vast import VastRetrieverConfig

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency: sentence-transformers. "
        "Install with: pip install -r scripts/rag/requirements.txt"
    ) from exc


@dataclass(frozen=True)
class RuntimeArtifacts:
    manifest: dict[str, Any]
    metadata: list[dict[str, Any]]
    runtime_config: RuntimeConfig
    system_prompt: str


def load_runtime_artifacts(
    *,
    index_dir: Path,
    runtime_config_file: str,
    system_prompt_file: str | None,
) -> RuntimeArtifacts:
    manifest = load_manifest(index_dir)
    metadata = load_metadata(index_dir)
    runtime_config = load_runtime_config(runtime_config_file)
    system_prompt = load_system_prompt(system_prompt_file, runtime_config.default_system_prompt_file)
    return RuntimeArtifacts(
        manifest=manifest,
        metadata=metadata,
        runtime_config=runtime_config,
        system_prompt=system_prompt,
    )


def build_local_retriever(
    *,
    index_dir: Path,
    manifest: dict[str, Any],
    metadata: list[dict[str, Any]],
    snippet_chars: int,
) -> LocalVectorRetriever:
    model = SentenceTransformer(manifest["embedding_model"])
    return LocalVectorRetriever(
        index_dir=index_dir,
        backend=manifest["backend"],
        metadata=metadata,
        model=model,
        query_prefix=manifest.get("query_prefix", ""),
        snippet_chars=snippet_chars,
    )


def build_backend_retrievers(
    *,
    local_retriever: LocalVectorRetriever,
    vast_endpoint: str,
    vast_collection: str,
    external_endpoint: str,
    external_provider: str,
    timeout_sec: int,
    local_fallback_on_retriever_error: bool,
) -> tuple[FallbackRetriever, FallbackRetriever]:
    vast_primary = VastRetriever(
        VastRetrieverConfig(
            endpoint=vast_endpoint,
            collection=vast_collection,
            timeout_sec=timeout_sec,
        )
    )
    external_primary = ExternalRetriever(
        ExternalRetrieverConfig(
            endpoint=external_endpoint,
            provider=external_provider,
            timeout_sec=timeout_sec,
        )
    )
    if local_fallback_on_retriever_error:
        vast_retriever = FallbackRetriever(
            primary_name="vast",
            primary=vast_primary,
            fallback_name="local",
            fallback=local_retriever,
        )
        external_retriever = FallbackRetriever(
            primary_name=external_provider,
            primary=external_primary,
            fallback_name="local",
            fallback=local_retriever,
        )
    else:
        vast_retriever = FallbackRetriever(
            primary_name="vast",
            primary=vast_primary,
            fallback_name="vast",
            fallback=vast_primary,
        )
        external_retriever = FallbackRetriever(
            primary_name=external_provider,
            primary=external_primary,
            fallback_name=external_provider,
            fallback=external_primary,
        )
    return vast_retriever, external_retriever

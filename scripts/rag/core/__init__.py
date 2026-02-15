"""Core modules shared by RAG CLI/API entrypoints.

Keep package import lightweight so tests that import ``core.<module>`` do not
require heavy optional runtime deps (numpy/sentence-transformers/faiss).
"""

from __future__ import annotations

import importlib

from .audit import current_times
from .audit import make_request_id
from .audit import system_prompt_sha256
from .audit import write_audit_log
from .bedrock_client import run_aws_cli
from .ingest_metadata import load_metadata_rules
from .ingest_metadata import missing_required_metadata
from .ingest_metadata import parse_default_metadata_json
from .ingest_metadata import parse_required_metadata_fields
from .ingest_metadata import resolve_document_metadata
from .prompt_builder import build_evidence
from .prompt_builder import build_runtime_evidence_block
from .prompt_builder import sanitize as sanitize_prompt_text
from .retriever_contract import ALLOWED_FILTER_KEYS
from .retriever_contract import RetrievalHit
from .retriever_contract import Retriever
from .retriever_contract import build_hit_from_row
from .retriever_contract import normalize_search_score
from .retriever_contract import serialize_hit
from .retriever_contract import validate_filters
from .scope_resolver import extract_scope_terms
from .scope_resolver import infer_doc_id_scope_filters

_LAZY_EXPORTS = {
    "LocalVectorRetriever": (".local_retriever", "LocalVectorRetriever"),
    "SearchDiagnostics": (".local_retriever", "SearchDiagnostics"),
    "build_retrieval_stats": (".local_retriever", "build_retrieval_stats"),
    "load_manifest": (".local_retriever", "load_manifest"),
    "load_metadata": (".local_retriever", "load_metadata"),
    "ExternalRetriever": (".retriever_external", "ExternalRetriever"),
    "ExternalRetrieverConfig": (".retriever_external", "ExternalRetrieverConfig"),
    "FallbackRetriever": (".retriever_fallback", "FallbackRetriever"),
    "FallbackSearchResult": (".retriever_fallback", "FallbackSearchResult"),
    "VastRetriever": (".retriever_vast", "VastRetriever"),
    "VastRetrieverConfig": (".retriever_vast", "VastRetrieverConfig"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = importlib.import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "ALLOWED_FILTER_KEYS",
    "RetrievalHit",
    "Retriever",
    "build_hit_from_row",
    "build_evidence",
    "build_runtime_evidence_block",
    "current_times",
    "extract_scope_terms",
    "infer_doc_id_scope_filters",
    "load_metadata_rules",
    "make_request_id",
    "missing_required_metadata",
    "normalize_search_score",
    "parse_default_metadata_json",
    "parse_required_metadata_fields",
    "resolve_document_metadata",
    "run_aws_cli",
    "serialize_hit",
    "sanitize_prompt_text",
    "system_prompt_sha256",
    "validate_filters",
    "write_audit_log",
]

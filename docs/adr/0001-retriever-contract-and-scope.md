# ADR-0001: Retriever Contract and Mandatory Retrieval Scope

- Status: Accepted
- Date: 2026-02-15

## Context

RAG quality degraded when retrieval operated without explicit scope and when each backend returned inconsistent hit schemas. We need a stable contract for Local/VAST/NetApp-ready backends and fail-closed behavior for scope resolution.

## Decision

1. Standardize retriever contract:
   - `search(query_text, top_k, filters) -> hits[]`
   - `hits[]` normalized to: `chunk_id`, `score`, `text_snippet`, `doc_meta`, `section_path`, `labels`, `metadata_index`, `vector_score`, `rerank_score`.
2. Standardize filter keys:
   - `doc_id`, `label`, `updated_at`, `dept`, `confidentiality`, `customer`, `product`, `doc_type`, `retention`.
3. Enforce retrieval scope by default (fail-closed):
   - priority: CLI `--filters-json` > auto doc scope inference > runtime default filters
   - if unresolved, abort unless `--allow-unscoped` is explicitly set.
4. Separate retrieval payload and LLM payload:
   - retriever may return long text, but prompt context always uses snippet limits.

## Consequences

- Improves retrieval precision and reduces accidental unrelated hits.
- Enables backend swap with minimal app-layer changes.
- Requires metadata quality gates during ingest to make filtering effective.

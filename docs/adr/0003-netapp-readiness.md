# ADR-0003: NetApp Readiness Strategy (Data/Labels/Snapshot-Aware)

- Status: Accepted
- Date: 2026-02-15

## Context

NetApp integration scope is phased. We need readiness now (not production integration) while preserving future external retriever replacement.

## Decision

1. Add external retriever adapter slot (`retriever-backend=external`) with provider label (default `netapp`) and shared hit normalization.
2. Standardize label-oriented metadata fields used by filters:
   - `dept`, `doc_type`, `confidentiality`, `customer`, `product`, `retention`, `updated_at`, `label(s)`.
3. Preserve audit extensibility for future data lineage:
   - retain request-level scope/filter traces and retrieval backend traces.
4. Keep current local retrieval as fallback baseline until NetApp-side API/SLA/score semantics are fixed.

## Consequences

- Supports phased adoption (storage/classification/retrieval delegation).
- Reduces migration risk by isolating provider-specific logic in adapters.
- Requires future alignment on snapshot/collection identifiers for full reproducibility claims.

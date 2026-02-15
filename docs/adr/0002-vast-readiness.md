# ADR-0002: VAST Readiness Strategy (Adapter + Fallback)

- Status: Accepted
- Date: 2026-02-15

## Context

Production VAST infrastructure is not available yet, but migration readiness is required without changing current QA flow.

## Decision

1. Add VAST retriever adapter slot (`retriever-backend=vast`) implementing the shared retriever contract.
2. Keep VAST implementation scaffolded until endpoint/auth/query specs are finalized.
3. Add operational fallback:
   - on VAST retriever error, fallback to local retriever when `--local-fallback-on-retriever-error` is enabled.
   - record `retriever_backend_used`, `fallback_triggered`, and `fallback_error` in retrieval stats/audit.
4. Treat filters as first-class constraints (not optional hints) for candidate reduction and predictable latency.

## Consequences

- Maintains service continuity before VAST go-live.
- Preserves migration path through adapter replacement instead of CLI contract changes.
- Requires future performance validation with and without filters at target scale.

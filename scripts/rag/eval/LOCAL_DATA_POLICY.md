# Local Eval Data Policy

This directory may be used for local-only evaluation datasets.

Rules:
- Do not commit real document names, real file paths, or customer-identifying IDs.
- Use anonymized IDs only, e.g. `DOC_POLICY_001`.
- Keep local-only golden sets outside git tracking.

Recommended local file name (ignored by git):
- `scripts/rag/eval/golden_questions.local.jsonl`

JSONL schema example:
```json
{"id":"q001","question":"...", "expected_doc_ids":["DOC_POLICY_001"], "filters":{"dept":"hr","label":["policy"]}}
```

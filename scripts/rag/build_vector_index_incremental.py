#!/usr/bin/env python3
"""Incremental index updater with doc_id-level merge/upsert/delete semantics.

This command merges new chunks into existing metadata on doc_id basis, then
rebuilds the vector index via build_vector_index.py for deterministic output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _doc_id(row: dict[str, Any]) -> str:
    return str(row.get("doc_id") or row.get("doc") or "").strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {i}: {exc}") from exc
            if not str(obj.get("text", "")).strip():
                continue
            doc = _doc_id(obj)
            if not doc:
                raise ValueError(f"Missing doc_id/doc in {path} line {i}.")
            rows.append(obj)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def group_by_doc(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_doc_id(row)].append(row)
    return dict(grouped)


def doc_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("page", -1)),
            int(row.get("chunk", -1)),
            int(row.get("start_offset", 0)),
            str(row.get("chunk_id", "")),
        ),
    )
    for row in sorted_rows:
        text = str(row.get("text", ""))
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_deleted_doc_ids(raw: str | None) -> set[str]:
    if not raw:
        return set()
    values = [item.strip() for item in raw.split(",")]
    return {item for item in values if item}


def merge_doc_rows(
    existing_by_doc: dict[str, list[dict[str, Any]]],
    new_by_doc: dict[str, list[dict[str, Any]]],
    deleted_doc_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    all_doc_ids = set(existing_by_doc.keys()) | set(new_by_doc.keys())
    all_doc_ids -= deleted_doc_ids

    merged_rows: list[dict[str, Any]] = []
    changed_docs = 0
    new_docs = 0
    unchanged_docs = 0

    for doc_id in sorted(all_doc_ids):
        if doc_id in new_by_doc:
            merged_rows.extend(new_by_doc[doc_id])
            if doc_id not in existing_by_doc:
                new_docs += 1
            else:
                if doc_fingerprint(new_by_doc[doc_id]) == doc_fingerprint(existing_by_doc[doc_id]):
                    unchanged_docs += 1
                else:
                    changed_docs += 1
        else:
            merged_rows.extend(existing_by_doc[doc_id])

    merged_rows.sort(
        key=lambda row: (
            _doc_id(row),
            int(row.get("page", -1)),
            int(row.get("chunk", -1)),
            int(row.get("start_offset", 0)),
        )
    )
    return merged_rows, {
        "new_docs": new_docs,
        "changed_docs": changed_docs,
        "unchanged_docs": unchanged_docs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="New chunks JSONL path (full or partial docs).")
    parser.add_argument("--index-dir", required=True, help="Target index directory.")
    parser.add_argument(
        "--mode",
        choices=["backfill", "incremental"],
        default="incremental",
        help="backfill ignores existing metadata; incremental merges by doc_id.",
    )
    parser.add_argument(
        "--merged-chunks-out",
        default=None,
        help="Merged chunks output path. Default: <index-dir>/chunks_merged.jsonl",
    )
    parser.add_argument(
        "--deleted-doc-ids",
        default=None,
        help="Comma-separated doc_id list to delete from merged index.",
    )
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backend", choices=["faiss", "numpy"], default="faiss")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0.")

    index_dir = Path(args.index_dir)
    metadata_path = index_dir / "metadata.jsonl"
    merged_out = Path(args.merged_chunks_out) if args.merged_chunks_out else index_dir / "chunks_merged.jsonl"

    new_rows = read_jsonl(Path(args.chunks))
    new_by_doc = group_by_doc(new_rows)
    deleted_doc_ids = parse_deleted_doc_ids(args.deleted_doc_ids)

    existing_rows: list[dict[str, Any]] = []
    if args.mode == "incremental" and metadata_path.exists():
        existing_rows = read_jsonl(metadata_path)

    existing_by_doc = group_by_doc(existing_rows)
    merged_rows, merge_stats = merge_doc_rows(
        existing_by_doc=existing_by_doc,
        new_by_doc=new_by_doc,
        deleted_doc_ids=deleted_doc_ids,
    )

    print(f"Mode: {args.mode}")
    print(f"Incoming docs: {len(new_by_doc)} / rows: {len(new_rows)}")
    print(f"Existing docs: {len(existing_by_doc)} / rows: {len(existing_rows)}")
    print(f"Deleted docs requested: {len(deleted_doc_ids)}")
    print(f"Merged docs: {len({ _doc_id(row) for row in merged_rows })} / rows: {len(merged_rows)}")
    print(
        "Doc changes: "
        f"new={merge_stats['new_docs']} "
        f"changed={merge_stats['changed_docs']} "
        f"unchanged={merge_stats['unchanged_docs']}"
    )

    write_jsonl(merged_rows, merged_out)
    print(f"Merged chunks written: {merged_out}")

    if args.dry_run:
        print("Dry-run mode: skipped build_vector_index.py execution.")
        return

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("build_vector_index.py")),
        "--chunks",
        str(merged_out),
        "--index-dir",
        str(index_dir),
        "--embedding-model",
        args.embedding_model,
        "--batch-size",
        str(args.batch_size),
        "--backend",
        args.backend,
    ]
    subprocess.run(cmd, check=True)

    state_path = index_dir / "doc_fingerprints.json"
    merged_by_doc = group_by_doc(merged_rows)
    fingerprints = {doc_id: doc_fingerprint(rows) for doc_id, rows in merged_by_doc.items()}
    state_payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "doc_count": len(fingerprints),
        "doc_fingerprints": fingerprints,
    }
    state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Fingerprint state updated: {state_path}")


if __name__ == "__main__":
    main()

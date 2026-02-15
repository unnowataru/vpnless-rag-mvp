"""Metadata enrichment helpers for chunk ingestion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_METADATA_KEYS = frozenset(
    {
        "doc_id",
        "dept",
        "label",
        "labels",
        "updated_at",
        "confidentiality",
        "customer",
        "product",
        "doc_type",
        "retention",
    }
)

HEURISTIC_LABEL_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"faq", re.IGNORECASE), "faq"),
    (re.compile(r"規程|規則|細則|制度|ガイドライン"), "policy"),
    (re.compile(r"旅費|出張|交通費"), "travel"),
    (re.compile(r"休暇|休業"), "leave"),
    (re.compile(r"給与|賞与|退職金"), "compensation"),
)


@dataclass(frozen=True)
class MetadataRule:
    pattern: re.Pattern[str]
    fields: dict[str, Any]


def _clean_str(value: Any) -> str:
    return str(value).strip()


def normalize_labels(value: Any) -> list[str]:
    if value is None:
        return []
    items: list[str] = []
    if isinstance(value, (list, tuple)):
        for part in value:
            label = _clean_str(part)
            if label:
                items.append(label)
    elif isinstance(value, str):
        for part in re.split(r"[,;/|]", value):
            label = _clean_str(part)
            if label:
                items.append(label)
    else:
        label = _clean_str(value)
        if label:
            items.append(label)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _normalize_metadata_fields(fields: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, raw in fields.items():
        if key not in SUPPORTED_METADATA_KEYS:
            continue
        if key == "labels":
            labels = normalize_labels(raw)
            if labels:
                normalized["labels"] = labels
                normalized["label"] = labels[0]
            continue
        if key == "label":
            if "labels" in normalized:
                continue
            labels = normalize_labels(raw)
            if labels:
                normalized["labels"] = labels
                normalized["label"] = labels[0]
            continue
        text = _clean_str(raw)
        if text:
            normalized[key] = text
    return normalized


def parse_default_metadata_json(raw: str | None) -> dict[str, Any]:
    if raw is None or not raw.strip():
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"default metadata JSON is invalid: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("default metadata JSON must be an object")
    return _normalize_metadata_fields(loaded)


def load_metadata_rules(path: str | None) -> list[MetadataRule]:
    if path is None:
        return []
    rule_path = Path(path).expanduser().resolve()
    if not rule_path.exists():
        raise FileNotFoundError(f"metadata rules file not found: {rule_path}")

    raw = json.loads(rule_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        rows = raw.get("rules")
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ValueError("metadata rules must be a list or an object containing 'rules'")

    rules: list[MetadataRule] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"metadata rule #{idx} must be an object")
        pattern_text = _clean_str(row.get("pattern"))
        if not pattern_text:
            raise ValueError(f"metadata rule #{idx} requires non-empty 'pattern'")

        fields_raw = row.get("fields")
        if fields_raw is None:
            fields_raw = {k: v for k, v in row.items() if k != "pattern"}
        if not isinstance(fields_raw, dict):
            raise ValueError(f"metadata rule #{idx} field set must be an object")
        fields = _normalize_metadata_fields(fields_raw)
        rules.append(MetadataRule(pattern=re.compile(pattern_text), fields=fields))
    return rules


def infer_labels_from_doc_id(doc_id: str) -> list[str]:
    text = doc_id.strip()
    if not text:
        return []
    labels: list[str] = []
    for pattern, label in HEURISTIC_LABEL_RULES:
        if pattern.search(text):
            labels.append(label)
    return normalize_labels(labels)


def infer_dept_from_relative_path(relative_doc_id: str) -> str | None:
    parts = [part for part in Path(relative_doc_id).parts if part not in {".", ""}]
    if len(parts) >= 2:
        first = parts[0].strip()
        if first:
            return first
    return None


def _resolve_doc_id(pdf_path: Path, pdf_root: Path) -> str:
    try:
        rel = pdf_path.resolve().relative_to(pdf_root.resolve())
        return rel.as_posix()
    except ValueError:
        return pdf_path.name


def _resolve_updated_at(pdf_path: Path) -> str:
    mtime = pdf_path.stat().st_mtime
    return datetime.fromtimestamp(mtime, timezone.utc).date().isoformat()


def resolve_document_metadata(
    pdf_path: Path,
    *,
    pdf_root: Path,
    default_metadata: dict[str, Any] | None = None,
    metadata_rules: list[MetadataRule] | None = None,
    include_updated_at_from_mtime: bool = True,
) -> dict[str, Any]:
    doc_id = _resolve_doc_id(pdf_path, pdf_root)
    target = doc_id

    merged: dict[str, Any] = {}
    if default_metadata:
        merged.update(_normalize_metadata_fields(default_metadata))
    if metadata_rules:
        for rule in metadata_rules:
            if rule.pattern.search(target):
                merged.update(rule.fields)

    merged.setdefault("doc_id", doc_id)
    if include_updated_at_from_mtime and "updated_at" not in merged:
        merged["updated_at"] = _resolve_updated_at(pdf_path)

    if "dept" not in merged:
        inferred_dept = infer_dept_from_relative_path(doc_id)
        if inferred_dept:
            merged["dept"] = inferred_dept

    labels = normalize_labels(merged.get("labels"))
    if not labels and "label" in merged:
        labels = normalize_labels(merged.get("label"))
    if not labels:
        labels = infer_labels_from_doc_id(doc_id)
    if labels:
        merged["labels"] = labels
        merged["label"] = labels[0]
        if "doc_type" not in merged:
            merged["doc_type"] = "faq" if "faq" in labels else "policy"

    return _normalize_metadata_fields(merged)


def parse_required_metadata_fields(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    fields = [part.strip() for part in raw.split(",") if part.strip()]
    if not fields:
        return ()
    unknown = sorted(set(fields) - SUPPORTED_METADATA_KEYS)
    if unknown:
        raise ValueError(
            "unsupported required metadata fields: "
            + ", ".join(unknown)
            + ". supported: "
            + ", ".join(sorted(SUPPORTED_METADATA_KEYS))
        )
    return tuple(fields)


def has_metadata_value(doc_metadata: dict[str, Any], field: str) -> bool:
    value = doc_metadata.get(field)
    if field == "labels":
        return bool(normalize_labels(value))
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def missing_required_metadata(doc_metadata: dict[str, Any], required_fields: tuple[str, ...]) -> list[str]:
    return [field for field in required_fields if not has_metadata_value(doc_metadata, field)]

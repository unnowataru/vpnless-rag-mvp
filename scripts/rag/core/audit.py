"""Audit/logging helpers shared across CLI and future API."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

JST = timezone(timedelta(hours=9), "JST")


def make_request_id(provided: str | None) -> str:
    if provided:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", provided).strip("._-")
        if safe:
            return safe[:128]
    return uuid.uuid4().hex


def current_times() -> tuple[datetime, datetime]:
    now_utc = datetime.now(timezone.utc)
    now_jst = now_utc.astimezone(JST)
    return now_utc, now_jst


def system_prompt_sha256(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def write_audit_log(log_dir: str, request_id: str, payload: dict[str, Any]) -> Path:
    root = Path(log_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    stamp = payload.get("executed_at_utc", "").replace(":", "").replace("-", "")
    safe_stamp = re.sub(r"[^0-9TZ\\.]", "", str(stamp)) or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out = root / f"{safe_stamp}_{request_id}.json"
    if out.exists():
        out = root / f"{safe_stamp}_{request_id}_{uuid.uuid4().hex[:8]}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

"""Shared text sanitization helpers."""

from __future__ import annotations

import re

EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
TEL_RE = re.compile(r"\b\d{2,4}-\d{2,4}-\d{3,4}\b")


def sanitize(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = TEL_RE.sub("[TEL]", text)
    return text

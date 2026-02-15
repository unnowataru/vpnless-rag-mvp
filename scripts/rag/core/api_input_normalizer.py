"""Shared input normalizers for OpenAI-compatible and Dify-compatible APIs."""

from __future__ import annotations

from typing import Any


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and str(item.get("type", "")).strip() == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def extract_openai_question(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array.")
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role != "user":
            continue
        text = _content_to_text(item.get("content")).strip()
        if text:
            return text
    raise ValueError("messages must contain at least one user message with text content.")


def build_openai_options_payload(
    payload: dict[str, Any],
    answer_profile_to_model: dict[str, str],
) -> dict[str, Any]:
    options_payload: dict[str, Any] = {}
    for key in (
        "top_k",
        "filters",
        "retriever_backend",
        "rerank",
        "allow_unscoped",
        "auto_scope_max_docs",
        "request_id",
    ):
        if key in payload:
            options_payload[key] = payload[key]

    model_selector = str(payload.get("model", "")).strip()
    if model_selector:
        if model_selector in answer_profile_to_model:
            options_payload["answer_profile"] = model_selector
        else:
            options_payload["bedrock_model"] = model_selector
    return options_payload


def extract_dify_question(payload: dict[str, Any]) -> str:
    inputs = payload.get("inputs")
    inputs_dict = inputs if isinstance(inputs, dict) else {}
    question = str(
        payload.get("question")
        or payload.get("query")
        or inputs_dict.get("question")
        or inputs_dict.get("query")
        or ""
    ).strip()
    if not question:
        raise ValueError("question (or query) is required.")
    return question


def build_dify_options_payload(payload: dict[str, Any], inputs_dict: dict[str, Any]) -> dict[str, Any]:
    options_payload: dict[str, Any] = {}
    for key in (
        "top_k",
        "filters",
        "retriever_backend",
        "rerank",
        "allow_unscoped",
        "auto_scope_max_docs",
        "answer_profile",
        "bedrock_model",
        "request_id",
    ):
        if key in payload:
            options_payload[key] = payload[key]
        elif key in inputs_dict:
            options_payload[key] = inputs_dict[key]
    return options_payload

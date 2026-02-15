"""HTTP transport and routing for RAG API server."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from api_endpoints import AppContext
from api_endpoints import _backend_from_request
from api_endpoints import _backend_status
from api_endpoints import _int_from_request
from api_endpoints import _json_error
from api_endpoints import _openai_error_body
from api_endpoints import _parse_qa_options
from api_endpoints import _read_json_body
from api_endpoints import _resolve_filters_and_scope
from api_endpoints import _run_qa_request
from api_endpoints import _search
from api_endpoints import _serialize_hits
from core.api_input_normalizer import build_dify_options_payload
from core.api_input_normalizer import build_openai_options_payload
from core.api_input_normalizer import extract_dify_question
from core.api_input_normalizer import extract_openai_question
from core.audit import system_prompt_sha256 as core_system_prompt_sha256
from core.audit import write_audit_log as core_write_audit_log
from core.query_runtime import sanitize


class RagHTTPServer(ThreadingHTTPServer):
    context: AppContext


class RagRequestHandler(BaseHTTPRequestHandler):
    server: RagHTTPServer

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", self.server.context.cors_allow_origin)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_openai_error(self, status: int, message: str, *, err_type: str) -> None:
        self._send(status, _openai_error_body(message, err_type=err_type))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", self.server.context.cors_allow_origin)
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,Authorization")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            ctx = self.server.context
            self._send(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "rag-api",
                    "backends": {
                        "local": {"configured": True, "status": "ready", "reason": None},
                        "vast": _backend_status(ctx.vast_endpoint),
                        "external": _backend_status(ctx.external_endpoint),
                    },
                    "fallback": {
                        "enabled": ctx.local_fallback_on_retriever_error,
                    },
                },
            )
            return
        if path == "/v1/models":
            self._handle_openai_models()
            return
        self._send(*_json_error("Not found.", status=HTTPStatus.NOT_FOUND))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = _read_json_body(self)
        except ValueError as exc:
            self._send(*_json_error(str(exc), status=HTTPStatus.BAD_REQUEST))
            return

        if path == "/search":
            self._handle_search(payload)
            return
        if path == "/qa":
            self._handle_qa(payload)
            return
        if path == "/v1/chat/completions":
            self._handle_openai_chat_completions(payload)
            return
        if path == "/integrations/dify/qa":
            self._handle_dify_qa(payload)
            return
        self._send(*_json_error("Not found.", status=HTTPStatus.NOT_FOUND))

    def _handle_search(self, payload: dict[str, Any]) -> None:
        ctx = self.server.context
        query_text = str(payload.get("query_text", "")).strip()
        if not query_text:
            self._send(*_json_error("query_text is required.", status=HTTPStatus.BAD_REQUEST))
            return

        try:
            top_k = _int_from_request(payload, "top_k", ctx.default_topk, minimum=1)
            backend = _backend_from_request(payload)
            filters, scope_source = _resolve_filters_and_scope(ctx, query_text=query_text, payload=payload)
            hits, retrieval_stats = _search(
                ctx,
                query_text=query_text,
                top_k=top_k,
                filters=filters,
                backend=backend,
            )
        except (ValueError, RuntimeError) as exc:
            status = HTTPStatus.BAD_REQUEST if isinstance(exc, ValueError) else HTTPStatus.BAD_GATEWAY
            self._send(*_json_error(str(exc), status=status))
            return

        response = {
            "query_text": query_text,
            "top_k": top_k,
            "scope_source": scope_source,
            "filters": filters,
            "retrieval_stats": retrieval_stats,
            "hits": _serialize_hits(hits),
        }
        self._send(HTTPStatus.OK, response)

    def _handle_openai_models(self) -> None:
        ctx = self.server.context
        rows: list[dict[str, Any]] = []
        for profile in ctx.runtime_config.answer_profiles:
            key = str(getattr(profile, "key", "")).strip()
            if not key:
                continue
            model_id = str(
                getattr(profile, "model_id", "") or ctx.runtime_config.answer_profile_to_model.get(key, "")
            ).strip()
            description = str(getattr(profile, "description", "")).strip()
            rows.append(
                {
                    "id": key,
                    "object": "model",
                    "created": 0,
                    "owned_by": "vpnless-rag",
                    "metadata": {
                        "bedrock_model_id": model_id,
                        "description": description,
                    },
                }
            )
        self._send(HTTPStatus.OK, {"object": "list", "data": rows})

    def _handle_openai_chat_completions(self, payload: dict[str, Any]) -> None:
        ctx = self.server.context
        try:
            question = extract_openai_question(payload.get("messages"))
            options_payload = build_openai_options_payload(payload, ctx.runtime_config.answer_profile_to_model)
            if "stream" in payload and bool(payload.get("stream")):
                raise ValueError("stream=true is not supported. Use stream=false.")
            options = _parse_qa_options(ctx, options_payload)
            result = _run_qa_request(ctx, question=question, options=options)
        except ValueError as exc:
            self._send_openai_error(HTTPStatus.BAD_REQUEST, str(exc), err_type="invalid_request_error")
            return
        except SystemExit as exc:
            self._send_openai_error(HTTPStatus.BAD_REQUEST, str(exc), err_type="invalid_request_error")
            return
        except RuntimeError as exc:
            self._send_openai_error(HTTPStatus.BAD_GATEWAY, str(exc), err_type="api_error")
            return

        completion = {
            "id": f"chatcmpl-{result.request_id[:24]}",
            "object": "chat.completion",
            "created": int(result.now_utc.timestamp()),
            "model": options.answer_profile if options.bedrock_model_override is None else options.bedrock_model_override,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._send(HTTPStatus.OK, completion)

    def _handle_dify_qa(self, payload: dict[str, Any]) -> None:
        ctx = self.server.context
        try:
            inputs = payload.get("inputs")
            inputs_dict = inputs if isinstance(inputs, dict) else {}
            question = extract_dify_question(payload)
            options_payload = build_dify_options_payload(payload, inputs_dict)
            options = _parse_qa_options(ctx, options_payload)
            result = _run_qa_request(ctx, question=question, options=options)
        except ValueError as exc:
            self._send(
                HTTPStatus.BAD_REQUEST,
                {"code": "invalid_request", "message": str(exc)},
            )
            return
        except SystemExit as exc:
            self._send(
                HTTPStatus.BAD_REQUEST,
                {"code": "invalid_request", "message": str(exc)},
            )
            return
        except RuntimeError as exc:
            self._send(
                HTTPStatus.BAD_GATEWAY,
                {"code": "backend_error", "message": str(exc)},
            )
            return

        self._send(
            HTTPStatus.OK,
            {
                "status": result.status,
                "request_id": result.request_id,
                "answer": result.answer,
                "outputs": {"answer": result.answer},
                "metadata": {
                    "model_id": result.model_id,
                    "scope_source": result.scope_source,
                    "filters": result.filters,
                    "retrieval_stats": result.retrieval_stats,
                },
            },
        )

    def _handle_qa(self, payload: dict[str, Any]) -> None:
        ctx = self.server.context
        question = str(payload.get("question") or payload.get("query_text") or "").strip()
        if not question:
            self._send(*_json_error("question is required.", status=HTTPStatus.BAD_REQUEST))
            return

        try:
            options = _parse_qa_options(ctx, payload)
        except ValueError as exc:
            self._send(*_json_error(str(exc), status=HTTPStatus.BAD_REQUEST))
            return

        try:
            result = _run_qa_request(ctx, question=question, options=options)
        except ValueError as exc:
            self._send(*_json_error(str(exc), status=HTTPStatus.BAD_REQUEST))
            return
        except SystemExit as exc:
            self._send(*_json_error(str(exc), status=HTTPStatus.BAD_REQUEST))
            return
        except RuntimeError as exc:
            self._send(*_json_error(str(exc), status=HTTPStatus.BAD_GATEWAY))
            return

        response = {
            "request_id": result.request_id,
            "status": result.status,
            "question": question,
            "answer": result.answer,
            "error": result.error,
            "answer_profile": options.answer_profile,
            "model_id": result.model_id,
            "scope_source": result.scope_source,
            "filters": result.filters,
            "retrieval_stats": result.retrieval_stats,
            "evidence_entries": result.evidence_entries,
        }

        if ctx.audit_log_dir:
            payload_row = {
                "request_id": result.request_id,
                "executed_at_utc": result.now_utc.isoformat(),
                "executed_at_jst": result.now_jst.isoformat(),
                "region": ctx.region,
                "profile": ctx.profile,
                "model_id": result.model_id,
                "question": sanitize(question),
                "answer": result.answer,
                "status": result.status,
                "error": result.error,
                "system_prompt_sha256": core_system_prompt_sha256(ctx.system_prompt),
                "evidence_entries": result.evidence_entries,
                "topk": options.top_k,
                "rerank": options.rerank_enabled,
                "rerank_model": ctx.rerank_model,
                "rerank_topn": ctx.rerank_topn,
                "filters": result.filters,
                "scope_source": result.scope_source,
                "retrieval_stats": result.retrieval_stats,
            }
            audit_path = core_write_audit_log(ctx.audit_log_dir, result.request_id, payload_row)
            response["audit_log_path"] = str(audit_path)

        self._send(HTTPStatus.OK, response)

#!/usr/bin/env python3
"""Vector-based RAG CLI: local vector search + Bedrock Converse."""

from __future__ import annotations

from pathlib import Path
from core.bootstrap import build_backend_retrievers
from core.bootstrap import build_local_retriever
from core.bootstrap import load_runtime_artifacts
from core.cli_args import parse_args
from core.cli_args import validate_args
from core.cli_query import prompt_answer_profile
from core.cli_query import run_single_query
from core.query_runtime import parse_filters_json
from core.query_runtime import resolve_bedrock_model
from core.query_runtime import select_default_answer_profile
from core.retriever_fallback import FallbackRetriever


def main() -> None:
    args = parse_args()
    app_config = validate_args(args)
    args.topk = app_config.topk
    args.rerank = app_config.rerank
    args.rerank_model = app_config.rerank_model
    args.rerank_topn = app_config.rerank_topn
    args.max_context_chars = app_config.max_context_chars
    args.max_tokens = app_config.max_tokens
    args.snippet_max_chars = app_config.snippet_max_chars
    args.region = app_config.region
    args.profile = app_config.profile
    args.aws_timeout_sec = app_config.aws_timeout_sec
    args.aws_retries = app_config.aws_retries
    args.aws_retry_backoff_sec = app_config.aws_retry_backoff_sec
    args.auto_scope_max_docs = app_config.auto_scope_max_docs
    args.allow_unscoped = app_config.allow_unscoped
    args.explicit_retrieval_filters = parse_filters_json(args.filters_json)

    index_dir = Path(args.index_dir)
    artifacts = load_runtime_artifacts(
        index_dir=index_dir,
        runtime_config_file=args.runtime_config_file,
        system_prompt_file=args.system_prompt_file,
    )
    runtime_config = artifacts.runtime_config
    metadata = artifacts.metadata
    args.runtime_default_retrieval_filters = runtime_config.default_retrieval_filters

    system_prompt = artifacts.system_prompt
    local_retriever = build_local_retriever(
        index_dir=index_dir,
        manifest=artifacts.manifest,
        metadata=metadata,
        snippet_chars=args.snippet_max_chars,
    )
    external_fallback_retriever: FallbackRetriever | None = None
    vast_retriever, external_retriever = build_backend_retrievers(
        local_retriever=local_retriever,
        vast_endpoint=args.vast_endpoint,
        vast_collection=args.vast_collection,
        external_endpoint=args.external_endpoint,
        external_provider=args.external_provider,
        timeout_sec=args.aws_timeout_sec,
        local_fallback_on_retriever_error=args.local_fallback_on_retriever_error,
    )
    if args.retriever_backend == "vast":
        external_fallback_retriever = vast_retriever
    elif args.retriever_backend == "external":
        external_fallback_retriever = external_retriever

    configured_default = select_default_answer_profile(runtime_config.answer_profiles)
    selected_profile = args.answer_profile or configured_default
    if selected_profile not in runtime_config.answer_profile_to_model:
        allowed = ", ".join(runtime_config.answer_profile_to_model.keys())
        raise SystemExit(f"Unsupported --answer-profile: {selected_profile}. Allowed: {allowed}")

    if args.interactive:
        if args.bedrock_model:
            print(f"[INFO] --bedrock-model is set. answer profile prompt is skipped: {args.bedrock_model}")
        else:
            selected_profile = prompt_answer_profile(configured_default, runtime_config.answer_profiles)
            selected_model = resolve_bedrock_model(
                selected_profile,
                None,
                runtime_config.answer_profile_to_model,
            )
            print(f"[INFO] Using answer profile '{selected_profile}' ({selected_model})")
        if args.system_prompt_file:
            print(f"[INFO] Using custom system prompt: {args.system_prompt_file}")
        print("Interactive mode started. Type 'exit' or 'quit' to finish.")
        while True:
            try:
                question = input("Q> ").strip()
            except EOFError:
                print()
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            run_single_query(
                question=question,
                args=args,
                metadata=metadata,
                answer_profile=selected_profile,
                system_prompt=system_prompt,
                temporal_rules=runtime_config.temporal_rules,
                answer_profile_to_model=runtime_config.answer_profile_to_model,
                local_retriever=local_retriever,
                external_fallback_retriever=external_fallback_retriever,
            )
            print()
        return

    question = " ".join(args.question).strip() or input("Q> ").strip()
    if not question:
        raise SystemExit("Question is required.")
    run_single_query(
        question=question,
        args=args,
        metadata=metadata,
        answer_profile=selected_profile,
        system_prompt=system_prompt,
        temporal_rules=runtime_config.temporal_rules,
        answer_profile_to_model=runtime_config.answer_profile_to_model,
        local_retriever=local_retriever,
        external_fallback_retriever=external_fallback_retriever,
    )


if __name__ == "__main__":
    main()

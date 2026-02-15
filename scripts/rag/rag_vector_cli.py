#!/usr/bin/env python3
"""Vector-based RAG CLI: local vector search + Bedrock Converse."""

from __future__ import annotations

from pathlib import Path
from core.cli_args import parse_args
from core.cli_args import validate_args
from core.cli_query import prompt_answer_profile
from core.cli_query import run_single_query
from core.local_retriever import LocalVectorRetriever
from core.local_retriever import load_manifest
from core.local_retriever import load_metadata
from core.query_runtime import load_runtime_config
from core.query_runtime import load_system_prompt
from core.query_runtime import parse_filters_json
from core.query_runtime import resolve_bedrock_model
from core.query_runtime import select_default_answer_profile
from core.retriever_external import ExternalRetriever
from core.retriever_external import ExternalRetrieverConfig
from core.retriever_fallback import FallbackRetriever
from core.retriever_vast import VastRetriever
from core.retriever_vast import VastRetrieverConfig

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency: sentence-transformers. "
        "Install with: pip install -r scripts/rag/requirements.txt"
    ) from exc


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.explicit_retrieval_filters = parse_filters_json(args.filters_json)

    index_dir = Path(args.index_dir)

    manifest = load_manifest(index_dir)
    metadata = load_metadata(index_dir)
    model_name = manifest["embedding_model"]
    query_prefix = manifest.get("query_prefix", "")
    runtime_config = load_runtime_config(args.runtime_config_file)
    args.runtime_default_retrieval_filters = runtime_config.default_retrieval_filters

    model = SentenceTransformer(model_name)
    system_prompt = load_system_prompt(args.system_prompt_file, runtime_config.default_system_prompt_file)
    local_retriever = LocalVectorRetriever(
        index_dir=index_dir,
        backend=manifest["backend"],
        metadata=metadata,
        model=model,
        query_prefix=query_prefix,
        snippet_chars=args.snippet_max_chars,
    )
    external_fallback_retriever: FallbackRetriever | None = None
    if args.retriever_backend == "vast":
        primary = VastRetriever(
            VastRetrieverConfig(
                endpoint=args.vast_endpoint,
                collection=args.vast_collection,
                timeout_sec=args.aws_timeout_sec,
            )
        )
        if args.local_fallback_on_retriever_error:
            external_fallback_retriever = FallbackRetriever(
                primary_name="vast",
                primary=primary,
                fallback_name="local",
                fallback=local_retriever,
            )
        else:
            external_fallback_retriever = FallbackRetriever(
                primary_name="vast",
                primary=primary,
                fallback_name="vast",
                fallback=primary,
            )
    elif args.retriever_backend == "external":
        primary = ExternalRetriever(
            ExternalRetrieverConfig(
                endpoint=args.external_endpoint,
                provider=args.external_provider,
                timeout_sec=args.aws_timeout_sec,
            )
        )
        if args.local_fallback_on_retriever_error:
            external_fallback_retriever = FallbackRetriever(
                primary_name=args.external_provider,
                primary=primary,
                fallback_name="local",
                fallback=local_retriever,
            )
        else:
            external_fallback_retriever = FallbackRetriever(
                primary_name=args.external_provider,
                primary=primary,
                fallback_name=args.external_provider,
                fallback=primary,
            )

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

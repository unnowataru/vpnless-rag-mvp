#!/usr/bin/env python3
"""Compatibility entrypoint for the split RAG API server modules."""

from __future__ import annotations

from api_bootstrap import main
from api_bootstrap import parse_args
from api_endpoints import AppContext
from api_endpoints import QaRequestOptions
from api_transport import RagHTTPServer
from api_transport import RagRequestHandler

__all__ = [
    "AppContext",
    "QaRequestOptions",
    "RagHTTPServer",
    "RagRequestHandler",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    main()

"""ASGI server entrypoint for the knowledge access service."""

from __future__ import annotations

import os


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run the HTTP service: pip install -e .[service]") from exc

    uvicorn.run(
        "rag.api:create_app",
        factory=True,
        host=os.getenv("RAG_HOST", "127.0.0.1"),
        port=int(os.getenv("RAG_PORT", "8000")),
        reload=os.getenv("RAG_RELOAD", "0") in {"1", "true", "True"},
    )


if __name__ == "__main__":
    main()

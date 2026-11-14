"""Command line entry point: `python -m app.retrieval <command>`."""

import argparse
import sys

from app.db.session import SessionLocal
from app.retrieval.client import RateLimitError, VoyageClient
from app.retrieval.service import embed_all
from app.settings import settings


def main(argv: list[str] | None = None) -> int:
    """Runs the CLI.

    Args:
        argv: Arguments, defaulting to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="python -m app.retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("embed", help="embed chunks that are new or changed")
    parser.parse_args(argv)

    if not settings.voyage_api_key:
        print("VOYAGE_API_KEY is not set", file=sys.stderr)
        return 1
    client = VoyageClient(
        settings.voyage_api_key, settings.embedding_model, settings.embedding_dimension
    )
    try:
        report = embed_all(SessionLocal, client)
    except RateLimitError as e:
        print(f"rate limited, giving up: {e}", file=sys.stderr)
        return 1
    print(f"embedded {report.embedded}, skipped {report.skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

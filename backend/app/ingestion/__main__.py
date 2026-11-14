"""Command line entry point: `python -m app.ingestion fetch`."""

import argparse
import asyncio
import sys
from pathlib import Path

from app.db.session import SessionLocal
from app.ingestion.service import fetch_all
from app.ingestion.titles import TITLES_PATH


def main(argv: list[str] | None = None) -> int:
    """Runs the CLI.

    Args:
        argv: Arguments, defaulting to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="python -m app.ingestion")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="fetch articles at pinned revisions")
    fetch.add_argument("--titles", type=Path, default=TITLES_PATH)
    args = parser.parse_args(argv)

    report = asyncio.run(fetch_all(SessionLocal, args.titles))
    print(f"stored {len(report.stored)}, unchanged {len(report.unchanged)}")
    for title, reason in report.skipped.items():
        print(f"skipped {title}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

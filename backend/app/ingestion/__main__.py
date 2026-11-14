"""Command line entry point: `python -m app.ingestion fetch|wikidata`."""

import argparse
import asyncio
import sys
from pathlib import Path

from app.db.session import SessionLocal
from app.ingestion.service import fetch_all
from app.ingestion.titles import TITLES_PATH
from app.ingestion.wikidata import fetch_wikidata


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
    sub.add_parser("wikidata", help="fetch defining formulas from Wikidata")
    args = parser.parse_args(argv)

    if args.command == "wikidata":
        wikidata = asyncio.run(fetch_wikidata(SessionLocal))
        print(
            f"formulas for {len(wikidata.with_formulas)}, none for {len(wikidata.none_recorded)}, "
            f"no item {len(wikidata.no_qid)}, failed {len(wikidata.failed)}"
        )
        for title, reason in wikidata.failed.items():
            print(f"failed {title}: {reason}", file=sys.stderr)
        return 0

    report = asyncio.run(fetch_all(SessionLocal, args.titles))
    print(f"stored {len(report.stored)}, unchanged {len(report.unchanged)}")
    for title, reason in report.skipped.items():
        print(f"skipped {title}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

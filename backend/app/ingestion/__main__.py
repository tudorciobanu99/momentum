"""Command line entry point: `python -m app.ingestion fetch|formulas|wikidata|citations|report`."""

import argparse
import asyncio
import sys
from pathlib import Path

from app.db.session import SessionLocal
from app.ingestion.report import render_report
from app.ingestion.service import citations_all, extract_all, fetch_all
from app.ingestion.titles import TITLES_PATH
from app.ingestion.wikidata import fetch_wikidata


def _kinds(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items())


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
    sub.add_parser("formulas", help="extract formulas from stored articles")
    sub.add_parser("wikidata", help="fetch defining formulas from Wikidata")
    sub.add_parser("citations", help="extract citations and link them to formulas")
    report_cmd = sub.add_parser("report", help="print the pilot report as Markdown")
    report_cmd.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "report":
        with SessionLocal() as session:
            markdown = render_report(session)
        if args.out:
            args.out.write_text(markdown, encoding="utf-8")
        else:
            sys.stdout.write(markdown)
        return 0

    if args.command == "formulas":
        counts = extract_all(SessionLocal)
        print(f"{sum(counts.values())} formulas from {len(counts)} articles")
        return 0

    if args.command == "citations":
        c = citations_all(SessionLocal)
        print(
            f"{c.citations} citations ({c.with_doi} doi, {c.with_isbn} isbn), "
            f"{c.merged_sources} sources merged; "
            f"display formulas {sum(c.display.values())}: {_kinds(c.display)}; "
            f"inline formulas {sum(c.inline.values())}: {_kinds(c.inline)}"
        )
        return 0

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

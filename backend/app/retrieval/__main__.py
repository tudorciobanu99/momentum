"""Command line entry point: `python -m app.retrieval <command>`."""

import argparse
import sys
from pathlib import Path

from app.db.models import Chunk
from app.db.session import SessionLocal
from app.retrieval.client import RateLimitError, VoyageClient
from app.retrieval.evals import EVALS_PATH, check_database, load_questions
from app.retrieval.hybrid import hybrid_search
from app.retrieval.service import embed_all
from app.settings import settings


def _validate(path: Path) -> int:
    questions, problems = load_questions(path)
    with SessionLocal() as session:
        problems += check_database(session, questions)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} problems in {len(questions)} questions", file=sys.stderr)
        return 1
    print(f"{len(questions)} questions valid")
    return 0


def _search(client: VoyageClient, query: str) -> int:
    try:
        with SessionLocal() as session:
            candidates = hybrid_search(session, client, query)
            paths = {
                c.chunk_id: session.get_one(Chunk, c.chunk_id).heading_path for c in candidates
            }
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    except RateLimitError as e:
        print(f"rate limited: {e}", file=sys.stderr)
        return 1
    for c in candidates:
        print(
            f"{c.chunk_id}\t{paths[c.chunk_id]}\t{c.score:.6f}"
            f"\tvector={c.vector_rank}\tlexical={c.lexical_rank}"
        )
    return 0


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
    search = sub.add_parser("search", help="print hybrid search candidates for a question")
    search.add_argument("query")
    evals = sub.add_parser("evals", help="retrieval evaluation dataset")
    evals_sub = evals.add_subparsers(dest="action", required=True)
    validate = evals_sub.add_parser("validate", help="check the dataset against the database")
    validate.add_argument("--path", type=Path, default=EVALS_PATH)
    args = parser.parse_args(argv)

    if args.command == "evals":
        return _validate(args.path)

    if not settings.voyage_api_key:
        print("VOYAGE_API_KEY is not set", file=sys.stderr)
        return 1
    client = VoyageClient(
        settings.voyage_api_key, settings.embedding_model, settings.embedding_dimension
    )
    if args.command == "search":
        return _search(client, args.query)
    try:
        report = embed_all(SessionLocal, client)
    except RateLimitError as e:
        print(f"rate limited, giving up: {e}", file=sys.stderr)
        return 1
    print(f"embedded {report.embedded}, skipped {report.skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

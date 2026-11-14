"""Command line entry point: `python -m app.retrieval <command>`."""

import argparse
import sys
from pathlib import Path

from app.db.session import SessionLocal
from app.retrieval.evals import EVALS_PATH, check_database, load_questions


def main(argv: list[str] | None = None) -> int:
    """Runs the CLI.

    Args:
        argv: Arguments, defaulting to `sys.argv[1:]`.

    Returns:
        Process exit code, 1 if any problem was found.
    """
    parser = argparse.ArgumentParser(prog="python -m app.retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    evals = sub.add_parser("evals", help="retrieval evaluation dataset")
    evals_sub = evals.add_subparsers(dest="action", required=True)
    validate = evals_sub.add_parser("validate", help="check the dataset against the database")
    validate.add_argument("--path", type=Path, default=EVALS_PATH)
    args = parser.parse_args(argv)

    questions, problems = load_questions(args.path)
    with SessionLocal() as session:
        problems += check_database(session, questions)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} problems in {len(questions)} questions", file=sys.stderr)
        return 1
    print(f"{len(questions)} questions valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line entry point: `python -m app.retrieval <command>`."""

import argparse
import json
import sys
from pathlib import Path

from app.db.models import Chunk
from app.db.session import SessionLocal
from app.retrieval.client import RateLimitError, RerankError, VoyageClient, VoyageReranker
from app.retrieval.eval_runner import format_reports, run_evals
from app.retrieval.evals import EVALS_PATH, check_database, load_questions
from app.retrieval.hybrid import hybrid_search
from app.retrieval.langsmith_results import LangSmithResults, ResultsClient
from app.retrieval.models import ProvenanceLink, RetrievedChunk
from app.retrieval.rerank import rerank_candidates
from app.retrieval.retrieve import retrieve
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


def _run_evals(
    client: VoyageClient,
    api_key: str,
    path: Path,
    compare_merged: bool,
    limit: int | None,
    upload: bool,
) -> int:
    questions, problems = load_questions(path)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    if limit is not None:
        questions = questions[: max(limit, 0)]
    results: ResultsClient | None = None
    if upload and settings.langsmith_api_key:
        results = LangSmithResults(settings.langsmith_api_key)
    elif upload:
        print(
            "warning: LANGSMITH_API_KEY is not set, results are printed but not uploaded",
            file=sys.stderr,
        )
    try:
        run = run_evals(
            SessionLocal,
            client,
            VoyageReranker(api_key, settings.rerank_model),
            questions,
            results,
            compare_merged,
            settings.langsmith_project,
        )
    except RateLimitError as e:
        print(f"rate limited: {e}", file=sys.stderr)
        return 1
    except RerankError as e:
        print(f"rerank failed: {e}", file=sys.stderr)
        return 1
    print(format_reports(run.reports))
    if run.uploaded:
        print("\nuploaded to LangSmith")
    return 0


def _search(client: VoyageClient, api_key: str, query: str, rerank: bool) -> int:
    try:
        with SessionLocal() as session:
            candidates = hybrid_search(session, client, query)
            ranked = None
            if rerank:
                reranker = VoyageReranker(api_key, settings.rerank_model)
                ranked = rerank_candidates(session, reranker, query, candidates)
            ids = [c.chunk_id for c in (ranked if ranked is not None else candidates)]
            paths = {i: session.get_one(Chunk, i).heading_path for i in ids}
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    except RateLimitError as e:
        print(f"rate limited: {e}", file=sys.stderr)
        return 1
    except RerankError as e:
        print(f"rerank failed: {e}", file=sys.stderr)
        return 1
    if ranked is not None:
        for r in ranked:
            print(
                f"{r.chunk_id}\t{paths[r.chunk_id]}\trerank={r.rerank_score:.6f}"
                f"\tfused={r.score:.6f}\tvector={r.vector_rank}\tlexical={r.lexical_rank}"
            )
        return 0
    for c in candidates:
        print(
            f"{c.chunk_id}\t{paths[c.chunk_id]}\t{c.score:.6f}"
            f"\tvector={c.vector_rank}\tlexical={c.lexical_rank}"
        )
    return 0


def _describe(link: ProvenanceLink) -> str:
    if link.kind == "none":
        return f"none -> {link.revision.title} rev {link.revision.rev_id}"
    title = link.source.title if link.source and link.source.title else "unknown source"
    page = f", p. {link.page}" if link.page else ""
    return f"{link.kind}: {title}{page} ({link.revision.title} rev {link.revision.rev_id})"


def _print_chunks(chunks: list[RetrievedChunk]) -> None:
    for c in chunks:
        print(
            f"{c.chunk_id}\t{c.heading_path}\trerank={c.rerank_score:.6f}"
            f"\tfused={c.fused_score:.6f}\tvector={c.vector_rank}\tlexical={c.lexical_rank}"
        )
        for f in c.formulas:
            print(f"  formula {f.index}: {f.latex}")
            if f.provenance:
                print(f"    {_describe(f.provenance[0])}")
        if c.citations:
            print("  Sources in this chunk")
        for cite in c.citations:
            title = cite.source.title if cite.source and cite.source.title else "unknown source"
            page = f", p. {cite.page}" if cite.page else ""
            print(f"    {title}{page} ({cite.revision.title} rev {cite.revision.rev_id})")


def _retrieve(
    client: VoyageClient, api_key: str, query: str, top_n: int | None, as_json: bool
) -> int:
    try:
        with SessionLocal() as session:
            chunks = retrieve(
                session, client, VoyageReranker(api_key, settings.rerank_model), query, top_n
            )
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    except RateLimitError as e:
        print(f"rate limited: {e}", file=sys.stderr)
        return 1
    except RerankError as e:
        print(f"rerank failed: {e}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps([c.model_dump(mode="json") for c in chunks], indent=2))
    else:
        _print_chunks(chunks)
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
    search = sub.add_parser("search", help="print search results for a question")
    search.add_argument("query")
    search.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rerank candidates with Voyage (default), --no-rerank shows fused order",
    )
    get = sub.add_parser("retrieve", help="print ranked chunks with formulas and provenance")
    get.add_argument("query")
    get.add_argument("--top-n", type=int, default=None)
    get.add_argument("--json", action="store_true", help="print the result models as JSON")
    evals = sub.add_parser("evals", help="retrieval evaluation dataset")
    evals_sub = evals.add_subparsers(dest="action", required=True)
    validate = evals_sub.add_parser("validate", help="check the dataset against the database")
    validate.add_argument("--path", type=Path, default=EVALS_PATH)
    run = evals_sub.add_parser("run", help="measure retrieval quality on the dataset")
    run.add_argument("--path", type=Path, default=EVALS_PATH)
    run.add_argument(
        "--compare-merged",
        action="store_true",
        help="also evaluate a chunking with short chunks merged, in a temporary schema",
    )
    run.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    run.add_argument("--no-upload", action="store_true", help="print results, skip LangSmith")
    args = parser.parse_args(argv)

    if args.command == "evals" and args.action == "validate":
        return _validate(args.path)

    if not settings.voyage_api_key:
        print("VOYAGE_API_KEY is not set", file=sys.stderr)
        return 1
    client = VoyageClient(
        settings.voyage_api_key, settings.embedding_model, settings.embedding_dimension
    )
    if args.command == "evals":
        return _run_evals(
            client,
            settings.voyage_api_key,
            args.path,
            args.compare_merged,
            args.limit,
            not args.no_upload,
        )
    if args.command == "search":
        return _search(client, settings.voyage_api_key, args.query, args.rerank)
    if args.command == "retrieve":
        return _retrieve(client, settings.voyage_api_key, args.query, args.top_n, args.json)
    try:
        report = embed_all(SessionLocal, client)
    except RateLimitError as e:
        print(f"rate limited, giving up: {e}", file=sys.stderr)
        return 1
    print(f"embedded {report.embedded}, skipped {report.skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

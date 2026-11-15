"""Tests for the evaluation runner, its metrics, the LangSmith upload and the merged corpus."""

import math
from collections.abc import Iterator
from typing import cast

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, Chunk, ChunkFormula, Formula
from app.ingestion.chunks import count_tokens
from app.retrieval import __main__ as cli
from app.retrieval.client import FakeClient, FakeReranker, InputType, VoyageClient
from app.retrieval.corpus import (
    MERGED_SCHEMA,
    ChunkRow,
    build_merged_corpus,
    corpus_session,
    merge_groups,
    merged_text,
)
from app.retrieval.eval_runner import (
    QuestionOutcome,
    Retrieved,
    _gold_sections,
    _load,
    build_report,
    format_reports,
    question_type,
    run_evals,
    score_question,
    summarize,
)
from app.retrieval.evals import EvalQuestion, GoldFormula, GoldPassage
from app.retrieval.hybrid import hybrid_search
from app.retrieval.langsmith_results import DATASET_NAME, FakeResults
from app.retrieval.rerank import dedup_sections, rank_all, rerank_candidates
from app.settings import settings

DIM = settings.embedding_dimension


def question(
    qid: str,
    anchors: list[str | list[str]],
    formulas: list[int] | None = None,
    topic: str = "mechanics",
    note: str | None = "concept",
) -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        question=f"question {qid}",
        topic=topic,
        gold_passages=[
            GoldPassage(article="A", rev_id=1, anchors=[a] if isinstance(a, str) else a)
            for a in anchors
        ],
        gold_formulas=[GoldFormula(article="A", rev_id=1, index=i) for i in formulas or []],
        note=note,
    )


def chunk(n: int, body: str = "filler", formulas: tuple[int, ...] = ()) -> Retrieved:
    return Retrieved(n, f"A > h{n}", body, 1, frozenset(("A", 1, i) for i in formulas))


def filler(count: int, start: int = 100) -> list[Retrieved]:
    return [chunk(start + i) for i in range(count)]


def outcomes() -> list[QuestionOutcome]:
    q1 = question("q1", ["alpha beta", "gamma delta"], note="derivation")
    # Fused: one anchor at rank 3. Ranked: anchors at ranks 2 and 5.
    f1 = [*filler(2), chunk(1, "alpha beta"), *filler(17, 200)]
    r1 = [*filler(1, 300), chunk(2, "alpha beta"), *filler(2, 310), chunk(3, "gamma delta")]
    q2 = question("q2", ["epsilon"], note="cite; source: x", topic="quantum")
    o2 = QuestionOutcome(q2, filler(20), filler(10))
    q3 = question("q3", ["zeta"], formulas=[7], topic="quantum", note="which-formula")
    f3 = [chunk(4, "zeta"), *filler(19, 400)]
    r3 = [chunk(4, "zeta"), *filler(3, 410), chunk(5, "x", formulas=(7,))]
    q4 = question("q4", ["eta"], formulas=[8], note="which-formula")
    # Fused: hit only at rank 21. Ranked: hit at rank 4, gold formula only at rank 6.
    f4 = [*filler(20, 500), chunk(6, "eta")]
    r4 = [*filler(3, 510), chunk(6, "eta"), chunk(8), chunk(9, "x", formulas=(8,))]
    return [
        QuestionOutcome(q1, f1, r1),
        o2,
        QuestionOutcome(q3, f3, r3),
        QuestionOutcome(q4, f4, r4),
    ]


def test_scores_of_one_question() -> None:
    s = score_question(outcomes()[0])
    assert s.recall is True
    assert (s.passages_found, s.passages_total) == (1, 2)
    assert s.reciprocal_rank == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    assert s.ndcg == pytest.approx(dcg / (1 + 1 / math.log2(3)))
    assert s.ndcg == pytest.approx(0.6240, abs=1e-4)
    assert s.formula_hit is None


def test_recall_ignores_hits_beyond_rank_20() -> None:
    s = score_question(outcomes()[3])
    assert s.recall is False
    assert s.passages_found == 0
    assert s.reciprocal_rank == 0.25
    assert s.ndcg == pytest.approx(1 / math.log2(5))


def test_formula_hit_uses_top_5_only() -> None:
    scores = [score_question(o) for o in outcomes()]
    assert [s.formula_hit for s in scores] == [None, None, True, False]


def test_metrics_overall() -> None:
    m = summarize([score_question(o) for o in outcomes()])
    assert m.questions == 4
    assert m.recall_at_20 == pytest.approx(0.5)
    assert m.passage_recall_at_20 == pytest.approx(2 / 5)
    assert m.ndcg_at_10 == pytest.approx((0.62403 + 0 + 1 + 1 / math.log2(5)) / 4, abs=1e-4)
    assert m.mrr == pytest.approx((0.5 + 0 + 1 + 0.25) / 4)
    assert m.formula_hit_at_5 == pytest.approx(0.5)
    assert m.formula_questions == 2


def test_formula_hit_is_undefined_without_gold_formulas() -> None:
    m = summarize([score_question(outcomes()[0])])
    assert m.formula_hit_at_5 is None
    assert "formula_hit@5" not in m.as_dict()


def test_ndcg_is_capped_at_one_and_ideal_uses_gold_count() -> None:
    q = question("q", ["a1", "b1"])
    ranked = [chunk(1, "a1 b1"), chunk(2, "a1")]
    both = score_question(QuestionOutcome(q, ranked, ranked))
    assert both.ndcg_strict == pytest.approx(1.0)
    assert both.ndcg == pytest.approx(1.0)
    one = score_question(QuestionOutcome(question("q", ["a1"]), ranked, ranked))
    assert one.ndcg == 1.0 and one.ndcg_strict == 1.0


SECTION = frozenset({(1, "A > h9")})


def sectioned(
    ranked: list[Retrieved], sections: list[frozenset[tuple[int, str]]] | None = None
) -> QuestionOutcome:
    q = question("q", [["alpha", "alt"], "beta"])
    return QuestionOutcome(q, [], ranked, [frozenset(), SECTION] if sections is None else sections)


def test_alternate_anchor_counts_as_hit() -> None:
    s = score_question(sectioned([chunk(1, "xx alt xx")]))
    assert s.hits == [True]
    assert s.reciprocal_rank == s.reciprocal_rank_strict == 1.0
    assert s.ndcg == s.ndcg_strict == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_section_credit_in_ndcg_and_mrr_but_not_strict() -> None:
    ranked = [chunk(9, "other"), chunk(2), chunk(1, "alt")]
    s = score_question(sectioned(ranked))
    ideal = 1 + 1 / math.log2(3)
    assert s.ndcg == pytest.approx((0.5 + 1 / 2) / ideal)
    assert s.ndcg == pytest.approx(0.6131, abs=1e-4)
    assert s.reciprocal_rank == 0.5
    assert s.ndcg_strict == pytest.approx(0.5 / ideal)
    assert s.reciprocal_rank_strict == pytest.approx(1 / 3)
    assert s.hits == [False, False, True]


def test_passage_without_section_gets_no_partial_credit() -> None:
    ranked = [chunk(9, "other")]
    s = score_question(sectioned(ranked, [frozenset(), frozenset()]))
    assert (s.ndcg, s.reciprocal_rank) == (0.0, 0.0)
    assert score_question(sectioned(ranked, [])).ndcg == 0.0


def test_passage_credited_once_at_best_position() -> None:
    q = question("q", ["alpha", "beta"])
    section = frozenset({(1, "A > h9"), (1, "A > h8")})
    ranked = [chunk(9, "x"), chunk(8, "y"), chunk(1, "alpha beta")]
    s = score_question(QuestionOutcome(q, [], ranked, [section, section]))
    # alpha and beta both take 1.0 at rank 3, section credit is not added on top
    assert s.ndcg == pytest.approx(2 / 2 / (1 + 1 / math.log2(3)))


def test_credited_scores_never_below_strict() -> None:
    q = question("q", ["alpha", "beta", "gamma"])
    section = frozenset({(1, "A > h9")})
    cases = [
        [chunk(1, "alpha"), chunk(2, "alpha")],
        [chunk(9, "x"), chunk(1, "alpha beta"), chunk(2, "alpha")],
        [chunk(2, "gamma"), chunk(9, "x"), chunk(3, "beta")],
        [chunk(9, "x")],
    ]
    for ranked in cases:
        s = score_question(QuestionOutcome(q, [], ranked, [section, frozenset(), section]))
        assert s.ndcg >= s.ndcg_strict - 1e-12
        assert s.reciprocal_rank >= s.reciprocal_rank_strict


def test_metrics_on_tiny_dataset_with_partial_credit() -> None:
    a = sectioned([chunk(9, "other"), chunk(2), chunk(1, "alt")])
    b = QuestionOutcome(question("q2", ["beta"]), [], [chunk(3, "beta")])
    m = summarize([score_question(a), score_question(b)])
    assert m.ndcg_at_10 == pytest.approx((0.6131 + 1) / 2, abs=1e-4)
    assert m.ndcg_at_10_strict == pytest.approx((0.5 / (1 + 1 / math.log2(3)) + 1) / 2)
    assert m.mrr == pytest.approx((0.5 + 1) / 2)
    assert m.mrr_strict == pytest.approx((1 / 3 + 1) / 2)
    assert {"ndcg@10_strict", "mrr_strict"} <= set(m.as_dict())


def test_report_groups_and_worst_order() -> None:
    report = build_report([score_question(o) for o in outcomes()])
    assert list(report.by_topic) == ["mechanics", "quantum"]
    assert list(report.by_type) == ["cite", "derivation", "which-formula"]
    assert report.by_topic["quantum"].questions == 2
    assert report.by_topic["quantum"].mrr == pytest.approx(0.5)
    assert report.by_type["which-formula"].formula_hit_at_5 == pytest.approx(0.5)
    assert [w.id for w in report.worst] == ["q2", "q4", "q1", "q3"]
    assert report.worst[0].top_paths == ["A > h100", "A > h101", "A > h102"]
    assert report.worst[0].anchors == ["epsilon"]


def test_worst_is_limited_to_ten_and_ties_break_by_id() -> None:
    outs = [QuestionOutcome(question(f"q{n:02d}", ["zz"]), [], filler(3)) for n in range(12)]
    report = build_report([score_question(o) for o in outs])
    assert [w.id for w in report.worst] == [f"q{n:02d}" for n in range(10)]


def test_question_type_is_text_before_semicolon() -> None:
    assert question_type(question("q", ["a"], note="cite; source: x")) == "cite"
    assert question_type(question("q", ["a"], note=None)) == "unknown"


def test_format_is_deterministic_and_side_by_side() -> None:
    scores = [score_question(o) for o in outcomes()]
    one = {"baseline": build_report(scores)}
    two = {"baseline": build_report(scores), "merged": build_report(scores)}
    assert format_reports(one) == format_reports(one)
    text_out = format_reports(two)
    assert "<- baseline" in text_out and "<- merged" in text_out
    assert "worst questions (merged)" in text_out
    assert "nDCG@10s" in text_out and "MRRs" in text_out


def test_sections_are_derived_from_stored_chunks(
    sessions: sessionmaker[Session], corpus: None
) -> None:
    q = question("q", [ANCHOR_A, [ANCHOR_B, "nowhere"], "absent"])
    with sessions() as session:
        assert _gold_sections(session, q) == [
            frozenset({(1, "A > S")}),
            frozenset({(1, "A > S")}),
            frozenset(),
        ]
        loaded = _load(session, {c for (c,) in session.execute(select(Chunk.id))})
    assert {c.rev_id for c in loaded.values()} == {1}


def row(n: int, tokens: int, path: str = "A > S", article: int = 1) -> ChunkRow:
    return ChunkRow(n, article, n, path, f"{path}\n\nbody{n}", tokens)


def test_merge_groups() -> None:
    rows = [
        row(0, 100),
        row(1, 100),
        row(2, 150),
        row(3, 400),
        row(4, 50),
        row(5, 80, "A > T"),
        row(6, 80, "A > T > U"),
        row(7, 80, "A > S", article=2),
    ]
    groups = [[r.id for r in g] for g in merge_groups(rows)]
    assert groups == [[0, 1, 2], [3, 4], [5, 6], [7]]


def test_merged_text_keeps_first_path() -> None:
    group = [row(0, 10), row(1, 10, "A > S > Sub")]
    assert merged_text(group) == "A > S\n\nbody0\n\nbody1"


def test_cli_dispatch_requires_known_action() -> None:
    with pytest.raises(SystemExit):
        cli.main(["evals"])


ANCHOR_A = "the first anchor"
ANCHOR_B = "the second anchor"


def vec(*head: float) -> list[float]:
    return [*head, *([0.0] * (DIM - len(head)))]


@pytest.fixture
def corpus(sessions: sessionmaker[Session]) -> Iterator[None]:
    with sessions.begin() as session:
        session.execute(delete(Article))
        article = Article(title="A", rev_id=1, url="u", topic="mechanics", license="l")
        session.add(article)
        session.flush()
        formulas = [
            Formula(
                article_id=article.id,
                index=i,
                latex="x",
                display=True,
                section_path="s",
                position=i,
            )
            for i in range(3)
        ]
        session.add_all(formulas)
        specs = [
            ("A > S", f"A > S\n\n{ANCHOR_A} energy", 100, vec(1.0, 0.0)),
            ("A > S", f"A > S\n\n{ANCHOR_B} momentum", 250, vec(0.0, 1.0)),
            ("A > S", "A > S\n\nlong " + "word " * 5, 400, vec(0.5, 0.5)),
            ("A > T", "A > T\n\nlone", 50, vec(-1.0, 0.0)),
        ]
        chunks = [
            Chunk(
                article_id=article.id,
                index=i,
                heading_path=h,
                text=t,
                token_count=n,
                embedding=e,
                embedding_hash=f"h{i}",
            )
            for i, (h, t, n, e) in enumerate(specs)
        ]
        session.add_all(chunks)
        session.flush()
        for c, f in [(0, 0), (1, 1), (2, 2), (3, 2)]:
            session.add(ChunkFormula(chunk_id=chunks[c].id, formula_id=formulas[f].id))
    yield
    with sessions.begin() as session:
        session.execute(delete(Article))
        session.execute(text(f"DROP SCHEMA IF EXISTS {MERGED_SCHEMA} CASCADE"))


def schema_exists(sessions: sessionmaker[Session]) -> bool:
    with sessions() as session:
        return bool(
            session.scalar(
                text("SELECT count(*) FROM pg_namespace WHERE nspname = :n"), {"n": MERGED_SCHEMA}
            )
        )


def public_snapshot(sessions: sessionmaker[Session]) -> list[tuple[int, str, int, str | None]]:
    with sessions() as session:
        return [
            (i, t, n, h)
            for i, t, n, h in session.execute(
                select(Chunk.id, Chunk.text, Chunk.token_count, Chunk.embedding_hash).order_by(
                    Chunk.id
                )
            )
        ]


class DocumentFailure(FakeClient):
    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        if input_type == "document":
            raise RuntimeError("boom")
        return super().embed(texts, input_type)


def test_merged_corpus_lives_in_its_own_schema(
    sessions: sessionmaker[Session], corpus: None
) -> None:
    before = public_snapshot(sessions)
    client = FakeClient(DIM)
    assert build_merged_corpus(sessions, client) == 3
    assert public_snapshot(sessions) == before
    assert [c[1] for c in client.calls] == ["document"]
    assert len(client.calls[0][0]) == 1
    with corpus_session(sessions, MERGED_SCHEMA) as session:
        rows = session.execute(
            select(
                Chunk.heading_path, Chunk.text, Chunk.token_count, Chunk.embedding_hash
            ).order_by(Chunk.index)
        ).all()
        links = session.execute(
            select(ChunkFormula.chunk_id, func.count()).group_by(ChunkFormula.chunk_id)
        ).all()
        indexes = session.execute(
            text("SELECT indexdef FROM pg_indexes WHERE schemaname = :s AND tablename = 'chunks'"),
            {"s": MERGED_SCHEMA},
        ).scalars()
        defs = " ".join(indexes)
    assert rows[0][0] == "A > S"
    assert rows[0][1].startswith("A > S\n\n" + ANCHOR_A)
    assert ANCHOR_B in rows[0][1]
    assert rows[0][2] == count_tokens(rows[0][1])
    assert rows[1][0] == "A > S" and rows[1][3] == "h2"
    assert sorted(n for _, n in links) == [1, 1, 2]
    assert "hnsw" in defs and "gin" in defs


def test_run_evals_compares_and_drops_schema(sessions: sessionmaker[Session], corpus: None) -> None:
    questions = [question("q1", [ANCHOR_A]), question("q2", [ANCHOR_B], formulas=[1])]
    run = run_evals(
        sessions,
        FakeClient(DIM, vector=lambda _: vec(1.0, 0.0)),
        FakeReranker(),
        questions,
        None,
        compare_merged=True,
    )
    assert list(run.reports) == ["baseline", "merged"]
    assert run.uploaded is False
    assert run.reports["merged"].overall.questions == 2
    assert not schema_exists(sessions)


def test_dedup_and_nodedup_from_one_rerank_call(
    sessions: sessionmaker[Session], corpus: None
) -> None:
    questions = [question("q1", [ANCHOR_A]), question("q2", [ANCHOR_B], formulas=[1])]
    client = FakeClient(DIM, vector=lambda _: vec(1.0, 0.0))
    reranker = FakeReranker()
    results = FakeResults()
    run = run_evals(
        sessions, client, reranker, questions, results, experiment="exp", compare_dedup=True
    )
    assert list(run.reports) == ["dedup", "nodedup"]
    assert len(reranker.calls) == len(questions)
    assert [e[1] for e in results.experiments] == ["exp-dedup", "exp-nodedup"]
    assert all(e[0] == DATASET_NAME for e in results.experiments)
    text_out = format_reports(run.reports)
    assert "<- dedup" in text_out and "<- nodedup" in text_out
    assert "topic" in text_out and "mechanics" in text_out and "concept" in text_out
    dedup, nodedup = (e[2]["q1"]["chunk_ids"] for e in results.experiments)
    assert len(nodedup) > len(dedup)

    with sessions() as session:
        candidates = hybrid_search(session, client, questions[0].question)
        every, sections = rank_all(session, FakeReranker(), questions[0].question, candidates)
        expected = rerank_candidates(session, FakeReranker(), questions[0].question, candidates, 10)
    assert nodedup == [r.chunk_id for r in every[:10]]
    assert dedup == [r.chunk_id for r in expected]
    assert dedup == [r.chunk_id for r in dedup_sections(every, sections)[:10]]


def test_compare_flags_are_exclusive(sessions: sessionmaker[Session]) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        run_evals(sessions, FakeClient(DIM), FakeReranker(), [], None, True, compare_dedup=True)
    with pytest.raises(SystemExit):
        cli.main(["evals", "run", "--compare-merged", "--compare-dedup"])


def test_merged_schema_dropped_on_error(sessions: sessionmaker[Session], corpus: None) -> None:
    before = public_snapshot(sessions)
    with pytest.raises(RuntimeError, match="boom"):
        run_evals(
            sessions,
            DocumentFailure(DIM),
            FakeReranker(),
            [question("q1", [ANCHOR_A])],
            None,
            compare_merged=True,
        )
    assert not schema_exists(sessions)
    assert public_snapshot(sessions) == before


def test_results_reach_langsmith_fake(sessions: sessionmaker[Session], corpus: None) -> None:
    questions = [question("q1", [ANCHOR_A]), question("q2", [ANCHOR_B], formulas=[1])]
    results = FakeResults()
    run = run_evals(
        sessions,
        FakeClient(DIM, vector=lambda _: vec(1.0, 0.0)),
        FakeReranker(),
        questions,
        results,
        experiment="exp",
    )
    run_evals(
        sessions,
        FakeClient(DIM, vector=lambda _: vec(1.0, 0.0)),
        FakeReranker(),
        questions,
        results,
    )
    assert run.uploaded is True
    assert [q.id for q in results.datasets[DATASET_NAME]] == ["q1", "q2"]
    dataset, name, outputs, metrics = results.experiments[0]
    assert (dataset, name) == (DATASET_NAME, "exp-baseline")
    assert set(outputs) == {"q1", "q2"}
    assert len(outputs["q1"]["chunk_ids"]) == len(outputs["q1"]["heading_paths"]) <= 10
    assert outputs["q1"]["hits"][0] in (True, False)
    assert set(metrics) >= {
        "recall@20",
        "passage_recall@20",
        "ndcg@10",
        "ndcg@10_strict",
        "mrr",
        "mrr_strict",
        "formula_hit@5",
    }
    assert {"ndcg@10_strict", "reciprocal_rank_strict"} <= set(outputs["q1"])
    assert metrics["recall@20"] == 1.0


def test_missing_key_prints_locally_with_warning(
    sessions: sessionmaker[Session],
    corpus: None,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads: list[str] = []

    class Boom:
        def __init__(self, key: str) -> None:
            uploads.append(key)

    monkeypatch.setattr(cli, "SessionLocal", sessions)
    monkeypatch.setattr(cli, "LangSmithResults", Boom)
    monkeypatch.setattr(cli.settings, "langsmith_api_key", None)
    monkeypatch.setattr(cli, "VoyageReranker", lambda *_: FakeReranker())
    monkeypatch.setattr(
        cli,
        "load_questions",
        lambda _: ([question("q1", [ANCHOR_A])], []),
    )
    code = cli._run_evals(
        cast(VoyageClient, FakeClient(DIM, vector=lambda _: vec(1.0, 0.0))),
        "key",
        cli.EVALS_PATH,
        False,
        None,
        True,
    )
    out = capsys.readouterr()
    assert code == 0
    assert "LANGSMITH_API_KEY is not set" in out.err
    assert "recall" in out.out.lower() or "R@20" in out.out
    assert uploads == []
    assert "uploaded" not in out.out

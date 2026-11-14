"""Tests for retrieval results with formulas and provenance."""

from collections.abc import Iterator

import pytest
from sqlalchemy import delete, event
from sqlalchemy.orm import Session, sessionmaker

from app.db import models as db
from app.retrieval.client import FakeClient, FakeReranker
from app.retrieval.models import RetrievedChunk
from app.retrieval.retrieve import retrieve
from app.settings import settings

DIM = settings.embedding_dimension


def vec(x: float) -> list[float]:
    return [x, *([0.0] * (DIM - 1))]


@pytest.fixture(autouse=True)
def clean(sessions: sessionmaker[Session]) -> Iterator[None]:
    with sessions.begin() as session:
        session.execute(delete(db.Article))
        session.execute(delete(db.Source))
        session.execute(delete(db.WikidataFormula))
    yield
    with sessions.begin() as session:
        session.execute(delete(db.Article))
        session.execute(delete(db.Source))


def seed_article(session: Session, title: str = "Energy", rev_id: int = 7) -> db.Article:
    article = db.Article(
        title=title, rev_id=rev_id, url=f"https://w/{rev_id}", topic="t", license="CC BY-SA"
    )
    session.add(article)
    session.flush()
    return article


def seed_chunk(session: Session, article: db.Article, index: int, text: str) -> db.Chunk:
    chunk = db.Chunk(
        article_id=article.id,
        index=index,
        heading_path=f"Energy > S{index}",
        text=text,
        token_count=3,
        embedding=vec(1.0),
    )
    session.add(chunk)
    session.flush()
    return chunk


def seed_formula(
    session: Session, article: db.Article, chunk: db.Chunk, index: int, qid: str | None = None
) -> db.Formula:
    formula = db.Formula(
        article_id=article.id,
        index=index,
        latex=f"x_{index}",
        display=index % 2 == 0,
        section_path="Energy > S",
        position=index,
        wikidata_qid=qid,
    )
    session.add(formula)
    session.flush()
    session.add(db.ChunkFormula(chunk_id=chunk.id, formula_id=formula.id))
    return formula


def seed_citation(
    session: Session,
    article: db.Article,
    key: str,
    title: str | None,
    page: str | None = None,
) -> db.Citation:
    source = (
        db.Source(title=title, authors="A. Author", year=1999, doi=f"10/{key}") if title else None
    )
    if source:
        session.add(source)
        session.flush()
    citation = db.Citation(
        article_id=article.id,
        ref_key=key,
        page=page,
        raw={},
        source_id=source.id if source else None,
    )
    session.add(citation)
    session.flush()
    return citation


def run(
    sessions: sessionmaker[Session], query: str = "energy", top_n: int = 10
) -> list[RetrievedChunk]:
    with sessions() as session:
        return retrieve(
            session, FakeClient(DIM, vector=lambda _: vec(1.0)), FakeReranker(), query, top_n
        )


def test_formula_fields_and_none_chain_ends_at_revision(sessions: sessionmaker[Session]) -> None:
    with sessions.begin() as session:
        article = seed_article(session)
        chunk = seed_chunk(session, article, 0, "energy")
        seed_formula(session, article, chunk, 2)
    [result] = run(sessions)
    [hit] = result.formulas
    assert (hit.index, hit.latex, hit.display, hit.section_path) == (2, "x_2", True, "Energy > S")
    assert hit.wikidata is None
    [link] = hit.provenance
    assert (link.kind, link.source, link.page) == ("none", None, None)
    assert (link.revision.title, link.revision.rev_id) == ("Energy", 7)
    assert (link.revision.url, link.revision.license) == ("https://w/7", "CC BY-SA")
    assert result.revision == link.revision


def test_direct_inherited_and_section_chains_are_ordered(sessions: sessionmaker[Session]) -> None:
    with sessions.begin() as session:
        article = seed_article(session)
        chunk = seed_chunk(session, article, 0, "energy")
        formula = seed_formula(session, article, chunk, 1)
        for key, title, page, kind in [
            ("s", "Zed", None, "section"),
            ("i", "Beta", "9", "inherited"),
            ("d2", "Beta", "5", "direct"),
            ("d1", "Alpha", "12", "direct"),
            ("d3", "Beta", "3", "direct"),
        ]:
            c = seed_citation(session, article, key, title, page)
            session.add(db.FormulaCitation(formula_id=formula.id, citation_id=c.id, kind=kind))
    [result] = run(sessions)
    links = result.formulas[0].provenance
    assert [(x.kind, x.source.title if x.source else None, x.page) for x in links] == [
        ("direct", "Alpha", "12"),
        ("direct", "Beta", "3"),
        ("direct", "Beta", "5"),
        ("inherited", "Beta", "9"),
        ("section", "Zed", None),
    ]
    first = links[0]
    assert first.source is not None
    assert (first.source.authors, first.source.year, first.source.doi) == (
        "A. Author",
        1999,
        "10/d1",
    )
    assert first.revision.rev_id == 7


def test_chunk_citations(sessions: sessionmaker[Session]) -> None:
    with sessions.begin() as session:
        article = seed_article(session)
        chunk = seed_chunk(session, article, 0, "energy")
        for key, title in [("b", "Book B"), ("a", "Book A")]:
            c = seed_citation(session, article, key, title, "4")
            session.add(db.ChunkCitation(chunk_id=chunk.id, citation_id=c.id))
    [result] = run(sessions)
    assert [c.source.title for c in result.citations if c.source] == ["Book A", "Book B"]
    assert all(c.kind == "direct" and c.page == "4" for c in result.citations)
    assert result.citations[0].revision.title == "Energy"
    assert result.formulas == []


def test_wikidata_attached_with_lowest_id(sessions: sessionmaker[Session]) -> None:
    with sessions.begin() as session:
        article = seed_article(session)
        chunk = seed_chunk(session, article, 0, "energy")
        seed_formula(session, article, chunk, 1, "Q1")
        seed_formula(session, article, chunk, 2, "Q404")
        seed_formula(session, article, chunk, 3)
        for latex in ["first", "second"]:
            session.add(
                db.WikidataFormula(
                    article_id=article.id,
                    qid="Q1",
                    latex=latex,
                    symbols={"m": "mass"},
                    references=[{"url": "u"}],
                    license="CC0",
                )
            )
    [result] = run(sessions)
    with_qid, missing, plain = result.formulas
    assert with_qid.wikidata is not None
    assert (with_qid.wikidata.qid, with_qid.wikidata.latex, with_qid.wikidata.license) == (
        "Q1",
        "first",
        "CC0",
    )
    assert with_qid.wikidata.symbols == {"m": "mass"}
    assert with_qid.wikidata.references == [{"url": "u"}]
    assert missing.wikidata is None
    assert plain.wikidata is None


def seed_many(sessions: sessionmaker[Session], n: int) -> None:
    with sessions.begin() as session:
        article = seed_article(session)
        for i in range(n):
            chunk = seed_chunk(session, article, i, f"energy {i}")
            for j in range(2):
                formula = seed_formula(session, article, chunk, i * 2 + j, "Q1")
                c = seed_citation(session, article, f"f{i}{j}", f"Src {i}{j}", "1")
                session.add(
                    db.FormulaCitation(formula_id=formula.id, citation_id=c.id, kind="direct")
                )
            c = seed_citation(session, article, f"c{i}", f"Book {i}")
            session.add(db.ChunkCitation(chunk_id=chunk.id, citation_id=c.id))
        session.add(
            db.WikidataFormula(
                article_id=article.id, qid="Q1", latex="l", symbols={}, references=[], license="CC0"
            )
        )


def count_queries(sessions: sessionmaker[Session], n: int) -> tuple[int, list[RetrievedChunk]]:
    seed_many(sessions, n)
    statements: list[str] = []
    with sessions() as session:
        engine = session.get_bind()
        listener = lambda conn, cur, stmt, *a: statements.append(stmt)  # noqa: E731
        event.listen(engine, "before_cursor_execute", listener)
        try:
            result = retrieve(
                session, FakeClient(DIM, vector=lambda _: vec(1.0)), FakeReranker(), "energy", n
            )
        finally:
            event.remove(engine, "before_cursor_execute", listener)
    return len(statements), result


def test_query_count_is_constant(sessions: sessionmaker[Session]) -> None:
    one, r1 = count_queries(sessions, 1)
    with sessions.begin() as session:
        session.execute(delete(db.Article))
        session.execute(delete(db.Source))
        session.execute(delete(db.WikidataFormula))
    ten, r10 = count_queries(sessions, 10)
    assert (len(r1), len(r10)) == (1, 10)
    assert one == ten


def test_end_to_end_orders_by_rerank_score_with_typed_models(
    sessions: sessionmaker[Session],
) -> None:
    with sessions.begin() as session:
        article = seed_article(session)
        for i, text in enumerate(["energy", "kinetic energy theorem", "kinetic"]):
            seed_chunk(session, article, i, text)
    with sessions() as session:
        result = retrieve(
            session,
            FakeClient(DIM, vector=lambda _: vec(1.0)),
            FakeReranker(),
            "kinetic energy",
            2,
        )
    assert [c.text for c in result] == ["kinetic energy theorem", "energy"]
    assert all(isinstance(c, RetrievedChunk) for c in result)
    assert result[0].rerank_score > result[1].rerank_score
    assert result[0].fused_score > 0
    assert result[0].heading_path == "Energy > S1"
    assert result[0].revision.rev_id == 7


def test_no_results_returns_empty(sessions: sessionmaker[Session]) -> None:
    assert run(sessions) == []

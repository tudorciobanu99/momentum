"""Tests for the pilot report on a small seeded dataset."""

import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from html import escape

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import (
    Article,
    ArticleHtml,
    Citation,
    Formula,
    FormulaCitation,
    Source,
    WikidataFormula,
)
from app.ingestion.report import render_report

MATH = escape(json.dumps({"name": "math", "body": {"extsrc": "x"}}))
REF = escape(json.dumps({"name": "ref", "body": {"id": "mw-reference-text-cite_note-a"}}))
CITE = escape(
    json.dumps(
        {"parts": [{"template": {"target": {"wt": "cite book"}, "params": {"title": {"wt": "T"}}}}]}
    )
)
RICH_HTML = (
    f'<math typeof="mw:Extension/math" data-mw="{MATH}"></math>'
    f'<sup typeof="mw:Extension/ref" data-mw="{REF}"></sup>'
    f'<span typeof="mw:Transclusion" data-mw="{CITE}"></span>'
)


def formula(article_id: int, index: int, latex: str, display: bool, qid: str | None = None):
    return Formula(
        article_id=article_id,
        index=index,
        latex=latex,
        display=display,
        section_path="",
        position=index + 1,
        wikidata_qid=qid,
    )


@pytest.fixture
def db(sessions: sessionmaker[Session]) -> Iterator[sessionmaker[Session]]:
    def clean() -> None:
        with sessions.begin() as s:
            s.execute(delete(Article))
            s.execute(delete(Source))

    clean()
    with sessions.begin() as s:
        a1 = Article(
            title="Alpha",
            rev_id=100,
            url="https://example.org/alpha",
            topic="mechanics",
            license="x",
            fetched_at=datetime(2026, 3, 1, 10, tzinfo=UTC),
        )
        a2 = Article(
            title="Beta",
            rev_id=250,
            url="https://example.org/beta",
            topic="quantum",
            license="x",
            fetched_at=datetime(2026, 3, 5, 23, 30, tzinfo=UTC),
        )
        s.add_all([a1, a2])
        s.flush()
        s.add_all(
            [
                ArticleHtml(article_id=a1.id, html=RICH_HTML),
                ArticleHtml(article_id=a2.id, html="<p></p>"),
            ]
        )
        f = [
            formula(a1.id, 0, "f1", True),
            formula(a1.id, 1, "f2", True),
            formula(a1.id, 2, "f3", False),
            formula(a1.id, 3, "f4", False),
            formula(a2.id, 0, "f5", True, "Q1"),
        ]
        source = Source(authors="Doe, J.", year=1999, title="Orbits", doi="10.1/x")
        s.add_all([*f, source])
        s.flush()
        c1 = Citation(article_id=a1.id, ref_key="a", page="7", raw={}, source_id=source.id)
        c2 = Citation(article_id=a1.id, ref_key="b", raw={})
        s.add_all([c1, c2])
        s.flush()
        s.add_all(
            [
                FormulaCitation(formula_id=f[0].id, citation_id=c1.id, kind="direct"),
                FormulaCitation(formula_id=f[0].id, citation_id=c2.id, kind="section"),
                FormulaCitation(formula_id=f[1].id, citation_id=c1.id, kind="inherited"),
                FormulaCitation(formula_id=f[2].id, citation_id=c1.id, kind="section"),
                WikidataFormula(
                    article_id=a1.id, qid="Q9", latex="y", symbols=[], references=[{}], license="x"
                ),
            ]
        )
    yield sessions
    clean()


def render(sessions: sessionmaker[Session]) -> str:
    with sessions() as s:
        return render_report(s)


def test_run_date_and_revision_range_from_data(db: sessionmaker[Session]) -> None:
    out = render(db)
    assert "- Run date: 2026-03-05" in out
    assert "- Revisions: 100 to 250" in out
    assert "- Articles: 2" in out


def test_output_is_deterministic(db: sessionmaker[Session]) -> None:
    assert render(db) == render(db)


def test_provenance_uses_strongest_kind_and_shares_sum_to_100(db: sessionmaker[Session]) -> None:
    section = render(db).split("### Display formulas")[1].split("### Inline formulas")[0]
    overall = next(line for line in section.splitlines() if line.startswith("| Overall"))
    assert overall == "| Overall | 3 | 1 (33.3%) | 1 (33.3%) | 0 (0.0%) | 1 (33.3%) |"
    for line in section.splitlines():
        if line.startswith("| ") and not line.startswith("| Topic"):
            shares = [float(x) for x in re.findall(r"\(([\d.]+)%\)", line)]
            assert sum(shares) == pytest.approx(100, abs=0.2)


def test_wikidata_coverage_per_topic(db: sessionmaker[Session]) -> None:
    out = render(db)
    assert "| mechanics | 1 | 1 | 100.0% |" in out
    assert "| quantum | 0 | 1 | 0.0% |" in out
    assert "| Overall | 1 | 2 | 50.0% |" in out
    assert "- Wikipedia formulas linked to a Wikidata QID: 1" in out
    assert "- With references: 1" in out
    assert "- With symbols: 0" in out


def test_availability_lists_articles_without_data(db: sessionmaker[Session]) -> None:
    out = render(db)
    assert "| formula LaTeX (math data-mw extsrc) | 1 | 50.0% |" in out
    assert "- Without cite template params: Beta" in out


def test_citation_counts(db: sessionmaker[Session]) -> None:
    out = render(db)
    assert "- Citations: 2" in out
    assert "- With DOI: 1 (50.0%)" in out
    assert "- Without either: 1 (50.0%)" in out
    assert "| Orbits | Doe, J. | 1999 | 1 |" in out


def test_sample_chain_format(db: sessionmaker[Session]) -> None:
    out = render(db)
    assert (
        "- direct: Orbits (Doe, J., 1999) → p. 7 → Alpha https://example.org/alpha → `f1`"
    ) in out
    assert "- inherited: Orbits (Doe, J., 1999) → p. 7 → Alpha" in out
    assert "→ `f3`" in out

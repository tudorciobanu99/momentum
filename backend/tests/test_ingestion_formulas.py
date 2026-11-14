"""Tests for formula extraction and storage; counts were taken from the fixtures with grep."""

import json
from collections.abc import Iterator
from html import escape
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, Formula
from app.ingestion.formulas import extract_formulas
from app.ingestion.service import extract_all
from app.ingestion.store import save_article, save_formulas

FIXTURES = Path(__file__).parent / "fixtures"

# name: (total math elements, display)
# Euler: 32 math elements, 1 display=block, 7 unset inside dd.
# Kirchhoff: 15 math elements, 12 display=block, 1 display=inline, 2 unset in running text.
# Kepler: 39 math elements, 13 unset inside dd.
COUNTS = {
    "euler_rigid_body": (32, 8),
    "kirchhoff_circuit_laws": (15, 12),
    "kepler_problem": (39, 13),
}


def load(name: str) -> str:
    return (FIXTURES / f"{name}.html").read_text(encoding="utf-8")


def page(body: str, title: str = "Test") -> str:
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>"


def math(latex: str, attrs: dict[str, str] | None = None, tag: str = "math") -> str:
    data = {"name": "math", "attrs": attrs or {}, "body": {"extsrc": latex}}
    return (
        f'<{tag} typeof="mw:Extension/math" data-mw="{escape(json.dumps(data))}">'
        f"<mi>ALT</mi></{tag}>"
    )


@pytest.mark.parametrize(("name", "counts"), COUNTS.items())
def test_hand_counted_totals(name: str, counts: tuple[int, int]) -> None:
    records = extract_formulas(load(name))
    total, display = counts
    assert len(records) == total
    assert sum(r.display for r in records) == display
    assert [r.index for r in records] == list(range(total))


def test_latex_from_extsrc_not_mathml() -> None:
    records = extract_formulas(page(math(r"\frac{a}{b}")))
    assert [r.latex for r in records] == [r"\frac{a}{b}"]


def test_latex_in_real_article() -> None:
    records = extract_formulas(load("euler_rigid_body"))
    assert records[0].latex == r"\textbf{F} =  \frac{d\mathbf{p}}{dt}"
    assert all("ALT" not in r.latex and r.latex for r in records)


def test_span_variant_is_found() -> None:
    records = extract_formulas(page(math("x", {"display": "block"}, tag="span")))
    assert [(r.latex, r.display) for r in records] == [("x", True)]


def test_display_attribute() -> None:
    html = page(
        "<p>a "
        + math("b", {"display": "block"})
        + " c "
        + math("d", {"display": "inline"})
        + "</p><dd>"
        + math("e", {"display": "inline"})
        + "</dd>"
    )
    assert [r.display for r in extract_formulas(html)] == [True, False, False]


def test_legacy_dd_is_display() -> None:
    html = page("<dl><dd>" + math("x") + "</dd></dl>")
    assert [r.display for r in extract_formulas(html)] == [True]


def test_standalone_paragraph_is_display() -> None:
    html = page("<p>" + math("x") + "</p><p>see " + math("y") + " here</p>")
    assert [r.display for r in extract_formulas(html)] == [True, False]


def test_dd_with_text_is_inline() -> None:
    html = page("<dl><dd>where " + math("x") + "</dd></dl>")
    assert [r.display for r in extract_formulas(html)] == [False]


def test_math_templates_excluded() -> None:
    html = page('<p><span class="texhtml"><i>x</i><sup>2</sup></span> and ' + math("y") + "</p>")
    assert [r.latex for r in extract_formulas(html)] == ["y"]
    kirchhoff = load("kirchhoff_circuit_laws")
    assert kirchhoff.count('class="texhtml') == 14
    assert len(extract_formulas(kirchhoff)) == 15


def test_section_paths() -> None:
    html = page(
        "<section>"
        + math("lead")
        + "</section><section><h2>One</h2>"
        + math("a")
        + "</section><section><h3>Sub</h3>"
        + math("b")
        + "</section><section><h2>Two</h2>"
        + math("c")
        + "</section>",
        title="Art",
    )
    assert [r.section_path for r in extract_formulas(html)] == [
        "Art",
        "Art > One",
        "Art > One > Sub",
        "Art > Two",
    ]


def test_section_path_in_real_article() -> None:
    paths = {r.section_path for r in extract_formulas(load("euler_rigid_body"))}
    title = "Euler's equations (rigid body dynamics)"
    assert title in paths
    assert f"{title} > Derivation > Principal axes form" in paths


def test_positions_increase_and_are_stable() -> None:
    html = load("kepler_problem")
    first = extract_formulas(html)
    positions = [r.position for r in first]
    assert positions == sorted(set(positions))
    assert extract_formulas(html) == first


@pytest.fixture
def db(sessions: sessionmaker[Session], alembic_config: Config) -> Iterator[sessionmaker[Session]]:
    command.upgrade(alembic_config, "head")
    with sessions.begin() as s:
        s.execute(delete(Article))
    yield sessions
    with sessions.begin() as s:
        s.execute(delete(Article))


def add_article(sessions: sessionmaker[Session], title: str, fixture: str) -> int:
    with sessions.begin() as s:
        save_article(
            s,
            title=title,
            rev_id=1,
            url="u",
            wikidata_qid=None,
            topic="mechanics",
            html=load(fixture),
        )
        return s.scalars(select(Article.id).where(Article.title == title)).one()


def test_store_and_idempotent(db: sessionmaker[Session]) -> None:
    article_id = add_article(db, "Kepler problem", "kepler_problem")
    records = extract_formulas(load("kepler_problem"))
    for _ in range(2):
        with db.begin() as s:
            save_formulas(s, article_id, records)
    with db() as s:
        rows = s.scalars(
            select(Formula).where(Formula.article_id == article_id).order_by(Formula.index)
        ).all()
    assert len(rows) == 39
    assert len({r.id for r in rows}) == 39
    assert [(r.index, r.latex, r.display, r.section_path, r.position) for r in rows] == [
        (r.index, r.latex, r.display, r.section_path, r.position) for r in records
    ]


def test_extract_all_replaces_per_article(db: sessionmaker[Session]) -> None:
    kepler = add_article(db, "Kepler problem", "kepler_problem")
    euler = add_article(db, "Euler", "euler_rigid_body")
    assert extract_all(db) == {"Kepler problem": 39, "Euler": 32}
    assert extract_all(db) == {"Kepler problem": 39, "Euler": 32}
    with db() as s:
        for article_id, n in ((kepler, 39), (euler, 32)):
            found = s.scalars(select(Formula).where(Formula.article_id == article_id)).all()
            assert len(found) == n

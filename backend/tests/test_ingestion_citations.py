"""Tests for citation parsing, formula linking, storage and the provenance chain."""

import json
from collections.abc import Iterator
from html import escape
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, Citation, Formula, FormulaCitation, Source
from app.ingestion.citations import ParsedArticle, normalize_doi, normalize_isbn, parse_citations
from app.ingestion.formulas import extract_formulas
from app.ingestion.provenance import chains_for_source
from app.ingestion.service import citations_all
from app.ingestion.store import save_article, save_citations, save_formulas

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / f"{name}.html").read_text(encoding="utf-8")


def parse(html: str) -> ParsedArticle:
    return parse_citations(html, {r.position: r.display for r in extract_formulas(html)})


def attr(data: dict) -> str:
    return escape(json.dumps(data))


def template(name: str, params: dict[str, str], about: str) -> str:
    data = {
        "parts": [
            {
                "template": {
                    "target": {"wt": name},
                    "params": {k: {"wt": v} for k, v in params.items()},
                    "i": 0,
                }
            }
        ]
    }
    return f'<link typeof="mw:Transclusion" about="#{about}" data-mw="{attr(data)}"/>'


def math(latex: str, display: bool = True) -> str:
    data = {
        "name": "math",
        "attrs": {"display": "block"} if display else {},
        "body": {"extsrc": latex},
    }
    return f'<math typeof="mw:Extension/math" data-mw="{attr(data)}"><mi>x</mi></math>'


def ref(note: str, first: bool = True) -> str:
    data: dict = {"name": "ref", "attrs": {}}
    if first:
        data["body"] = {"id": f"mw-reference-text-cite_note-{note}"}
    return (
        f'<sup class="mw-ref" typeof="mw:Extension/ref" data-mw="{attr(data)}">'
        f'<a href="./T#cite_note-{note}">[1]</a></sup>'
    )


def note(key: str, content: str) -> str:
    return (
        f'<li id="cite_note-{key}"><span id="mw-reference-text-cite_note-{key}" '
        f'class="mw-reference-text">{content}</span></li>'
    )


BOOK = template(
    "cite book ",
    {
        "last": "Smith",
        "first": "Jo",
        "title": "''Classical'' [[mechanics]]",
        "date": "March 2001",
        "isbn": "0-306-40615-2",
        "page": "7",
    },
    "n1",
)
JOURNAL = template(
    "Cite journal",
    {
        "last1": "Doe",
        "first1": "A.",
        "last2": "Roe",
        "title": "Orbits",
        "year": "1999",
        "doi": "https://doi.org/10.1000/ABC",
    },
    "n2",
)
SFN_DATA = {
    "parts": [
        {
            "template": {
                "target": {"wt": "sfn"},
                "params": {"1": {"wt": "Lamb"}, "2": {"wt": "1932"}, "pp": {"wt": "42-43"}},
                "i": 0,
            }
        }
    ]
}
SFN = (
    '<a href="./T#CITEREFLamb1932" typeof="mw:Transclusion" about="#n3" '
    f'data-mw="{attr(SFN_DATA)}">'
    "Lamb 1932, pp. 42-43</a>"
)
BIB = (
    template(
        "cite book",
        {
            "last": "Lamb",
            "first": "Horace",
            "title": "Hydrodynamics",
            "year": "1932",
            "isbn": "978-0-521-45868-9",
        },
        "b1",
    )
    + '<cite id="CITEREFLamb1932" about="#b1">Lamb, Horace (1932). Hydrodynamics.</cite>'
)

SNIPPET = (
    "<html><head><title>Snippet</title></head><body>"
    "<section><h2>Mapping</h2>"
    f"<p>Lead-in.{ref('A')}</p>"
    f"<dl><dd>{math('m1')}{ref('B')}</dd></dl>"
    f"<p>Second lead-in.{ref('C')}{ref('A', first=False)}</p>"
    f"<dl><dd>{math('m2')}</dd></dl>"
    "<p>No refs here.</p>"
    f"<dl><dd>{math('m3')}</dd></dl>"
    "<p>Bare lead-in.</p>"
    f"<dl><dd>{math('m4')}</dd><dd>{ref('B', first=False)}</dd></dl>"
    f"<p>Inline {math('m5', display=False)} in text.{ref('C', first=False)}</p>"
    f"<p>Contains {math('m6')} and more.{ref('A', first=False)}</p>"
    "</section>"
    "<section><h2>Elsewhere</h2><p>Text only.</p>"
    f"<dl><dd>{math('m7')}</dd></dl></section>"
    f"<ol>{note('A', BOOK)}{note('B', SFN)}{note('C', JOURNAL)}</ol>"
    f"<div>{BIB}</div>"
    "</body></html>"
)


def links_by_latex(html: str) -> dict[str, dict[str, str]]:
    records = extract_formulas(html)
    parsed = parse(html)
    return {r.latex: parsed.links.get(r.position, {}) for r in records}


def test_cite_book_fields_from_fixture() -> None:
    citations = {c.ref_key: c for c in parse(load("euler_rigid_body")).citations}
    hsw = citations["cite_note-HSW-1"]
    assert hsw.page == "65"
    assert hsw.isbn == "9780198504214"
    assert hsw.year == 2011
    assert hsw.authors is not None and hsw.authors.startswith("Hitchin, Nigel J.; Segal, Graeme B.")
    assert hsw.title is not None and hsw.title.startswith("Integrable systems")
    assert hsw.raw["template"] == "cite book"
    assert hsw.raw["params"]["isbn"] == "9780198504214"


def test_cite_journal_doi_and_pages_from_fixture() -> None:
    citations = {c.ref_key: c for c in parse(load("kepler_problem")).citations}
    speiser = citations["cite_note-Speiser1996-3"]
    assert (speiser.doi, speiser.isbn, speiser.page) == ("10.1007/bf02327155", None, "103–116")
    goldstein = citations["cite_note-goldstein_1980-1"]
    assert goldstein.isbn == "9780201029697"
    assert goldstein.raw["params"]["publisher"] == "[[Addison-Wesley]]"


def test_reused_named_ref_is_one_citation() -> None:
    parsed = parse(load("kepler_problem"))
    keys = [c.ref_key for c in parsed.citations]
    assert keys.count("cite_note-goldstein_1980-1") == 1
    assert len(keys) == len(set(keys)) == 5
    assert [c.ref_key for c in parse(SNIPPET).citations].count("cite_note-A") == 1


def test_markup_stripped_and_template_fields() -> None:
    by_key = {c.ref_key: c for c in parse(SNIPPET).citations}
    book = by_key["cite_note-A"]
    assert book.title == "Classical mechanics"
    assert (book.authors, book.year, book.page) == ("Smith, Jo", 2001, "7")
    assert book.isbn == "9780306406157"
    journal = by_key["cite_note-C"]
    assert journal.authors == "Doe, A.; Roe"
    assert (journal.year, journal.doi, journal.isbn) == (1999, "10.1000/abc", None)


def test_sfn_resolves_to_bibliography_with_page() -> None:
    by_key = {c.ref_key: c for c in parse(SNIPPET).citations}
    sfn = by_key["cite_note-B"]
    assert sfn.page == "42-43"
    assert (sfn.title, sfn.authors, sfn.year) == ("Hydrodynamics", "Lamb, Horace", 1932)
    assert sfn.isbn == "9780521458689"
    assert sfn.raw["citeref"] == "CITEREFLamb1932"
    assert sfn.raw["bibliography"]["template"] == "cite book"


def test_direct_inherited_none_mapping() -> None:
    links = links_by_latex(SNIPPET)
    assert links["m1"] == {"cite_note-A": "inherited", "cite_note-B": "direct"}
    assert links["m2"] == {"cite_note-C": "inherited", "cite_note-A": "inherited"}
    assert links["m3"] == {}
    assert links["m4"] == {"cite_note-B": "direct"}
    assert links["m5"] == {"cite_note-C": "inherited"}
    assert links["m6"] == {"cite_note-A": "inherited"}
    assert links["m7"] == {}


def test_section_level_inheritance_only_counted() -> None:
    assert parse(SNIPPET).section_extra == 1


def test_mapping_on_fixtures_has_no_direct_links_without_refs() -> None:
    parsed = parse(load("kirchhoff_circuit_laws"))
    assert parsed.links == {}
    assert len(parsed.citations) == 3


def test_inline_inherits_paragraph_in_fixture() -> None:
    html = load("euler_rigid_body")
    records = extract_formulas(html)
    parsed = parse(html)
    linked = {r.latex: parsed.links[r.position] for r in records if r.position in parsed.links}
    assert linked[r"\mathfrak{so}(3)"] == {
        "cite_note-HSW-1": "inherited",
        "cite_note-2": "inherited",
    }
    assert not next(r for r in records if r.latex == r"\mathfrak{so}(3)").display


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.1000/ABC", "10.1000/abc"),
        ("https://doi.org/10.1000/ABC", "10.1000/abc"),
        ("http://dx.doi.org/10.1000/Abc ", "10.1000/abc"),
        ("doi:10.1000/abc", "10.1000/abc"),
        ("not a doi", None),
        (None, None),
    ],
)
def test_normalize_doi(value: str | None, expected: str | None) -> None:
    assert normalize_doi(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0-306-40615-2", "9780306406157"),
        ("978-0-306-40615-7", "9780306406157"),
        ("9780306406157", "9780306406157"),
        ("0-8044-2957-X", "9780804429573"),
        ("0306406153", None),
        ("0-306-40615-2; 978-0-521-45868-9", "9780306406157"),
        ("garbage", None),
    ],
)
def test_normalize_isbn(value: str, expected: str | None) -> None:
    assert normalize_isbn(value) == expected


@pytest.fixture
def db(sessions: sessionmaker[Session], alembic_config: Config) -> Iterator[sessionmaker[Session]]:
    command.upgrade(alembic_config, "head")
    with sessions.begin() as s:
        s.execute(delete(Article))
        s.execute(delete(Source))
    yield sessions
    with sessions.begin() as s:
        s.execute(delete(Article))
        s.execute(delete(Source))


def add_article(sessions: sessionmaker[Session], title: str, html: str) -> int:
    with sessions.begin() as s:
        save_article(
            s,
            title=title,
            rev_id=1,
            url=f"https://example.org/{title}",
            wikidata_qid=None,
            topic="mechanics",
            html=html,
        )
        article_id = s.scalars(select(Article.id).where(Article.title == title)).one()
        save_formulas(s, article_id, extract_formulas(html))
        return article_id


def counts(sessions: sessionmaker[Session]) -> tuple[int, int, int]:
    with sessions() as s:
        return (
            s.scalar(select(func.count(Citation.id))) or 0,
            s.scalar(select(func.count(Source.id))) or 0,
            s.scalar(select(func.count()).select_from(FormulaCitation)) or 0,
        )


def test_store_is_idempotent(db: sessionmaker[Session]) -> None:
    add_article(db, "Snippet", SNIPPET)
    first = citations_all(db)
    stored = counts(db)
    second = citations_all(db)
    assert counts(db) == stored == (3, 3, 7)
    assert first == second
    assert (first.display_direct, first.display_inherited, first.display_none) == (2, 2, 2)
    assert (first.inline_linked, first.section_extra) == (1, 1)


def test_sources_deduped_across_articles(db: sessionmaker[Session]) -> None:
    add_article(db, "One", SNIPPET)
    add_article(
        db,
        "Two",
        SNIPPET.replace("10.1000/ABC", "10.1000/abc").replace("0-306-40615-2", "978-0-306-40615-7"),
    )
    citations_all(db)
    citations, sources, _ = counts(db)
    assert (citations, sources) == (6, 3)
    with db() as s:
        per_source = (
            s.execute(
                select(func.count(func.distinct(Citation.article_id))).group_by(Citation.source_id)
            )
            .scalars()
            .all()
        )
    assert per_source == [2, 2, 2]


def test_citation_without_doi_or_isbn_has_no_source(db: sessionmaker[Session]) -> None:
    html = SNIPPET.replace("https://doi.org/10.1000/ABC", "")
    add_article(db, "NoIds", html)
    citations_all(db)
    with db() as s:
        journal = s.scalars(select(Citation).where(Citation.ref_key == "cite_note-C")).one()
    assert journal.source_id is None
    assert journal.raw["title"] == "Orbits"
    assert journal.raw["authors"] == "Doe, A.; Roe"
    assert journal.raw["year"] == 1999


def test_chain_from_source_to_formula(db: sessionmaker[Session]) -> None:
    add_article(db, "Snippet", SNIPPET)
    citations_all(db)
    with db() as s:
        source_id = s.scalars(select(Source.id).where(Source.title == "Hydrodynamics")).one()
        chains = chains_for_source(s, source_id)
    assert [
        (c.source_title, c.page, c.article_title, c.rev_id, c.latex, c.kind) for c in chains
    ] == [
        ("Hydrodynamics", "42-43", "Snippet", 1, "m1", "direct"),
        ("Hydrodynamics", "42-43", "Snippet", 1, "m4", "direct"),
    ]
    assert chains[0].url == "https://example.org/Snippet"


def test_citations_replaced_per_article(db: sessionmaker[Session]) -> None:
    article_id = add_article(db, "Snippet", SNIPPET)
    with db.begin() as s:
        save_citations(s, article_id, parse(SNIPPET))
        save_citations(s, article_id, ParsedArticle())
    assert counts(db)[0] == 0
    assert counts(db)[1] == 3
    with db() as s:
        assert s.scalar(select(func.count(Formula.id))) == 7

"""Tests for article chunking and chunk storage."""

import json
from collections.abc import Iterator
from html import escape
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import (
    Article,
    Chunk,
    ChunkCitation,
    ChunkFormula,
)
from app.ingestion.chunks import MAX_TOKENS, MIN_TOKENS, ChunkRecord, chunk_article, count_tokens
from app.ingestion.citations import parse_citations
from app.ingestion.formulas import extract_formulas
from app.ingestion.service import chunks_all
from app.ingestion.store import save_article, save_chunks, save_citations, save_formulas

FIXTURES = Path(__file__).parent / "fixtures"
NAMES = ["euler_rigid_body", "kepler_problem", "kirchhoff_circuit_laws"]


def load(name: str) -> str:
    return (FIXTURES / f"{name}.html").read_text(encoding="utf-8")


def paragraphs(chunk: ChunkRecord) -> list[str]:
    return chunk.text.removeprefix(chunk.heading_path).strip().split("\n\n")


def by_section(chunks: list[ChunkRecord]) -> dict[str, list[ChunkRecord]]:
    sections: dict[str, list[ChunkRecord]] = {}
    for c in chunks:
        sections.setdefault(c.heading_path, []).append(c)
    return sections


def math(latex: str, display: bool = True) -> str:
    data = {
        "name": "math",
        "attrs": {"display": "block"} if display else {},
        "body": {"extsrc": latex},
    }
    return (
        f'<math typeof="mw:Extension/math" data-mw="{escape(json.dumps(data))}"><mi>x</mi></math>'
    )


def ref(note: str, first: bool = True) -> str:
    data: dict = {"name": "ref", "attrs": {}}
    if first:
        data["body"] = {"id": f"mw-reference-text-cite_note-{note}"}
    return (
        f'<sup class="mw-ref" typeof="mw:Extension/ref" data-mw="{escape(json.dumps(data))}">'
        f'<a href="./T#cite_note-{note}">[1]</a></sup>'
    )


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n)) + "."


def page(body: str, notes: str = "") -> str:
    return f"<html><head><title>Snippet</title></head><body>{body}<ol>{notes}</ol></body></html>"


NOTES = "".join(
    f'<li id="cite_note-{k}"><span id="mw-reference-text-cite_note-{k}">Note {k}</span></li>'
    for k in "ABC"
)


@pytest.mark.parametrize("name", NAMES)
def test_text_starts_with_heading_path(name: str) -> None:
    chunks = chunk_article(load(name))
    assert chunks
    for c in chunks:
        assert c.text.startswith(c.heading_path + "\n\n")
        assert c.heading_path.split(" > ")[0] == chunks[0].heading_path.split(" > ")[0]
        assert c.token_count == count_tokens(c.text)
    assert any(" > " in c.heading_path for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("name", NAMES)
def test_size_bounds_with_equation_exception(name: str) -> None:
    sections = by_section(chunk_article(load(name)))
    for chunks in sections.values():
        for c in chunks:
            if "$$" not in c.text:
                assert c.token_count <= MAX_TOKENS
            if len(chunks) > 1 and c.token_count < MIN_TOKENS:
                assert any("$$" in other.text for other in chunks)


def test_short_section_is_one_chunk_and_long_section_splits() -> None:
    html = page(
        f"<section><h2>Short</h2><p>{words(40)}</p></section>"
        f"<section><h2>Long</h2>{''.join(f'<p>{words(120, f"p{i}x")}</p>' for i in range(12))}"
        "</section>"
    )
    sections = by_section(chunk_article(html))
    assert len(sections["Snippet > Short"]) == 1
    assert sections["Snippet > Short"][0].token_count < MIN_TOKENS
    long = sections["Snippet > Long"]
    assert len(long) > 1
    assert all(MIN_TOKENS <= c.token_count <= MAX_TOKENS for c in long)


def test_long_paragraph_is_cut_at_sentences() -> None:
    sentence = words(30, "s")
    html = page(f"<section><h2>A</h2><p>{' '.join([sentence] * 40)}</p></section>")
    chunks = chunk_article(html)
    assert len(chunks) > 1
    assert all(c.token_count <= MAX_TOKENS for c in chunks)


def test_overlap_is_last_paragraph_of_previous_chunk() -> None:
    pairs = 0
    for name in NAMES:
        for chunks in by_section(chunk_article(load(name))).values():
            for earlier, later in zip(chunks, chunks[1:], strict=False):
                assert paragraphs(later)[0] == paragraphs(earlier)[-1]
                pairs += 1
    assert pairs


def test_overlap_in_synthetic_section() -> None:
    html = page(
        f"<section><h2>Long</h2>{''.join(f'<p>{words(120, f"p{i}x")}</p>' for i in range(12))}"
        "</section>"
    )
    chunks = chunk_article(html)
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert paragraphs(later)[0] == paragraphs(earlier)[-1]


@pytest.mark.parametrize("name", NAMES)
def test_display_equation_keeps_its_lead_in(name: str) -> None:
    for c in chunk_article(load(name)):
        for i, paragraph in enumerate(paragraphs(c)):
            if paragraph.startswith("$$"):
                assert i > 0 or c.index == 0 or c.heading_path != paragraphs(c)[0]


def test_equation_stays_with_lead_in_across_chunk_boundary() -> None:
    for fill in range(2, 6):
        lead = f"<p>Lead-in {words(60, 'l')} so that:</p>"
        body = "".join(f"<p>{words(110, f'p{i}x')}</p>" for i in range(fill))
        html = page(
            f"<section><h2>S</h2>{body}{lead}<dl><dd>{math('Z=1')}</dd></dl>{body}</section>"
        )
        for c in chunk_article(html):
            if "$$Z=1$$" in c.text:
                assert "so that:\n$$Z=1$$" in c.text


def test_oversized_equation_unit_exceeds_max_and_stays_whole() -> None:
    html = page(
        f"<section><h2>S</h2><p>{words(400, 'a')}</p>"
        f"<p>Lead-in {words(150, 'b')} here:</p><dl><dd>{math('Q')}</dd></dl></section>"
    )
    chunks = chunk_article(html)
    holder = [c for c in chunks if "$$Q$$" in c.text]
    assert holder and "here:\n$$Q$$" in holder[0].text


def test_exact_formula_and_citation_ids_and_markup() -> None:
    html = page(
        "<section><h2>S</h2>"
        f"<p>Intro {math('i1', display=False)} text.{ref('A')}</p>"
        f"<p>Second {math('i2', display=False)} here:{ref('B')}</p>"
        f"<dl><dd>{math('D1')}</dd></dl>"
        f"<p>{words(200, 'x')}{ref('C')}</p>"
        f"<p>{words(200, 'y')} {math('i3', display=False)}</p>"
        f"<p>{words(200, 'z')}</p>"
        "</section>",
        NOTES,
    )
    formulas = {f.latex: f.index for f in extract_formulas(html)}
    chunks = chunk_article(html)
    assert len(chunks) > 1
    first = chunks[0]
    assert "$i1$" in first.text and "$i2$" in first.text and "\n$$D1$$" in first.text
    assert "[1]" not in first.text
    for c in chunks:
        expected = tuple(
            i
            for latex, i in formulas.items()
            if f"${latex}$" in c.text.replace("$$", "") or f"$${latex}$$" in c.text
        )
        assert sorted(c.formula_indices) == sorted(expected)
        assert set(c.citation_keys) == {
            (f"cite_note-{k}", None)
            for k, text in (("A", "Intro"), ("B", "Second"), ("C", "x0"))
            if text in c.text
        }
    assert {formulas["i1"], formulas["i2"], formulas["D1"]} <= set(first.formula_indices)
    assert {("cite_note-A", None), ("cite_note-B", None)} <= set(first.citation_keys)


@pytest.mark.parametrize("name", NAMES)
def test_ids_match_stored_extraction_and_text(name: str) -> None:
    html = load(name)
    formulas = {f.index: f for f in extract_formulas(html)}
    keys = {
        (c.ref_key, c.page)
        for c in parse_citations(html, {f.position: f.display for f in formulas.values()}).citations
    }
    chunks = chunk_article(html)
    for c in chunks:
        for i in c.formula_indices:
            latex = " ".join(formulas[i].latex.split())
            wrapped = f"$${latex}$$" if formulas[i].display else f"${latex}$"
            assert wrapped in " ".join(c.text.split())
        assert set(c.citation_keys) <= keys
    assert any(c.formula_indices for c in chunks) and any(c.citation_keys for c in chunks)
    assert not any("[1]" in c.text or "[2]" in c.text for c in chunks)


def test_skips_non_content() -> None:
    html = page(
        '<section><div class="hatnote">Hatnote text</div><p>Body text.</p>'
        "<table><tr><td>Infobox cell</td></tr></table>"
        '<div class="navbox">Navbox text</div></section>'
        "<section><h2>See also</h2><p>Skip one</p>"
        "<section><h3>Nested</h3><p>Skip nested</p></section></section>"
        "<section><h2>Keep</h2><p>Kept text.</p></section>"
        "<section><h2>References</h2><p>Skip refs</p></section>"
        "<section><h2>Notes</h2><p>Skip notes</p></section>"
        "<section><h2>Further reading</h2><p>Skip reading</p></section>"
        "<section><h2>External links</h2><p>Skip links</p></section>"
        "<section><h2>Bibliography</h2><p>Skip bib</p></section>"
    )
    text = "\n".join(c.text for c in chunk_article(html))
    assert "Body text." in text and "Kept text." in text
    for gone in ("Hatnote", "Infobox", "Navbox", "Skip"):
        assert gone not in text


@pytest.fixture
def db(sessions: sessionmaker[Session], alembic_config: Config) -> Iterator[sessionmaker[Session]]:
    command.upgrade(alembic_config, "head")
    with sessions.begin() as s:
        s.execute(delete(Article))
    yield sessions
    with sessions.begin() as s:
        s.execute(delete(Article))


def store(sessions: sessionmaker[Session], name: str, rev_id: int = 7) -> int:
    html = load(name)
    with sessions.begin() as s:
        save_article(
            s,
            title=name,
            rev_id=rev_id,
            url=f"https://example.org/{name}",
            wikidata_qid=None,
            topic="mechanics",
            html=html,
        )
        article_id = s.scalars(select(Article.id).where(Article.title == name)).one()
        records = extract_formulas(html)
        save_formulas(s, article_id, records)
        save_citations(
            s, article_id, parse_citations(html, {r.position: r.display for r in records})
        )
        return article_id


def test_store_links_chunks_to_revision_formulas_and_citations(
    db: sessionmaker[Session],
) -> None:
    article_id = store(db, "kepler_problem")
    records = chunk_article(load("kepler_problem"))
    with db.begin() as s:
        save_chunks(s, article_id, records)
    with db() as s:
        rows = s.scalars(select(Chunk).where(Chunk.article_id == article_id)).all()
        assert len(rows) == len(records)
        assert s.scalar(select(Article.rev_id).where(Article.id == article_id)) == 7
        for row, record in zip(sorted(rows, key=lambda r: r.index), records, strict=True):
            assert row.text == record.text and row.token_count == record.token_count
            assert sorted(f.index for f in row.formulas) == sorted(record.formula_indices)
            assert {(c.ref_key, c.page) for c in row.citations} == set(record.citation_keys)
            assert all(f.article_id == article_id for f in row.formulas)


def test_store_is_idempotent(db: sessionmaker[Session]) -> None:
    store(db, "kepler_problem")
    first = chunks_all(db)
    with db() as s:
        counts = [
            s.scalar(select(func.count()).select_from(t))
            for t in (Chunk, ChunkFormula, ChunkCitation)
        ]
        ids = set(s.scalars(select(Chunk.id)))
    assert chunks_all(db) == first
    with db() as s:
        again = [
            s.scalar(select(func.count()).select_from(t))
            for t in (Chunk, ChunkFormula, ChunkCitation)
        ]
        assert again == counts
        assert not ids & set(s.scalars(select(Chunk.id)))

"""Pilot report: Markdown summary of what ingestion stored, computed from the database."""

import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC
from typing import Any

from selectolax.lexbor import LexborHTMLParser
from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app.db.models import (
    Article,
    ArticleHtml,
    Citation,
    Formula,
    FormulaCitation,
    Source,
    WikidataFormula,
)
from app.ingestion.citations import DIRECT, INHERITED, SECTION
from app.ingestion.formulas import MATH_TYPEOF

KINDS = (DIRECT, INHERITED, SECTION, "none")
RANK = {DIRECT: 0, INHERITED: 1, SECTION: 2}
TOP_SOURCES = 10


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "0.0%"


def _cell(value: object) -> str:
    return "" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return [*lines, ""]


def _data_mw(html: str, selector: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for node in LexborHTMLParser(html).css(selector):
        try:
            found.append(json.loads(node.attributes["data-mw"] or "{}"))
        except ValueError:
            continue
    return found


def _has_latex(html: str) -> bool:
    return any(
        str((d.get("body") or {}).get("extsrc") or "").strip()
        for d in _data_mw(html, f'[typeof="{MATH_TYPEOF}"][data-mw]')
    )


def _has_ref_details(html: str) -> bool:
    return any(d.get("body") for d in _data_mw(html, 'sup[typeof~="mw:Extension/ref"][data-mw]'))


def _has_cite_params(html: str) -> bool:
    for data in _data_mw(html, '[typeof~="mw:Transclusion"][data-mw]'):
        for part in data.get("parts", []):
            template = part.get("template") if isinstance(part, dict) else None
            if not template:
                continue
            name = " ".join(template["target"].get("wt", "").replace("_", " ").lower().split())
            if (name.startswith("cite ") or name == "citation") and template.get("params"):
                return True
    return False


def _summary(articles: Sequence[Article]) -> list[str]:
    topics = sorted({a.topic for a in articles})
    revs = [a.rev_id for a in articles]
    run = max(a.fetched_at for a in articles).astimezone(UTC).date() if articles else None
    return [
        "## Summary",
        "",
        f"- Articles: {len(articles)}",
        f"- Topics: {len(topics)} ({', '.join(topics)})",
        f"- Revisions: {min(revs)} to {max(revs)}" if revs else "- Revisions: none",
        f"- Run date: {run}" if run else "- Run date: none",
        "",
    ]


def _availability(session: Session, articles: Sequence[Article]) -> list[str]:
    html = {
        article_id: body
        for article_id, body in session.execute(
            select(ArticleHtml.article_id, ArticleHtml.html).order_by(ArticleHtml.article_id)
        )
    }
    checks = {
        "formula LaTeX (math data-mw extsrc)": _has_latex,
        "reference details (ref data-mw)": _has_ref_details,
        "cite template params": _has_cite_params,
    }
    rows: list[list[object]] = []
    missing: list[str] = []
    for label, check in checks.items():
        ok = {a.id for a in articles if check(html.get(a.id, ""))}
        rows.append([label, len(ok), _pct(len(ok), len(articles))])
        titles = [a.title for a in articles if a.id not in ok]
        missing.append(f"- Without {label}: {', '.join(titles) if titles else 'none'}")
    return [
        "## Data availability",
        "",
        *_table(["Exposed", "Articles", "Share"], rows),
        *missing,
        "",
    ]


def _formulas(rows: Sequence[tuple[str, bool, str]]) -> list[str]:
    by_topic: dict[str, Counter[bool]] = {}
    for topic, display, _ in rows:
        by_topic.setdefault(topic, Counter())[display] += 1
    table = [[t, c[True], c[False], c[True] + c[False]] for t, c in sorted(by_topic.items())]
    display = sum(1 for _, d, _ in rows if d)
    table.append(["Total", display, len(rows) - display, len(rows)])
    return [
        "## Formulas",
        "",
        f"- Total: {len(rows)} (display {display}, inline {len(rows) - display})",
        "",
        *_table(["Topic", "Display", "Inline", "Total"], table),
    ]


def _provenance(rows: Sequence[tuple[str, bool, str]]) -> list[str]:
    lines = ["## Provenance", "", "Strongest link kind per formula.", ""]
    for label, display in (("Display formulas", True), ("Inline formulas", False)):
        subset = [(t, k) for t, d, k in rows if d is display]
        groups: dict[str, Counter[str]] = {}
        for topic, kind in subset:
            groups.setdefault(topic, Counter())[kind] += 1
        table: list[list[object]] = []
        for name, counts in [*sorted(groups.items()), ("Overall", Counter(k for _, k in subset))]:
            total = sum(counts.values())
            table.append([name, total, *(f"{counts[k]} ({_pct(counts[k], total)})" for k in KINDS)])
        lines += [f"### {label}", "", *_table(["Topic", "Total", *KINDS], table)]
    return lines


def _sources(session: Session) -> list[str]:
    def count(*where: Any) -> int:
        return session.scalar(select(func.count(Citation.id)).outerjoin(Source).where(*where)) or 0

    total = count()
    doi = count(Source.doi.is_not(None))
    isbn = count(Source.isbn.is_not(None))
    neither = count(Citation.source_id.is_(None))
    sources = session.scalar(select(func.count(distinct(Citation.source_id)))) or 0
    articles = func.count(distinct(Citation.article_id))
    top = session.execute(
        select(Source.title, Source.authors, Source.year, articles)
        .join(Citation, Citation.source_id == Source.id)
        .group_by(Source.id)
        .order_by(articles.desc(), Source.id)
        .limit(TOP_SOURCES)
    ).all()
    return [
        "## Citations and sources",
        "",
        f"- Citations: {total}",
        f"- With DOI: {doi} ({_pct(doi, total)})",
        f"- With ISBN: {isbn} ({_pct(isbn, total)})",
        f"- Without either: {neither} ({_pct(neither, total)})",
        f"- Distinct sources cited: {sources}",
        "",
        f"### Top {TOP_SOURCES} sources by articles citing them",
        "",
        *_table(["Title", "Authors", "Year", "Articles"], [list(r) for r in top]),
    ]


def _wikidata(session: Session, articles: Sequence[Article]) -> list[str]:
    with_item = set(session.scalars(select(WikidataFormula.article_id).distinct()))
    topics: dict[str, list[int]] = {}
    for a in articles:
        counts = topics.setdefault(a.topic, [0, 0])
        counts[0] += 1
        counts[1] += a.id in with_item
    table: list[list[object]] = [
        [t, c[1], c[0], _pct(c[1], c[0])] for t, c in sorted(topics.items())
    ]
    table.append(["Overall", len(with_item), len(articles), _pct(len(with_item), len(articles))])

    def count(*where: Any) -> int:
        return session.scalar(select(func.count(WikidataFormula.id)).where(*where)) or 0

    linked = session.scalar(select(func.count(Formula.id)).where(Formula.wikidata_qid.is_not(None)))
    return [
        "## Wikidata",
        "",
        "Articles with a defining formula on Wikidata.",
        "",
        *_table(["Topic", "With formula", "Articles", "Share"], table),
        f"- Wikidata formulas: {count()}",
        f"- With references: {count(func.jsonb_array_length(WikidataFormula.references) > 0)}",
        f"- With symbols: {count(func.jsonb_array_length(WikidataFormula.symbols) > 0)}",
        f"- Wikipedia formulas linked to a Wikidata QID: {linked}",
        "",
    ]


def _samples(session: Session) -> list[str]:
    lines = [
        "## Sample chains",
        "",
        "Lowest formula id per kind, among citations with a source.",
        "",
    ]
    for kind in (DIRECT, INHERITED, SECTION):
        row = session.execute(
            select(Source, Citation.page, Article, Formula)
            .join(Citation, Citation.source_id == Source.id)
            .join(Article, Article.id == Citation.article_id)
            .join(FormulaCitation, FormulaCitation.citation_id == Citation.id)
            .join(Formula, Formula.id == FormulaCitation.formula_id)
            .where(FormulaCitation.kind == kind)
            .order_by(Formula.id, Citation.id)
            .limit(1)
        ).first()
        if row is None:
            lines.append(f"- {kind}: none")
            continue
        source, page, article, formula = row
        by = f" ({source.authors}, {source.year})" if source.authors and source.year else ""
        at = f" page {page}" if page else ""
        lines.append(
            f"- {kind}: {_cell(source.title)}{_cell(by)} → citation{_cell(at)} → "
            f"{article.title} {article.url} → `{formula.latex}`"
        )
    return [*lines, ""]


def render_report(session: Session) -> str:
    """Renders the pilot report from the database.

    Args:
        session: Open session.

    Returns:
        Markdown. Identical for identical data: ordering is explicit and the run date is the
        latest `articles.fetched_at`.
    """
    articles = list(session.scalars(select(Article).order_by(Article.id)))
    strongest: dict[int, str] = {}
    links = session.execute(
        select(FormulaCitation.formula_id, FormulaCitation.kind).order_by(
            FormulaCitation.formula_id, FormulaCitation.kind, FormulaCitation.citation_id
        )
    )
    for formula_id, kind in links:
        if formula_id not in strongest or RANK[kind] < RANK[strongest[formula_id]]:
            strongest[formula_id] = kind
    rows = [
        (topic, display, strongest.get(formula_id, "none"))
        for formula_id, topic, display in session.execute(
            select(Formula.id, Article.topic, Formula.display)
            .join(Article, Article.id == Formula.article_id)
            .order_by(Formula.id)
        )
    ]
    parts = [
        ["# Pilot report", ""],
        _summary(articles),
        _availability(session, articles),
        _formulas(rows),
        _provenance(rows),
        _sources(session),
        _wikidata(session, articles),
        _samples(session),
    ]
    return "\n".join(line for part in parts for line in part).rstrip() + "\n"

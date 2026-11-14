"""Writing fetched articles to the database."""

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import (
    Article,
    ArticleHtml,
    Citation,
    Formula,
    FormulaCitation,
    Source,
)
from app.ingestion.citations import ParsedArticle, ParsedCitation
from app.ingestion.formulas import FormulaRecord

LICENSE = "CC BY-SA 4.0"


def save_article(
    session: Session,
    *,
    title: str,
    rev_id: int,
    url: str,
    wikidata_qid: str | None,
    topic: str,
    html: str,
) -> bool:
    """Stores an article revision and its HTML, skipping one already stored.

    Args:
        session: Open session; the caller owns the transaction.
        title: Page title.
        rev_id: Revision id.
        url: Permanent URL of the revision.
        wikidata_qid: Wikidata item id, if the page has one.
        topic: Topic label.
        html: Revision HTML.

    Returns:
        True if a row was inserted, False if (title, rev_id) already existed.
    """
    article_id = session.execute(
        insert(Article)
        .values(
            title=title,
            rev_id=rev_id,
            url=url,
            wikidata_qid=wikidata_qid,
            topic=topic,
            license=LICENSE,
        )
        .on_conflict_do_nothing(index_elements=["title", "rev_id"])
        .returning(Article.id)
    ).scalar_one_or_none()
    if article_id is None:
        return False
    session.add(ArticleHtml(article_id=article_id, html=html))
    session.flush()
    return True


def is_stored(session: Session, title: str, rev_id: int) -> bool:
    """Tells whether a (title, rev_id) pair is already stored.

    Args:
        session: Open session.
        title: Page title.
        rev_id: Revision id.

    Returns:
        True if the article row exists.
    """
    return (
        session.scalar(select(Article.id).where(Article.title == title, Article.rev_id == rev_id))
        is not None
    )


def save_formulas(session: Session, article_id: int, records: list[FormulaRecord]) -> None:
    """Replaces an article's formulas with the given records.

    Args:
        session: Open session; the caller owns the transaction.
        article_id: Article the formulas belong to.
        records: Extracted formulas.
    """
    session.execute(delete(Formula).where(Formula.article_id == article_id))
    session.add_all(
        Formula(
            article_id=article_id,
            index=r.index,
            latex=r.latex,
            display=r.display,
            section_path=r.section_path,
            position=r.position,
            wikidata_qid=r.wikidata_qid,
        )
        for r in records
    )
    session.flush()


def _upsert_source(session: Session, c: ParsedCitation) -> int | None:
    if c.doi:
        key, where = "doi", "doi IS NOT NULL"
    elif c.isbn:
        key, where = "isbn", "isbn IS NOT NULL"
    else:
        return None
    stmt = insert(Source).values(
        authors=c.authors, year=c.year, title=c.title, doi=c.doi, isbn=c.isbn
    )
    excluded = stmt.excluded
    stmt = stmt.on_conflict_do_update(
        index_elements=[key],
        index_where=text(where),
        set_={
            "authors": func.coalesce(Source.authors, excluded.authors),
            "year": func.coalesce(Source.year, excluded.year),
            "title": func.coalesce(Source.title, excluded.title),
        },
    ).returning(Source.id)
    return session.execute(stmt).scalar_one()


def save_citations(session: Session, article_id: int, parsed: ParsedArticle) -> None:
    """Replaces an article's citations and formula links; sources are upserted, never deleted.

    Sources are deduplicated by DOI, else by ISBN. A citation with neither has no source.
    Formulas are matched to links by position, so the article's formulas must be stored first.

    Args:
        session: Open session; the caller owns the transaction.
        article_id: Article the citations belong to.
        parsed: Output of `parse_citations`.
    """
    session.execute(delete(Citation).where(Citation.article_id == article_id))
    by_key: dict[str, Citation] = {}
    for c in parsed.citations:
        citation = Citation(
            article_id=article_id,
            ref_key=c.ref_key,
            page=c.page,
            raw={**c.raw, "authors": c.authors, "year": c.year, "title": c.title},
            source_id=_upsert_source(session, c),
        )
        by_key[c.ref_key] = citation
        session.add(citation)
    session.flush()
    formula_ids = {
        position: formula_id
        for position, formula_id in session.execute(
            select(Formula.position, Formula.id).where(Formula.article_id == article_id)
        )
    }
    session.add_all(
        FormulaCitation(formula_id=formula_ids[position], citation_id=by_key[key].id, kind=kind)
        for position, links in parsed.links.items()
        if position in formula_ids
        for key, kind in links.items()
    )
    session.flush()

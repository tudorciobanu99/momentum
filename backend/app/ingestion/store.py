"""Writing fetched articles to the database."""

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import (
    Article,
    ArticleHtml,
    Chunk,
    ChunkCitation,
    ChunkFormula,
    Citation,
    Formula,
    FormulaCitation,
    Source,
)
from app.ingestion.chunks import ChunkRecord
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


def _upsert_source(session: Session, c: ParsedCitation) -> tuple[int | None, bool]:
    if not (c.doi or c.isbn):
        return None, False
    by_doi = session.scalar(select(Source).where(Source.doi == c.doi)) if c.doi else None
    by_isbn = session.scalar(select(Source).where(Source.isbn == c.isbn)) if c.isbn else None
    merged = False
    if by_doi and by_isbn and by_doi.id != by_isbn.id:
        session.execute(
            update(Citation).where(Citation.source_id == by_isbn.id).values(source_id=by_doi.id)
        )
        for name in ("authors", "year", "title"):
            if getattr(by_doi, name) is None:
                setattr(by_doi, name, getattr(by_isbn, name))
        session.delete(by_isbn)
        session.flush()
        merged = True
    source = by_doi or by_isbn
    if source is None:
        source = Source(authors=c.authors, year=c.year, title=c.title, doi=c.doi, isbn=c.isbn)
        session.add(source)
    else:
        source.authors = source.authors or c.authors
        source.year = source.year or c.year
        source.title = source.title or c.title
        source.doi = source.doi or c.doi
        source.isbn = source.isbn or c.isbn
    session.flush()
    return source.id, merged


def save_citations(session: Session, article_id: int, parsed: ParsedArticle) -> int:
    """Replaces an article's citations and formula links; sources are upserted, never deleted.

    A citation matches an existing source by DOI or ISBN and fills its missing fields. If its
    DOI and ISBN match different sources they are merged into the DOI one. A citation with
    neither has no source.
    Formulas are matched to links by position, so the article's formulas must be stored first.

    Args:
        session: Open session; the caller owns the transaction.
        article_id: Article the citations belong to.
        parsed: Output of `parse_citations`.

    Returns:
        Number of sources merged into another.
    """
    session.execute(delete(Citation).where(Citation.article_id == article_id))
    by_key: dict[str, Citation] = {}
    merges = 0
    for c in parsed.citations:
        source_id, merged = _upsert_source(session, c)
        merges += merged
        citation = Citation(
            article_id=article_id,
            ref_key=c.ref_key,
            page=c.page,
            raw={**c.raw, "authors": c.authors, "year": c.year, "title": c.title},
            source_id=source_id,
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
    return merges


def save_chunks(session: Session, article_id: int, records: list[ChunkRecord]) -> None:
    """Replaces an article's chunks and their formula and citation links.

    Formulas are matched by index and citations by (ref_key, page), so both must already be
    stored for the article.

    Args:
        session: Open session; the caller owns the transaction.
        article_id: Article the chunks belong to.
        records: Output of `chunk_article`.

    Raises:
        LookupError: A chunk refers to a formula or citation that is not stored.
    """
    session.execute(delete(Chunk).where(Chunk.article_id == article_id))
    formula_ids = {
        index: formula_id
        for index, formula_id in session.execute(
            select(Formula.index, Formula.id).where(Formula.article_id == article_id)
        )
    }
    citation_ids = {
        (ref_key, page): citation_id
        for ref_key, page, citation_id in session.execute(
            select(Citation.ref_key, Citation.page, Citation.id).where(
                Citation.article_id == article_id
            )
        )
    }
    wanted_formulas = {i for r in records for i in r.formula_indices}
    wanted_citations = {k for r in records for k in r.citation_keys}
    if not wanted_formulas <= formula_ids.keys() or not wanted_citations <= citation_ids.keys():
        raise LookupError("formulas or citations missing")
    chunks = [
        Chunk(
            article_id=article_id,
            index=r.index,
            heading_path=r.heading_path,
            text=r.text,
            token_count=r.token_count,
        )
        for r in records
    ]
    session.add_all(chunks)
    session.flush()
    session.add_all(
        ChunkFormula(chunk_id=chunk.id, formula_id=formula_ids[i])
        for chunk, r in zip(chunks, records, strict=True)
        for i in r.formula_indices
    )
    session.add_all(
        ChunkCitation(chunk_id=chunk.id, citation_id=citation_ids[key])
        for chunk, r in zip(chunks, records, strict=True)
        for key in r.citation_keys
    )
    session.flush()

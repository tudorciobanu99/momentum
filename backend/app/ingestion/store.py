"""Writing fetched articles to the database."""

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import Article, ArticleHtml, Formula
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
        )
        for r in records
    )
    session.flush()

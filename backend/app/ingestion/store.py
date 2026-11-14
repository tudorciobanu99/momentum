"""Writing fetched articles to the database."""

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import Article, ArticleHtml

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

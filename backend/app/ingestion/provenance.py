"""Reading the provenance chain: source, citation, article revision, formula."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Article, Citation, Formula, FormulaCitation, Source


@dataclass(frozen=True)
class Chain:
    """One link path from a source to a formula.

    Attributes:
        source_title: Title of the source.
        page: Page cited, if any.
        article_title: Title of the article that cites the source.
        rev_id: Revision of that article.
        url: Permanent URL of the revision.
        formula_id: Linked formula.
        latex: LaTeX of the formula.
        kind: `direct` or `inherited`.
    """

    source_title: str | None
    page: str | None
    article_title: str
    rev_id: int
    url: str
    formula_id: int
    latex: str
    kind: str


def chains_for_source(session: Session, source_id: int) -> list[Chain]:
    """Lists every formula reachable from a source through citations.

    Args:
        session: Open session.
        source_id: Source to start from.

    Returns:
        Chains ordered by article title and formula position.
    """
    rows = session.execute(
        select(Source.title, Citation.page, Article, Formula, FormulaCitation.kind)
        .join(Citation, Citation.source_id == Source.id)
        .join(Article, Article.id == Citation.article_id)
        .join(FormulaCitation, FormulaCitation.citation_id == Citation.id)
        .join(Formula, Formula.id == FormulaCitation.formula_id)
        .where(Source.id == source_id)
        .order_by(Article.title, Formula.position)
    ).all()
    return [
        Chain(title, page, a.title, a.rev_id, a.url, f.id, f.latex, kind)
        for title, page, a, f, kind in rows
    ]

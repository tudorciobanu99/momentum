"""Retrieve ranked chunks with their formulas and provenance."""

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as db
from app.retrieval.client import EmbeddingClient, Reranker
from app.retrieval.hybrid import hybrid_search
from app.retrieval.models import (
    ArticleRevision,
    ChunkCitation,
    FormulaHit,
    ProvenanceLink,
    RetrievedChunk,
    SourceRef,
    WikidataFormula,
)
from app.retrieval.rerank import rerank_candidates
from app.retrieval.symbols import is_single_symbol

_KIND_ORDER = {"direct": 0, "inherited": 1, "section": 2, "none": 3}


def _source_ref(citation: db.Citation, source: db.Source | None) -> SourceRef | None:
    if source is not None:
        return SourceRef(
            title=source.title,
            authors=source.authors,
            year=source.year,
            doi=source.doi,
            isbn=source.isbn,
        )
    title = citation.raw.get("title")
    if isinstance(title, str) and title.strip():
        return SourceRef(title=title.strip())
    return None


def _link(
    kind: str, citation: db.Citation, source: db.Source | None, revision: ArticleRevision
) -> ProvenanceLink:
    return ProvenanceLink.model_validate(
        {
            "kind": kind,
            "source": _source_ref(citation, source),
            "page": citation.page,
            "revision": revision,
        }
    )


def _sort_key(link: ProvenanceLink) -> tuple[int, bool, str, bool, str]:
    title = link.source.title if link.source else None
    return (_KIND_ORDER[link.kind], title is None, title or "", link.page is None, link.page or "")


def retrieve(
    session: Session,
    client: EmbeddingClient,
    reranker: Reranker,
    query: str,
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """Finds the best chunks for a question, with formulas and provenance chains.

    Runs hybrid search and reranking, then loads everything in a constant number of queries.
    A formula without citations gets one `none` link, so its chain ends at the article revision.
    Single-symbol formulas are left out. A citation without a source takes its title from the
    stored raw data, if any. Chunk citations are reported separately, without a kind. When several
    Wikidata records share a QID, the one with the lowest id is attached.

    Args:
        session: Open database session.
        client: Embedding client.
        reranker: Reranking client.
        query: Free-text question.
        top_n: Maximum chunks, defaulting to `settings.rerank_top_n`.

    Returns:
        Chunks in rerank order, best first.

    Raises:
        ValueError: If the query is blank.
        RateLimitError: If a client is rate limiting.
        RerankError: If the reranker fails.
    """
    candidates = hybrid_search(session, client, query)
    ranked = rerank_candidates(session, reranker, query, candidates, top_n)
    if not ranked:
        return []
    chunk_ids = [r.chunk_id for r in ranked]

    chunks = {c.id: c for c in session.scalars(select(db.Chunk).where(db.Chunk.id.in_(chunk_ids)))}
    formula_rows = session.execute(
        select(db.ChunkFormula.chunk_id, db.Formula)
        .join(db.Formula, db.Formula.id == db.ChunkFormula.formula_id)
        .where(db.ChunkFormula.chunk_id.in_(chunk_ids))
        .order_by(db.Formula.index, db.Formula.id)
    ).all()
    formula_ids = list({f.id for _, f in formula_rows})
    qids = list({f.wikidata_qid for _, f in formula_rows if f.wikidata_qid})

    formula_cites = session.execute(
        select(db.FormulaCitation.formula_id, db.FormulaCitation.kind, db.Citation, db.Source)
        .join(db.Citation, db.Citation.id == db.FormulaCitation.citation_id)
        .outerjoin(db.Source, db.Source.id == db.Citation.source_id)
        .where(db.FormulaCitation.formula_id.in_(formula_ids))
    ).all()
    chunk_cites = session.execute(
        select(db.ChunkCitation.chunk_id, db.Citation, db.Source)
        .join(db.Citation, db.Citation.id == db.ChunkCitation.citation_id)
        .outerjoin(db.Source, db.Source.id == db.Citation.source_id)
        .where(db.ChunkCitation.chunk_id.in_(chunk_ids))
    ).all()

    article_ids = {c.article_id for c in chunks.values()}
    article_ids |= {f.article_id for _, f in formula_rows}
    article_ids |= {c.article_id for _, _, c, _ in formula_cites}
    article_ids |= {c.article_id for _, c, _ in chunk_cites}
    revisions = {
        a.id: ArticleRevision(title=a.title, rev_id=a.rev_id, url=a.url, license=a.license)
        for a in session.scalars(select(db.Article).where(db.Article.id.in_(article_ids)))
    }

    wikidata: dict[str, WikidataFormula] = {}
    for w in session.scalars(
        select(db.WikidataFormula)
        .where(db.WikidataFormula.qid.in_(qids))
        .order_by(db.WikidataFormula.id.desc())
    ):
        wikidata[w.qid] = WikidataFormula(
            qid=w.qid, latex=w.latex, symbols=w.symbols, references=w.references, license=w.license
        )

    links: dict[int, list[ProvenanceLink]] = defaultdict(list)
    for formula_id, kind, citation, source in formula_cites:
        links[formula_id].append(_link(kind, citation, source, revisions[citation.article_id]))

    hits: dict[int, list[FormulaHit]] = defaultdict(list)
    for chunk_id, f in formula_rows:
        if is_single_symbol(f.latex):
            continue
        provenance = sorted(links.get(f.id, []), key=_sort_key)
        if not provenance:
            provenance = [ProvenanceLink(kind="none", revision=revisions[f.article_id])]
        hits[chunk_id].append(
            FormulaHit(
                id=f.id,
                index=f.index,
                latex=f.latex,
                display=f.display,
                section_path=f.section_path,
                provenance=provenance,
                wikidata=wikidata.get(f.wikidata_qid) if f.wikidata_qid else None,
            )
        )

    cited: dict[int, list[ChunkCitation]] = defaultdict(list)
    for chunk_id, citation, source in chunk_cites:
        cited[chunk_id].append(
            ChunkCitation(
                source=_source_ref(citation, source),
                page=citation.page,
                revision=revisions[citation.article_id],
            )
        )

    return [
        RetrievedChunk(
            chunk_id=r.chunk_id,
            heading_path=chunks[r.chunk_id].heading_path,
            text=chunks[r.chunk_id].text,
            rerank_score=r.rerank_score,
            fused_score=r.score,
            vector_rank=r.vector_rank,
            lexical_rank=r.lexical_rank,
            revision=revisions[chunks[r.chunk_id].article_id],
            formulas=hits[r.chunk_id],
            citations=sorted(
                cited[r.chunk_id],
                key=lambda x: (
                    x.source is None or x.source.title is None,
                    (x.source.title if x.source else None) or "",
                    x.page is None,
                    x.page or "",
                ),
            ),
        )
        for r in ranked
    ]

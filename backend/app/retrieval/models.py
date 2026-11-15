"""Typed retrieval results: chunks with their formulas and provenance."""

from typing import Any, Literal

from pydantic import BaseModel


class ArticleRevision(BaseModel):
    """A Wikipedia article at a fixed revision.

    Attributes:
        title: Article title.
        rev_id: Revision id.
        url: Permanent URL of the revision.
        license: License of the article text.
    """

    title: str
    rev_id: int
    url: str
    license: str


class SourceRef(BaseModel):
    """A cited source.

    Attributes:
        title: Title, if known.
        authors: Authors, if known.
        year: Publication year, if known.
        doi: DOI, if any.
        isbn: ISBN, if any.
    """

    title: str | None = None
    authors: str | None = None
    year: int | None = None
    doi: str | None = None
    isbn: str | None = None


class ProvenanceLink(BaseModel):
    """One step of a provenance chain, ending at a Wikipedia revision.

    Attributes:
        kind: How the citation attaches: `direct`, `inherited`, `section`, or `none` when nothing
            is cited and the chain ends at the revision.
        source: The cited source, None for kind `none` or a citation with neither a source nor a
            title; a citation without a source carries only its title.
        page: Page cited, if any.
        revision: Article revision that carries the citation.
    """

    kind: Literal["direct", "inherited", "section", "none"]
    source: SourceRef | None = None
    page: str | None = None
    revision: ArticleRevision


class ChunkCitation(BaseModel):
    """A source cited in a chunk's text, without a formula-level strength.

    Attributes:
        source: The cited source, None if unresolved and untitled; unresolved with a title carries
            only the title.
        page: Page cited, if any.
        revision: Article revision that carries the citation.
    """

    source: SourceRef | None = None
    page: str | None = None
    revision: ArticleRevision


class WikidataFormula(BaseModel):
    """A formula recorded on Wikidata.

    Attributes:
        qid: Wikidata item id.
        latex: LaTeX of the formula.
        symbols: Symbols as stored.
        references: References as stored.
        license: License of the Wikidata record.
    """

    qid: str
    latex: str
    symbols: Any
    references: Any
    license: str


class FormulaHit(BaseModel):
    """A formula of a retrieved chunk.

    Attributes:
        id: Formula id.
        index: Position of the formula in its article.
        latex: LaTeX source.
        display: True for display math, false for inline.
        section_path: Heading path of the section containing it.
        provenance: Links ordered by kind strength, source title, then page.
        wikidata: Matching Wikidata record, if any.
    """

    id: int
    index: int
    latex: str
    display: bool
    section_path: str
    provenance: list[ProvenanceLink]
    wikidata: WikidataFormula | None = None


class RetrievedChunk(BaseModel):
    """A reranked chunk with everything needed to cite it.

    Attributes:
        chunk_id: Chunk id.
        heading_path: Heading path of the chunk.
        text: Chunk text.
        rerank_score: Score from the reranker.
        fused_score: Reciprocal rank fusion score.
        vector_rank: 1-based rank in the vector list, None if absent.
        lexical_rank: 1-based rank in the lexical list, None if absent.
        revision: Revision of the article the chunk comes from.
        formulas: Formulas in the chunk, ordered by index.
        citations: Sources cited in the chunk text, ordered by source title then page.
    """

    chunk_id: int
    heading_path: str
    text: str
    rerank_score: float
    fused_score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None
    revision: ArticleRevision
    formulas: list[FormulaHit]
    citations: list[ChunkCitation]

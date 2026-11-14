"""SQLAlchemy models."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all models."""


class Article(Base):
    __tablename__ = "articles"
    __table_args__ = (UniqueConstraint("title", "rev_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    rev_id: Mapped[int] = mapped_column(Integer)
    url: Mapped[str] = mapped_column(Text)
    wikidata_qid: Mapped[str | None] = mapped_column(Text)
    topic: Mapped[str] = mapped_column(Text)
    license: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ArticleHtml(Base):
    __tablename__ = "article_html"

    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True
    )
    html: Mapped[str] = mapped_column(Text)


class Formula(Base):
    __tablename__ = "formulas"

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), index=True
    )
    index: Mapped[int] = mapped_column(Integer)
    latex: Mapped[str] = mapped_column(Text)
    display: Mapped[bool]
    section_path: Mapped[str] = mapped_column(Text)
    position: Mapped[int] = mapped_column(Integer)


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (
        Index("uq_sources_doi", "doi", unique=True, postgresql_where=text("doi IS NOT NULL")),
        Index("uq_sources_isbn", "isbn", unique=True, postgresql_where=text("isbn IS NOT NULL")),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    authors: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(Text)
    isbn: Mapped[str | None] = mapped_column(Text)


class Citation(Base):
    __tablename__ = "citations"
    __table_args__ = (
        UniqueConstraint("article_id", "ref_key", "page", postgresql_nulls_not_distinct=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    ref_key: Mapped[str] = mapped_column(Text)
    page: Mapped[str | None] = mapped_column(Text)
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB)
    source_id: Mapped[int | None] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), index=True
    )

    source: Mapped[Source | None] = relationship()


class FormulaCitation(Base):
    __tablename__ = "formula_citations"
    __table_args__ = (CheckConstraint("kind IN ('direct', 'inherited')", name="kind"),)

    formula_id: Mapped[int] = mapped_column(
        ForeignKey("formulas.id", ondelete="CASCADE"), primary_key=True
    )
    citation_id: Mapped[int] = mapped_column(
        ForeignKey("citations.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    kind: Mapped[str] = mapped_column(Text)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), index=True
    )
    index: Mapped[int] = mapped_column(Integer)
    heading_path: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)

    citations: Mapped[list[Citation]] = relationship(secondary="chunk_citations")
    formulas: Mapped[list[Formula]] = relationship(secondary="chunk_formulas")


class ChunkCitation(Base):
    __tablename__ = "chunk_citations"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    citation_id: Mapped[int] = mapped_column(
        ForeignKey("citations.id", ondelete="CASCADE"), primary_key=True, index=True
    )


class ChunkFormula(Base):
    __tablename__ = "chunk_formulas"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    formula_id: Mapped[int] = mapped_column(
        ForeignKey("formulas.id", ondelete="CASCADE"), primary_key=True, index=True
    )


class WikidataFormula(Base):
    __tablename__ = "wikidata_formulas"

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), index=True
    )
    qid: Mapped[str] = mapped_column(Text)
    latex: Mapped[str] = mapped_column(Text)
    symbols: Mapped[Any] = mapped_column(JSONB)
    references: Mapped[Any] = mapped_column(JSONB)
    license: Mapped[str] = mapped_column(Text)

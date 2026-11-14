"""SQLAlchemy models."""

from datetime import datetime
from typing import Any

from sqlalchemy import ARRAY, DateTime, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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


class Reference(Base):
    __tablename__ = "references"

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), index=True
    )
    ref_key: Mapped[str] = mapped_column(Text)
    authors: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(Text)
    isbn: Mapped[str | None] = mapped_column(Text)
    page: Mapped[str | None] = mapped_column(Text)
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB)


class ProvenanceLink(Base):
    __tablename__ = "provenance_links"

    id: Mapped[int] = mapped_column(primary_key=True)
    target_type: Mapped[str] = mapped_column(Text)
    target_id: Mapped[int] = mapped_column(Integer)
    reference_id: Mapped[int | None] = mapped_column(
        ForeignKey("references.id", ondelete="SET NULL"), index=True
    )
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), index=True
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
    formula_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    reference_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer))


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

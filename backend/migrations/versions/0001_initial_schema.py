"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-11-14 20:51:42.904082
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "articles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("rev_id", sa.Integer(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("wikidata_qid", sa.Text(), nullable=True),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("license", sa.Text(), nullable=False),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("title", "rev_id"),
    )
    op.create_table(
        "article_html",
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("html", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("article_id"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("heading_path", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chunks_article_id"), "chunks", ["article_id"], unique=False)
    op.create_table(
        "formulas",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("latex", sa.Text(), nullable=False),
        sa.Column("display", sa.Boolean(), nullable=False),
        sa.Column("section_path", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_formulas_article_id"), "formulas", ["article_id"], unique=False)
    op.create_table(
        "wikidata_formulas",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("qid", sa.Text(), nullable=False),
        sa.Column("latex", sa.Text(), nullable=False),
        sa.Column("symbols", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("references", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("license", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_wikidata_formulas_article_id"), "wikidata_formulas", ["article_id"], unique=False
    )
    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("authors", sa.Text(), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("doi", sa.Text(), nullable=True),
        sa.Column("isbn", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_sources_doi",
        "sources",
        ["doi"],
        unique=True,
        postgresql_where=sa.text("doi IS NOT NULL"),
    )
    op.create_index(
        "uq_sources_isbn",
        "sources",
        ["isbn"],
        unique=True,
        postgresql_where=sa.text("isbn IS NOT NULL"),
    )
    op.create_table(
        "citations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("ref_key", sa.Text(), nullable=False),
        sa.Column("page", sa.Text(), nullable=True),
        sa.Column("raw", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("article_id", "ref_key", "page", postgresql_nulls_not_distinct=True),
    )
    op.create_index(op.f("ix_citations_source_id"), "citations", ["source_id"], unique=False)
    op.create_table(
        "formula_citations",
        sa.Column("formula_id", sa.Integer(), nullable=False),
        sa.Column("citation_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "kind IN ('direct', 'inherited')", name=op.f("ck_formula_citations_kind")
        ),
        sa.ForeignKeyConstraint(["formula_id"], ["formulas.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["citation_id"], ["citations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("formula_id", "citation_id"),
    )
    op.create_index(
        op.f("ix_formula_citations_citation_id"), "formula_citations", ["citation_id"], unique=False
    )
    op.create_table(
        "chunk_citations",
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("citation_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["citation_id"], ["citations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id", "citation_id"),
    )
    op.create_index(
        op.f("ix_chunk_citations_citation_id"), "chunk_citations", ["citation_id"], unique=False
    )
    op.create_table(
        "chunk_formulas",
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("formula_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["formula_id"], ["formulas.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id", "formula_id"),
    )
    op.create_index(
        op.f("ix_chunk_formulas_formula_id"), "chunk_formulas", ["formula_id"], unique=False
    )


def downgrade() -> None:
    op.drop_table("chunk_formulas")
    op.drop_table("chunk_citations")
    op.drop_table("formula_citations")
    op.drop_table("citations")
    op.drop_table("sources")
    op.drop_table("wikidata_formulas")
    op.drop_table("formulas")
    op.drop_table("chunks")
    op.drop_table("article_html")
    op.drop_table("articles")

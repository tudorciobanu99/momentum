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
        sa.Column("formula_ids", sa.ARRAY(sa.Integer()), nullable=False),
        sa.Column("reference_ids", sa.ARRAY(sa.Integer()), nullable=False),
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
        "references",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("ref_key", sa.Text(), nullable=False),
        sa.Column("authors", sa.Text(), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("doi", sa.Text(), nullable=True),
        sa.Column("isbn", sa.Text(), nullable=True),
        sa.Column("page", sa.Text(), nullable=True),
        sa.Column("raw", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_references_article_id"), "references", ["article_id"], unique=False)
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
        "provenance_links",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("target_type", sa.Text(), nullable=False),
        sa.Column("target_id", sa.Integer(), nullable=False),
        sa.Column("reference_id", sa.Integer(), nullable=True),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reference_id"], ["references.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_provenance_links_article_id"), "provenance_links", ["article_id"], unique=False
    )
    op.create_index(
        op.f("ix_provenance_links_reference_id"), "provenance_links", ["reference_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_provenance_links_reference_id"), table_name="provenance_links")
    op.drop_index(op.f("ix_provenance_links_article_id"), table_name="provenance_links")
    op.drop_table("provenance_links")
    op.drop_index(op.f("ix_wikidata_formulas_article_id"), table_name="wikidata_formulas")
    op.drop_table("wikidata_formulas")
    op.drop_index(op.f("ix_references_article_id"), table_name="references")
    op.drop_table("references")
    op.drop_index(op.f("ix_formulas_article_id"), table_name="formulas")
    op.drop_table("formulas")
    op.drop_index(op.f("ix_chunks_article_id"), table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("article_html")
    op.drop_table("articles")

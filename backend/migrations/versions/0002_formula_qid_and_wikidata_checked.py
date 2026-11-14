"""formula wikidata qid and article wikidata check time

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-14 22:10:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(op.f("ck_formula_citations_kind"), "formula_citations", type_="check")
    op.create_check_constraint(
        op.f("ck_formula_citations_kind"),
        "formula_citations",
        "kind IN ('direct', 'inherited', 'section')",
    )
    op.add_column("formulas", sa.Column("wikidata_qid", sa.Text(), nullable=True))
    op.create_index(op.f("ix_formulas_wikidata_qid"), "formulas", ["wikidata_qid"], unique=False)
    op.add_column(
        "articles", sa.Column("wikidata_checked_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.execute("DELETE FROM formula_citations WHERE kind = 'section'")
    op.drop_constraint(op.f("ck_formula_citations_kind"), "formula_citations", type_="check")
    op.create_check_constraint(
        op.f("ck_formula_citations_kind"), "formula_citations", "kind IN ('direct', 'inherited')"
    )
    op.drop_column("articles", "wikidata_checked_at")
    op.drop_index(op.f("ix_formulas_wikidata_qid"), table_name="formulas")
    op.drop_column("formulas", "wikidata_qid")

"""Alternative chunkings in a temporary Postgres schema, for comparison runs.

A corpus is a session whose `search_path` puts the schema first, so the unqualified `chunks` and
`chunk_formulas` tables used by search, rerank and formula lookup resolve to the schema's copies.
`public.chunks` is never written.
"""

import re
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from pgvector.sqlalchemy import Vector
from sqlalchemy import column, insert, select, table, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Chunk, ChunkFormula
from app.ingestion.chunks import MIN_TOKENS, count_tokens
from app.retrieval.client import EmbeddingClient
from app.retrieval.service import _embed_with_retry, text_hash
from app.settings import settings

MERGED_SCHEMA = "eval_merged"
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*")


@dataclass(frozen=True)
class ChunkRow:
    """A stored chunk as far as merging needs it.

    Attributes:
        id: Chunk id.
        article_id: Owning article id.
        index: Order within the article.
        heading_path: Headings from the article title down, joined with " > ".
        text: Chunk text, starting with the heading path.
        token_count: Tokens in `text`.
    """

    id: int
    article_id: int
    index: int
    heading_path: str
    text: str
    token_count: int


def _section(row: ChunkRow) -> str:
    return " > ".join(row.heading_path.split(" > ")[:2])


def merge_groups(rows: list[ChunkRow], min_tokens: int = MIN_TOKENS) -> list[list[ChunkRow]]:
    """Groups consecutive short chunks of one top-level section.

    A group starts at a chunk under `min_tokens` and absorbs following chunks of the same article
    and top-level section until it reaches `min_tokens`. A short group left at the end of a
    section joins the group before it. Chunks of `min_tokens` or more stay alone.

    Args:
        rows: Chunks ordered by article, then index.
        min_tokens: Token count a merged chunk should reach.

    Returns:
        Groups in input order, each of one or more chunks.
    """
    groups: list[list[ChunkRow]] = []
    tokens: list[int] = []
    i = 0
    while i < len(rows):
        group = [rows[i]]
        total = rows[i].token_count
        i += 1
        while (
            total < min_tokens
            and i < len(rows)
            and rows[i].article_id == group[0].article_id
            and _section(rows[i]) == _section(group[0])
        ):
            group.append(rows[i])
            total += rows[i].token_count
            i += 1
        if (
            total < min_tokens
            and groups
            and groups[-1][0].article_id == group[0].article_id
            and _section(groups[-1][0]) == _section(group[0])
        ):
            groups[-1].extend(group)
            tokens[-1] += total
        else:
            groups.append(group)
            tokens.append(total)
    return groups


def merged_text(group: list[ChunkRow]) -> str:
    """Joins the chunks of a group under the heading path of the first.

    Args:
        group: Chunks to merge, at least one.

    Returns:
        The first chunk's text followed by the bodies of the others.
    """
    parts = [group[0].text]
    for row in group[1:]:
        prefix = f"{row.heading_path}\n\n"
        parts.append(row.text.removeprefix(prefix))
    return "\n\n".join(parts)


def _check_schema_name(schema: str) -> str:
    if not _IDENTIFIER.fullmatch(schema) or schema == "public":
        raise ValueError(f"invalid schema name {schema!r}")
    return schema


@contextmanager
def corpus_session(sessions: sessionmaker[Session], schema: str | None) -> Iterator[Session]:
    """Opens a session that reads the chunks of a corpus.

    Args:
        sessions: Session factory.
        schema: Schema holding the corpus tables, or None for the default `public` corpus.

    Yields:
        A session for search, rerank and formula lookup. Nothing is committed.
    """
    with sessions() as session:
        if schema is not None:
            session.execute(text(f"SET search_path TO {_check_schema_name(schema)}, public"))
        try:
            yield session
        finally:
            session.rollback()


def drop_schema(sessions: sessionmaker[Session], schema: str) -> None:
    """Drops a corpus schema and everything in it.

    Args:
        sessions: Session factory.
        schema: Schema name.
    """
    with sessions.begin() as session:
        session.execute(text(f"DROP SCHEMA IF EXISTS {_check_schema_name(schema)} CASCADE"))


def build_merged_corpus(
    sessions: sessionmaker[Session],
    client: EmbeddingClient,
    schema: str = MERGED_SCHEMA,
    min_tokens: int = MIN_TOKENS,
) -> int:
    """Builds a chunking with short chunks merged in a fresh schema.

    The schema gets `chunks` with the same generated `search_vector`, HNSW and GIN indexes as
    `public.chunks`, and `chunk_formulas` holding the union of the merged chunks' formula links.
    Chunks that are not merged keep their stored embedding, merged chunks are embedded as documents.

    Args:
        sessions: Session factory.
        client: Embedding client.
        schema: Schema to create, replaced if it exists.
        min_tokens: Token count a merged chunk should reach.

    Returns:
        Number of chunks in the new corpus.
    """
    _check_schema_name(schema)
    drop_schema(sessions, schema)
    with sessions.begin() as session:
        session.execute(text(f"CREATE SCHEMA {schema}"))
        session.execute(text(f"CREATE TABLE {schema}.chunks (LIKE public.chunks INCLUDING ALL)"))
        session.execute(
            text(
                f"CREATE TABLE {schema}.chunk_formulas ("
                "chunk_id integer NOT NULL REFERENCES "
                f"{schema}.chunks (id) ON DELETE CASCADE, "
                "formula_id integer NOT NULL REFERENCES public.formulas (id) ON DELETE CASCADE, "
                "PRIMARY KEY (chunk_id, formula_id))"
            )
        )
    with sessions() as session:
        stored = session.execute(
            select(
                Chunk.id,
                Chunk.article_id,
                Chunk.index,
                Chunk.heading_path,
                Chunk.text,
                Chunk.token_count,
                Chunk.embedding,
                Chunk.embedding_hash,
            ).order_by(Chunk.article_id, Chunk.index)
        ).all()
        links: dict[int, list[int]] = defaultdict(list)
        for chunk_id, formula_id in session.execute(
            select(ChunkFormula.chunk_id, ChunkFormula.formula_id)
        ):
            links[chunk_id].append(formula_id)
    rows = [ChunkRow(*r[:6]) for r in stored]
    stored_by_id = {r[0]: r for r in stored}
    groups = merge_groups(rows, min_tokens)

    out: list[dict[str, object]] = []
    formula_links: list[dict[str, int]] = []
    for new_id, group in enumerate(groups, 1):
        body = merged_text(group)
        first = group[0]
        embedding = stored_by_id[first.id][6] if len(group) == 1 else None
        out.append(
            {
                "id": new_id,
                "article_id": first.article_id,
                "index": first.index,
                "heading_path": first.heading_path,
                "text": body,
                "token_count": count_tokens(body),
                "embedding": embedding,
                "embedding_hash": stored_by_id[first.id][7] if embedding is not None else None,
            }
        )
        for formula_id in sorted({f for row in group for f in links.get(row.id, [])}):
            formula_links.append({"chunk_id": new_id, "formula_id": formula_id})

    pending = [o for o in out if o["embedding"] is None]
    size = settings.embedding_batch_size
    for start in range(0, len(pending), size):
        batch = pending[start : start + size]
        texts = [str(o["text"]) for o in batch]
        vectors = _embed_with_retry(client, texts, 5, time.sleep)
        if len(vectors) != len(batch):
            raise ValueError(f"expected {len(batch)} vectors, got {len(vectors)}")
        for o, vector, t in zip(batch, vectors, texts, strict=True):
            o["embedding"] = vector
            o["embedding_hash"] = text_hash(t)

    chunks = table(
        "chunks",
        column("id"),
        column("article_id"),
        column("index"),
        column("heading_path"),
        column("text"),
        column("token_count"),
        column("embedding", Vector(settings.embedding_dimension)),
        column("embedding_hash"),
        schema=schema,
    )
    chunk_formulas = table(
        "chunk_formulas", column("chunk_id"), column("formula_id"), schema=schema
    )
    with sessions.begin() as session:
        for start in range(0, len(out), 500):
            session.execute(insert(chunks), out[start : start + 500])
        if formula_links:
            session.execute(insert(chunk_formulas), formula_links)
    return len(out)

"""Fetch service: resolve revisions, download HTML, store, and pin rev ids."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlencode

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, ArticleHtml, Formula
from app.ingestion.chunks import chunk_article
from app.ingestion.citations import DIRECT, INHERITED, SECTION, parse_citations
from app.ingestion.client import WikiClient
from app.ingestion.formulas import extract_formulas
from app.ingestion.store import (
    is_stored,
    save_article,
    save_chunks,
    save_citations,
    save_formulas,
)
from app.ingestion.titles import TITLES_PATH, TitleEntry, load_titles, write_titles

KINDS = (DIRECT, INHERITED, SECTION, "none")


@dataclass
class FetchReport:
    """Outcome of a fetch run.

    Attributes:
        stored: Titles newly written to the database.
        unchanged: Titles already stored at their revision.
        skipped: Title to reason, for missing, redirected or failed titles.
    """

    stored: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)


@dataclass
class CitationsReport:
    """Outcome of a citations run.

    Attributes:
        citations: Citations stored.
        with_doi: Citations with a DOI.
        with_isbn: Citations with an ISBN.
        merged_sources: Sources merged into another because DOI and ISBN matched two rows.
        display: Display formulas by strongest link kind: direct, inherited, section, none.
        inline: Inline formulas by strongest link kind, same keys.
    """

    citations: int = 0
    with_doi: int = 0
    with_isbn: int = 0
    merged_sources: int = 0
    display: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KINDS, 0))
    inline: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KINDS, 0))


def revision_url(title: str, rev_id: int) -> str:
    """Builds the permanent URL of a revision.

    Args:
        title: Page title.
        rev_id: Revision id.

    Returns:
        URL of the form `https://en.wikipedia.org/w/index.php?title=...&oldid=...`.
    """
    query = urlencode({"title": title.replace(" ", "_"), "oldid": rev_id})
    return f"https://en.wikipedia.org/w/index.php?{query}"


async def _latest_rev_id(client: WikiClient, title: str) -> int:
    path = quote(title.replace(" ", "_"), safe="")
    response = await client.get(f"/w/rest.php/v1/page/{path}/bare")
    if response.is_redirect:
        raise LookupError("redirect")
    if response.status_code == 404:
        raise LookupError("missing")
    response.raise_for_status()
    data = response.json()
    if data["title"].replace("_", " ") != title:
        raise LookupError(f"redirect to {data['title']}")
    return data["latest"]["id"]


async def _wikidata_qid(client: WikiClient, rev_id: int) -> str | None:
    response = await client.get(
        "/w/api.php",
        params={
            "action": "query",
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "revids": rev_id,
            "format": "json",
            "formatversion": 2,
        },
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    return pages[0].get("pageprops", {}).get("wikibase_item") if pages else None


async def _fetch_one(
    client: WikiClient,
    sessions: sessionmaker[Session],
    entry: TitleEntry,
    report: FetchReport,
) -> None:
    try:
        if entry.rev_id is None:
            entry.rev_id = await _latest_rev_id(client, entry.title)
        rev_id = entry.rev_id
        with sessions() as session:
            if is_stored(session, entry.title, rev_id):
                report.unchanged.append(entry.title)
                return
        response = await client.get(f"/w/rest.php/v1/revision/{rev_id}/html")
        if response.status_code == 404:
            raise LookupError("revision missing")
        response.raise_for_status()
        qid = await _wikidata_qid(client, rev_id)
        with sessions.begin() as session:
            save_article(
                session,
                title=entry.title,
                rev_id=rev_id,
                url=revision_url(entry.title, rev_id),
                wikidata_qid=qid,
                topic=entry.topic,
                html=response.text,
            )
        report.stored.append(entry.title)
    except LookupError as e:
        report.skipped[entry.title] = str(e)
    except Exception as e:
        report.skipped[entry.title] = f"error: {e}"


async def fetch_all(
    sessions: sessionmaker[Session],
    titles_path: Path = TITLES_PATH,
    client: WikiClient | None = None,
) -> FetchReport:
    """Fetches every listed article at its pinned revision and stores it.

    Entries without a `rev_id` get the latest revision, which is written back to the
    titles file. Missing, redirected and failed titles are reported and skipped.

    Args:
        sessions: Session factory for the target database.
        titles_path: TOML titles file.
        client: HTTP client; a default one is created when omitted.

    Returns:
        The run report.
    """
    entries = load_titles(titles_path)
    report = FetchReport()
    async with client or WikiClient() as c:
        await asyncio.gather(*(_fetch_one(c, sessions, e, report) for e in entries))
    write_titles(entries, titles_path)
    return report


def extract_all(sessions: sessionmaker[Session]) -> dict[str, int]:
    """Extracts and stores formulas for every stored article, replacing earlier results.

    Args:
        sessions: Session factory for the target database.

    Returns:
        Title to number of formulas stored.
    """
    counts: dict[str, int] = {}
    with sessions() as session:
        rows = session.execute(select(Article.id, Article.title).order_by(Article.id)).all()
    for article_id, title in rows:
        with sessions.begin() as session:
            html = session.scalar(
                select(ArticleHtml.html).where(ArticleHtml.article_id == article_id)
            )
            records = extract_formulas(html or "")
            save_formulas(session, article_id, records)
        counts[title] = len(records)
    return counts


def citations_all(sessions: sessionmaker[Session]) -> CitationsReport:
    """Extracts and stores citations and formula links for every article, replacing earlier ones.

    Formulas must already be extracted.

    Args:
        sessions: Session factory for the target database.

    Returns:
        Counts over all articles.
    """
    report = CitationsReport()
    with sessions() as session:
        ids = session.scalars(select(Article.id).order_by(Article.id)).all()
    for article_id in ids:
        with sessions.begin() as session:
            html = session.scalar(
                select(ArticleHtml.html).where(ArticleHtml.article_id == article_id)
            )
            formulas = {
                position: display
                for position, display in session.execute(
                    select(Formula.position, Formula.display).where(
                        Formula.article_id == article_id
                    )
                )
            }
            parsed = parse_citations(html or "", formulas)
            report.merged_sources += save_citations(session, article_id, parsed)
        report.citations += len(parsed.citations)
        report.with_doi += sum(1 for c in parsed.citations if c.doi)
        report.with_isbn += sum(1 for c in parsed.citations if c.isbn)
        for position, display in formulas.items():
            kinds = set(parsed.links.get(position, {}).values())
            kind = next((k for k in KINDS if k in kinds), "none")
            (report.display if display else report.inline)[kind] += 1
    return report


def chunks_all(sessions: sessionmaker[Session]) -> dict[str, int]:
    """Chunks every stored article and stores the chunks, replacing earlier ones.

    Formulas and citations must already be extracted.

    Args:
        sessions: Session factory for the target database.

    Returns:
        Title to number of chunks stored.

    Raises:
        LookupError: An article's formulas or citations are not stored yet.
    """
    counts: dict[str, int] = {}
    with sessions() as session:
        rows = session.execute(select(Article.id, Article.title).order_by(Article.id)).all()
    for article_id, title in rows:
        with sessions.begin() as session:
            html = session.scalar(
                select(ArticleHtml.html).where(ArticleHtml.article_id == article_id)
            )
            records = chunk_article(html or "")
            try:
                save_chunks(session, article_id, records)
            except LookupError:
                raise LookupError(
                    f"run `formulas` and `citations` first: missing for {title}"
                ) from None
        counts[title] = len(records)
    return counts

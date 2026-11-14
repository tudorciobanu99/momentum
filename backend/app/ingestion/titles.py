"""Loading, validating and updating the committed article titles file."""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

TITLES_PATH = Path(__file__).parent / "articles.toml"
TOPICS = ("mechanics", "gravitation", "electromagnetism", "circuits", "relativity", "quantum")
MIN_TITLES = 45
MAX_TITLES = 55
MIN_PER_TOPIC = 3


@dataclass
class TitleEntry:
    """One article to ingest.

    Attributes:
        title: English Wikipedia page title.
        topic: One of `TOPICS`.
        rev_id: Pinned revision id, or None until pinned.
    """

    title: str
    topic: str
    rev_id: int | None = None


def validate(entries: list[TitleEntry]) -> None:
    """Checks the entries against the size, uniqueness and topic rules.

    Args:
        entries: Entries to check.

    Raises:
        ValueError: If any rule is broken.
    """
    if not MIN_TITLES <= len(entries) <= MAX_TITLES:
        raise ValueError(f"expected {MIN_TITLES}-{MAX_TITLES} titles, got {len(entries)}")
    titles = [e.title for e in entries]
    duplicates = sorted({t for t in titles if titles.count(t) > 1})
    if duplicates:
        raise ValueError(f"duplicate titles: {duplicates}")
    unknown = sorted({e.topic for e in entries} - set(TOPICS))
    if unknown:
        raise ValueError(f"unknown topics: {unknown}")
    for topic in TOPICS:
        count = sum(e.topic == topic for e in entries)
        if count < MIN_PER_TOPIC:
            raise ValueError(f"topic {topic!r} has {count} titles, need {MIN_PER_TOPIC}")


def load_titles(path: Path = TITLES_PATH) -> list[TitleEntry]:
    """Reads and validates the titles file.

    Args:
        path: TOML file with an `[[articles]]` table per entry.

    Returns:
        The validated entries in file order.

    Raises:
        ValueError: If the file breaks a validation rule.
    """
    with path.open("rb") as f:
        rows = tomllib.load(f).get("articles", [])
    entries = [TitleEntry(r["title"], r["topic"], r.get("rev_id")) for r in rows]
    validate(entries)
    return entries


def write_titles(entries: list[TitleEntry], path: Path = TITLES_PATH) -> None:
    """Writes the entries back to the titles file, omitting unpinned `rev_id`s.

    Args:
        entries: Entries to write.
        path: Destination TOML file.
    """
    blocks = []
    for e in entries:
        lines = ["[[articles]]", f"title = {json.dumps(e.title, ensure_ascii=False)}"]
        lines.append(f"topic = {json.dumps(e.topic)}")
        if e.rev_id is not None:
            lines.append(f"rev_id = {e.rev_id}")
        blocks.append("\n".join(lines) + "\n")
    path.write_text("\n".join(blocks), encoding="utf-8")

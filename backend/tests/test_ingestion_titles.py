"""Tests for the titles file loader and validator."""

from pathlib import Path

import pytest

from app.ingestion.titles import (
    TOPICS,
    TitleEntry,
    load_titles,
    validate,
    write_titles,
)


def _entries(per_topic: int) -> list[TitleEntry]:
    return [TitleEntry(f"{t} {i}", t) for t in TOPICS for i in range(per_topic)]


def test_committed_file_is_valid() -> None:
    entries = load_titles()
    assert 45 <= len(entries) <= 55
    assert len({e.title for e in entries}) == len(entries)
    for topic in TOPICS:
        assert sum(e.topic == topic for e in entries) >= 3


def test_rejects_wrong_size() -> None:
    with pytest.raises(ValueError, match="expected"):
        validate(_entries(2))


def test_rejects_duplicates() -> None:
    entries = _entries(8)
    entries[1].title = entries[0].title
    with pytest.raises(ValueError, match="duplicate"):
        validate(entries)


def test_rejects_unknown_topic() -> None:
    entries = _entries(8)
    entries[0].topic = "optics"
    with pytest.raises(ValueError, match="unknown topics"):
        validate(entries)


def test_rejects_sparse_topic() -> None:
    entries = [e for e in _entries(9) if e.topic != "quantum"] + [
        TitleEntry(f"q{i}", "quantum") for i in range(2)
    ]
    with pytest.raises(ValueError, match="quantum"):
        validate(entries)


def test_write_round_trip(tmp_path: Path) -> None:
    entries = _entries(8)
    entries[0].rev_id = 123
    entries[1].title = "Biot–Savart law"
    path = tmp_path / "t.toml"
    write_titles(entries, path)
    assert load_titles(path) == entries

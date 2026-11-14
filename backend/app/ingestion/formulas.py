"""Extracting math formulas from Parsoid HTML."""

import json
from dataclasses import dataclass

from selectolax.lexbor import LexborHTMLParser, LexborNode

MATH_TYPEOF = "mw:Extension/math"
REF_TYPEOF = "mw:Extension/ref"
HEADING_LEVELS = {f"h{n}": n for n in range(1, 7)}
BLOCK_PARENTS = {"dd", "p"}


@dataclass(frozen=True)
class FormulaRecord:
    """One formula found in an article.

    Attributes:
        index: Zero-based order of the formula in the document.
        latex: LaTeX source from `data-mw.body.extsrc`, stripped of surrounding whitespace.
        display: True for display math, False for inline.
        section_path: Headings from the article title down, joined with " > ".
        position: Zero-based ordinal of the formula element among all elements of the
            document in document order. Stable for a pinned revision.
        wikidata_qid: Item id from `data-mw.attrs.qid`, when the formula has one.
    """

    index: int
    latex: str
    display: bool
    section_path: str
    position: int
    wikidata_qid: str | None = None


def _extsrc(data_mw: dict) -> tuple[str, str | None, str | None]:
    attrs = data_mw.get("attrs") or {}
    body = data_mw.get("body") or {}
    return str(body.get("extsrc") or "").strip(), attrs.get("display"), attrs.get("qid") or None


def _is_display(node: LexborNode, display: str | None) -> bool:
    if display is not None:
        return display == "block"
    parent = node.parent
    if parent is None or parent.tag not in BLOCK_PARENTS:
        return False
    own = "".join(
        child.text(strip=True)
        for child in parent.iter(include_text=True)
        if REF_TYPEOF not in (child.attributes.get("typeof") or "").split()
    )
    return own == node.text(strip=True)


def push_heading(headings: list[tuple[int, str]], tag: str, text: str) -> int:
    """Pushes a heading onto the section stack, popping headings of the same or a lower level.

    Args:
        headings: Stack of (level, text) from the outermost heading down; modified in place.
        tag: Heading tag, `h1` to `h6`.
        text: Heading text.

    Returns:
        The heading level.
    """
    level = HEADING_LEVELS[tag]
    while headings and headings[-1][0] >= level:
        headings.pop()
    headings.append((level, text))
    return level


def extract_formulas(html: str) -> list[FormulaRecord]:
    """Finds every `mw:Extension/math` element in document order.

    Display is taken from `attrs.display` when set. When unset, a formula that is the only
    content of its `dd` (legacy indented math) or `p`, not counting ref markers, counts as
    display. `{{math}}` and
    `{{mvar}}` templates render as plain spans and are not formulas.

    Args:
        html: Parsoid HTML of an article.

    Returns:
        Formula records ordered by position.
    """
    tree = LexborHTMLParser(html)
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else ""
    headings: list[tuple[int, str]] = []
    records: list[FormulaRecord] = []
    for position, node in enumerate(tree.css("*")):
        tag = node.tag
        attrs = node.attributes
        if tag in HEADING_LEVELS and node.parent is not None and node.parent.tag == "section":
            push_heading(headings, tag, node.text(strip=True))
        elif attrs.get("typeof") == MATH_TYPEOF and attrs.get("data-mw"):
            latex, display, qid = _extsrc(json.loads(attrs["data-mw"] or "{}"))
            if not latex:
                continue
            path = " > ".join([title, *(text for _, text in headings)])
            records.append(
                FormulaRecord(len(records), latex, _is_display(node, display), path, position, qid)
            )
    return records

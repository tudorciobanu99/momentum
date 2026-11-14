"""Extracting math formulas from Parsoid HTML."""

import json
from dataclasses import dataclass

from selectolax.lexbor import LexborHTMLParser, LexborNode

MATH_TYPEOF = "mw:Extension/math"
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
    """

    index: int
    latex: str
    display: bool
    section_path: str
    position: int


def _extsrc(data_mw: dict) -> tuple[str, str | None]:
    attrs = data_mw.get("attrs") or {}
    body = data_mw.get("body") or {}
    return str(body.get("extsrc") or "").strip(), attrs.get("display")


def _is_display(node: LexborNode, display: str | None) -> bool:
    if display is not None:
        return display == "block"
    parent = node.parent
    if parent is None or parent.tag not in BLOCK_PARENTS:
        return False
    return parent.text(strip=True) == node.text(strip=True)


def extract_formulas(html: str) -> list[FormulaRecord]:
    """Finds every `mw:Extension/math` element in document order.

    Display is taken from `attrs.display` when set. When unset, a formula that is the only
    content of its `dd` (legacy indented math) or `p` counts as display. `{{math}}` and
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
            level = HEADING_LEVELS[tag]
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, node.text(strip=True)))
        elif attrs.get("typeof") == MATH_TYPEOF and attrs.get("data-mw"):
            latex, display = _extsrc(json.loads(attrs["data-mw"] or "{}"))
            if not latex:
                continue
            path = " > ".join([title, *(text for _, text in headings)])
            records.append(
                FormulaRecord(len(records), latex, _is_display(node, display), path, position)
            )
    return records

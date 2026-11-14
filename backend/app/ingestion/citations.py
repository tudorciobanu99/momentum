"""Extracting references from Parsoid HTML and linking them to formulas."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote

import mwparserfromhell
from selectolax.lexbor import LexborHTMLParser, LexborNode

REF_TYPEOF = "mw:Extension/ref"
MATH_TYPEOF = "mw:Extension/math"
NOTE_PREFIX = "mw-reference-text-"
BLOCK_TAGS = {"p", "dd", "dt", "li", "td", "th", "blockquote", "div"}
WRAP_TAGS = {
    "dl",
    "dd",
    "dt",
    "ul",
    "ol",
    "table",
    "tbody",
    "thead",
    "tr",
    "td",
    "th",
    "blockquote",
}
PAGE_PARAMS = ("p", "pp", "loc", "page", "pages")
DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
PUNCTUATION = ".,;:"

DIRECT = "direct"
INHERITED = "inherited"
SECTION = "section"


@dataclass(frozen=True)
class ParsedCitation:
    """One reference note of an article, resolved to its bibliographic data.

    Attributes:
        ref_key: Id of the note in the references list, unique within the article.
        page: Page or location from the cite template or the short citation.
        raw: Template name and raw params, plus the bibliography entry for short citations.
        authors: Authors as "Last, First; Last, First".
        year: Publication year.
        title: Title with markup stripped.
        doi: Normalized DOI.
        isbn: Normalized ISBN-13.
    """

    ref_key: str
    page: str | None
    raw: dict[str, Any]
    authors: str | None = None
    year: int | None = None
    title: str | None = None
    doi: str | None = None
    isbn: str | None = None


@dataclass
class ParsedArticle:
    """Citations of an article and how formulas link to them.

    Attributes:
        citations: One per referenced note, in order of first use.
        links: Formula position to `{ref_key: "direct" | "inherited" | "section"}`.
    """

    citations: list[ParsedCitation] = field(default_factory=list)
    links: dict[int, dict[str, str]] = field(default_factory=dict)


def normalize_doi(value: str | None) -> str | None:
    """Lowercases a DOI and strips resolver prefixes.

    Args:
        value: DOI as written, possibly a doi.org URL.

    Returns:
        The bare DOI, or None if it does not look like one.
    """
    if not value:
        return None
    doi = DOI_PREFIX.sub("", value.strip()).strip().lower()
    return doi if doi.startswith("10.") and "/" in doi else None


def _isbn13_check(digits: str) -> str:
    total = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(digits[:12]))
    return str((10 - total % 10) % 10)


def _isbn_digits(candidate: str) -> str | None:
    s = re.sub(r"[^0-9Xx]", "", candidate).upper()
    if len(s) == 13 and s.isdigit() and s[:3] in ("978", "979"):
        return s if _isbn13_check(s) == s[12] else None
    if len(s) == 10 and s[:9].isdigit():
        total = sum((10 - i) * int(c) for i, c in enumerate(s[:9]))
        total += 10 if s[9] == "X" else int(s[9]) if s[9].isdigit() else -100
        if total % 11:
            return None
        body = "978" + s[:9]
        return body + _isbn13_check(body)
    return None


def normalize_isbn(value: str | None) -> str | None:
    """Reduces an ISBN to digits, converting ISBN-10 to ISBN-13.

    Args:
        value: ISBN as written, with hyphens or spaces; may list several.

    Returns:
        The first valid ISBN as 13 digits, or None.
    """
    if not value:
        return None
    for piece in re.split(r"[;,|/]|\bISBN\b", value, flags=re.IGNORECASE):
        isbn = _isbn_digits(piece)
        if isbn:
            return isbn
    return None


def _plain(wikitext: str) -> str:
    text = mwparserfromhell.parse(wikitext).strip_code(keep_template_params=True)
    return " ".join(text.split())


def _templates(node: LexborNode) -> list[tuple[str, dict[str, str]]]:
    found: list[tuple[str, dict[str, str]]] = []
    nodes = (
        [node, *node.css("[data-mw]")] if "data-mw" in node.attributes else node.css("[data-mw]")
    )
    for n in nodes:
        try:
            data = json.loads(n.attributes["data-mw"] or "{}")
        except ValueError:
            continue
        for part in data.get("parts", []):
            if not isinstance(part, dict) or "template" not in part:
                continue
            template = part["template"]
            name = " ".join(template["target"].get("wt", "").replace("_", " ").lower().split())
            params = {
                str(k).strip().lower(): str(v.get("wt", ""))
                for k, v in template.get("params", {}).items()
            }
            found.append((name, params))
    return found


def _is_cite(name: str) -> bool:
    return name.startswith("cite ") or name == "citation"


def _is_short(name: str) -> bool:
    return name.startswith(("sfn", "harv"))


def _authors(params: dict[str, str]) -> str | None:
    people: list[str] = []
    for n in ["", *map(str, range(1, 31))]:
        last = params.get(f"last{n}") or params.get(f"surname{n}") or params.get(f"author{n}")
        first = params.get(f"first{n}") or params.get(f"given{n}")
        if last:
            last = _plain(last)
            people.append(f"{last}, {_plain(first)}" if first else last)
    if not people:
        for key in ("authors", "vauthors", "people"):
            if params.get(key):
                people.append(_plain(params[key]))
                break
    return "; ".join(dict.fromkeys(p for p in people if p)) or None


def _fields(params: dict[str, str]) -> dict[str, Any]:
    year_text = _plain(params.get("year") or params.get("date") or "")
    year = re.search(r"\b(\d{4})\b", year_text)
    title = params.get("title") or params.get("chapter") or params.get("work")
    doi = normalize_doi(_plain(params["doi"])) if params.get("doi") else None
    isbn = normalize_isbn(_plain(params["isbn"])) if params.get("isbn") else None
    return {
        "authors": _authors(params),
        "year": int(year.group(1)) if year else None,
        "title": (_plain(title) or None) if title else None,
        "doi": doi,
        "isbn": isbn,
    }


def _text_fields(text: str) -> dict[str, Any]:
    doi = re.search(r"\b(10\.\d{4,9}/\S+)", text)
    isbn = re.search(r"\bISBN[:\s]*([0-9Xx][0-9Xx\- ]{8,16}[0-9Xx])", text)
    year = re.search(r"\((?:[^()]*?)(\d{4})[a-z]?\)", text)
    doi_value = normalize_doi(doi.group(1).rstrip(".,;)")) if doi else None
    isbn_value = normalize_isbn(isbn.group(1)) if isbn else None
    return {
        "year": int(year.group(1)) if year else None,
        "doi": doi_value,
        "isbn": isbn_value,
    }


def _page(params: dict[str, str]) -> str | None:
    for key in PAGE_PARAMS:
        if params.get(key):
            return _plain(params[key]) or None
    return None


@dataclass
class _Bibliography:
    about_templates: dict[str, list[tuple[str, dict[str, str]]]]
    entries: dict[str, LexborNode]

    def resolve(self, fragment: str) -> dict[str, Any] | None:
        node = self.entries.get(fragment)
        if node is None:
            return None
        for name, params in self.about_templates.get(node.attributes.get("about") or "", []):
            if _is_cite(name):
                return {"template": name, "params": params}
        return {"text": " ".join(node.text().split())}


def _bibliography(tree: LexborHTMLParser) -> _Bibliography:
    about_templates: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for n in tree.css("[about][data-mw]"):
        templates = _templates(n)
        if templates:
            about_templates.setdefault(n.attributes["about"] or "", []).extend(templates)
    entries: dict[str, LexborNode] = {}
    for n in tree.css('[id^="CITEREF"]'):
        entries.setdefault(n.attributes["id"] or "", n)
    return _Bibliography(about_templates, entries)


def _citation(ref_key: str, note: LexborNode, bib: _Bibliography) -> ParsedCitation:
    templates = _templates(note)
    cite = next(((n, p) for n, p in templates if _is_cite(n)), None)
    if cite:
        name, params = cite
        return ParsedCitation(
            ref_key, _page(params), {"template": name, "params": params}, **_fields(params)
        )
    short = next(((n, p) for n, p in templates if _is_short(n)), None)
    link = note.css_first('a[href*="#CITEREF"]')
    if short and link:
        name, params = short
        fragment = unquote((link.attributes["href"] or "").split("#", 1)[1])
        raw: dict[str, Any] = {"template": name, "params": params, "citeref": fragment}
        entry = bib.resolve(fragment)
        raw["bibliography"] = entry
        fields: dict[str, Any] = {}
        if entry and "params" in entry:
            fields = _fields(entry["params"])
        elif entry:
            fields = _text_fields(entry["text"])
        return ParsedCitation(ref_key, _page(params), raw, **fields)
    text = " ".join(note.text().split())
    return ParsedCitation(ref_key, None, {"text": text}, **_text_fields(text))


def ref_note_id(sup: LexborNode) -> str | None:
    """Returns the note id a `mw:Extension/ref` marker points to, if any."""
    link = sup.css_first('a[href*="#cite_note"]')
    if link:
        return unquote((link.attributes["href"] or "").split("#", 1)[1])
    try:
        body = json.loads(sup.attributes["data-mw"] or "{}").get("body") or {}
    except ValueError:
        return None
    body_id = body.get("id")
    return body_id.removeprefix(NOTE_PREFIX) if body_id else None


@dataclass
class _Flat:
    tokens: list[tuple[str, Any]] = field(default_factory=list)
    spans: dict[int, tuple[int, int]] = field(default_factory=dict)
    index: dict[int, int] = field(default_factory=dict)
    section_refs: dict[int | None, dict[str, None]] = field(default_factory=dict)


def _flatten(root: LexborNode, notes: Mapping[str, LexborNode], flat: _Flat) -> None:
    sections: list[int | None] = [None]

    def walk(node: LexborNode) -> None:
        for child in node.iter(include_text=True):
            tag = child.tag or ""
            if tag == "-text":
                flat.tokens.append(("text", child.text()))
            elif tag.startswith("-") or tag in ("style", "script"):
                continue
            else:
                typeof = (child.attributes.get("typeof") or "").split()
                if MATH_TYPEOF in typeof:
                    flat.index[child.mem_id] = len(flat.tokens)
                    flat.tokens.append(("math", child))
                elif REF_TYPEOF in typeof and tag == "sup":
                    note_id = ref_note_id(child)
                    if note_id is not None and note_id in notes:
                        flat.tokens.append(("ref", note_id))
                        flat.section_refs.setdefault(sections[-1], {})[note_id] = None
                else:
                    start = len(flat.tokens)
                    flat.tokens.append(("enter", child))
                    if tag == "section":
                        sections.append(child.mem_id)
                    walk(child)
                    if tag == "section":
                        sections.pop()
                    flat.tokens.append(("exit", child))
                    flat.spans[child.mem_id] = (start, len(flat.tokens) - 1)

    walk(root)


def _refs_in(flat: _Flat, start: int, end: int) -> list[str]:
    return [v for k, v in flat.tokens[start:end] if k == "ref"]


def _only_refs(flat: _Flat, start: int, end: int) -> bool:
    for kind, value in flat.tokens[start:end]:
        if kind == "text" and value.strip(" \n\t\xa0" + PUNCTUATION):
            return False
    return True


def _enclosing(node: LexborNode, tags: set[str]) -> LexborNode | None:
    cur = node.parent
    while cur is not None and cur.tag not in tags:
        cur = cur.parent
    return cur


def _element_sibling(node: LexborNode, forward: bool) -> LexborNode | None:
    cur = node.next if forward else node.prev
    while cur is not None and cur.tag == "-text" and not cur.text().strip():
        cur = cur.next if forward else cur.prev
    return cur


def _direct(flat: _Flat, block: LexborNode, formula_index: int) -> list[str]:
    start, end = flat.spans[block.mem_id]
    mathless = [t for t in flat.tokens[start:end] if t[0] != "math"]
    found: list[str] = []
    if all(k != "text" or not v.strip(" \n\t\xa0" + PUNCTUATION) for k, v in mathless):
        found = _refs_in(flat, start, end)
    else:
        for kind, value in flat.tokens[formula_index + 1 : end]:
            if kind == "ref":
                found.append(value)
            elif kind == "math" or (kind == "text" and value.strip(" \n\t\xa0" + PUNCTUATION)):
                return found
    sibling = _element_sibling(block, forward=True)
    j = end + 1
    while j < len(flat.tokens):
        kind, value = flat.tokens[j]
        if kind == "ref":
            found.append(value)
        elif kind == "enter" and value.mem_id == (sibling.mem_id if sibling else None):
            s_, e_ = flat.spans[value.mem_id]
            if _only_refs(flat, s_, e_) and all(k != "math" for k, _ in flat.tokens[s_:e_]):
                found.extend(_refs_in(flat, s_, e_))
            break
        elif kind != "text" or value.strip():
            break
        j += 1
    return found


def _lead_in(flat: _Flat, block: LexborNode) -> list[str]:
    unit = block
    while unit.parent is not None and unit.parent.tag in WRAP_TAGS:
        unit = unit.parent
    if unit.tag == "p":
        return _refs_in(flat, *flat.spans[unit.mem_id])
    previous = _element_sibling(unit, forward=False)
    if previous is not None and previous.tag == "p" and previous.mem_id in flat.spans:
        return _refs_in(flat, *flat.spans[previous.mem_id])
    return []


def parse_citations(html: str, formulas: Mapping[int, bool]) -> ParsedArticle:
    """Extracts the citations of an article and maps formulas to them.

    Every `mw:Extension/ref` marker resolves to its note in the references list. A note with a
    cite template yields the template's fields; a short citation (`sfn`, `harvnb`, ...) is
    resolved to its `CITEREF` bibliography entry and keeps its own page. Reused named refs
    share one citation.

    Formula links: a display formula gets `direct` links from refs inside its block or right
    after it, and `inherited` links from the paragraph just before its block or from the
    paragraph containing it. An inline formula inherits from its own block. A formula with
    neither gets a weaker `section` link to every ref in its section.

    Args:
        html: Parsoid HTML of an article.
        formulas: Position of each formula (as in `FormulaRecord.position`) to whether it is
            display math.

    Returns:
        The citations and formula links.
    """
    tree = LexborHTMLParser(html)
    bib = _bibliography(tree)
    notes: dict[str, LexborNode] = {}
    for span in tree.css(f'[id^="{NOTE_PREFIX}"]'):
        notes.setdefault((span.attributes["id"] or "").removeprefix(NOTE_PREFIX), span)

    flat = _Flat()
    root = tree.root
    if root is None:
        return ParsedArticle()
    _flatten(root, notes, flat)

    used = dict.fromkeys(v for k, v in flat.tokens if k == "ref")
    result = ParsedArticle([_citation(note_id, notes[note_id], bib) for note_id in used])

    positions = {n.mem_id: i for i, n in enumerate(tree.css("*"))}
    nodes = {
        positions[v.mem_id]: v for k, v in flat.tokens if k == "math" and v.mem_id in positions
    }
    for position, display in formulas.items():
        node = nodes.get(position)
        if node is None:
            continue
        block = _enclosing(node, BLOCK_TAGS)
        links: dict[str, str] = {}
        if block is not None and block.mem_id in flat.spans:
            if display:
                for key in _lead_in(flat, block):
                    links[key] = INHERITED
                for key in _direct(flat, block, flat.index[node.mem_id]):
                    links[key] = DIRECT
            else:
                for key in _refs_in(flat, *flat.spans[block.mem_id]):
                    links[key] = INHERITED
        if not links:
            section = _enclosing(node, {"section"})
            for key in flat.section_refs.get(section.mem_id if section else None, {}):
                links[key] = SECTION
        if links:
            result.links[position] = links
    return result

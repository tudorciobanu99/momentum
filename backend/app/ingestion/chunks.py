"""Splitting Parsoid HTML into section-aware text chunks for retrieval."""

import re
from dataclasses import dataclass, field

from selectolax.lexbor import LexborHTMLParser, LexborNode

from app.ingestion.citations import MATH_TYPEOF, REF_TYPEOF, parse_citations, ref_note_id
from app.ingestion.formulas import HEADING_LEVELS, FormulaRecord, extract_formulas, push_heading

MIN_TOKENS = 300
MAX_TOKENS = 500
SPLIT_TOKENS = 200
PIECE_TOKENS = 150
TOKEN = re.compile(r"\w+|[^\w\s]")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

SKIP_SECTIONS = {
    "see also",
    "external links",
    "references",
    "notes",
    "further reading",
    "bibliography",
}
SKIP_TAGS = {"head", "style", "script", "link", "meta", "noscript", "table", "figure"}
SKIP_CLASSES = {
    "hatnote",
    "navbox",
    "reflist",
    "refbegin",
    "references",
    "mw-references-wrap",
    "infobox",
    "sidebar",
    "thumb",
    "metadata",
    "ambox",
    "shortdescription",
    "noprint",
    "mw-empty-elt",
    "navigation-not-searchable",
    "toc",
}
SKIP_TYPEOF = {"mw:Extension/references"}
BLOCK_TAGS = {
    "section",
    "div",
    "p",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    "blockquote",
    "pre",
    "center",
    *HEADING_LEVELS,
}

FORMULA_OPEN, FORMULA_CLOSE = "", ""
CITE_OPEN, CITE_CLOSE = "", ""
FORMULA_MARK = re.compile(f"{FORMULA_OPEN}(\\d+){FORMULA_CLOSE}")
CITE_MARK = re.compile(f"{CITE_OPEN}([^{CITE_CLOSE}]*){CITE_CLOSE}")
DISPLAY_FILLER = re.compile(r"[\s.,;:]*")


@dataclass(frozen=True)
class ChunkRecord:
    """One chunk of an article.

    Attributes:
        index: Zero-based order of the chunk in the article.
        heading_path: Headings from the article title down, joined with " > ".
        text: Chunk text, starting with the heading path.
        token_count: Tokens in `text`, see `count_tokens`.
        formula_indices: `FormulaRecord.index` of every formula in the text.
        citation_keys: `(ref_key, page)` of every citation whose marker is in the text span.
    """

    index: int
    heading_path: str
    text: str
    token_count: int
    formula_indices: tuple[int, ...] = ()
    citation_keys: tuple[tuple[str, str | None], ...] = ()


def count_tokens(text: str) -> int:
    """Counts tokens with a deterministic regex tokenizer.

    A token is a run of word characters or a single other non-space character. It needs no
    model or network and only approximates a model tokenizer.

    Args:
        text: Text to count.

    Returns:
        Number of tokens.
    """
    return len(TOKEN.findall(text))


@dataclass
class _Block:
    text: str
    display_only: bool
    has_display: bool


@dataclass
class _Unit:
    text: str
    tokens: int
    atomic: bool


@dataclass
class _Walk:
    title: str
    records: dict[int, FormulaRecord]
    by_position: dict[int, FormulaRecord]
    position_of: dict[int, int]
    pages: dict[str, str | None]
    headings: list[tuple[int, str]] = field(default_factory=list)
    skipping: int | None = None
    parts: list[str] = field(default_factory=list)
    sections: list[tuple[str, list[_Block]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sections.append((self.title, []))

    def heading(self, tag: str, text: str) -> None:
        self.flush()
        level = push_heading(self.headings, tag, text)
        if self.skipping is not None and level <= self.skipping:
            self.skipping = None
        if self.skipping is None and text.strip().lower() in SKIP_SECTIONS:
            self.skipping = level
        path = " > ".join([self.title, *(h for _, h in self.headings)])
        self.sections.append((path, []))

    def flush(self) -> None:
        raw = "".join(self.parts)
        self.parts = []
        text = " ".join(raw.split())
        if not text or self.skipping is not None:
            return
        bare = CITE_MARK.sub("", text)
        marks = FORMULA_MARK.findall(bare)
        shown = [self.records[int(m)].display for m in marks]
        has_display = any(shown)
        filler = FORMULA_MARK.sub("", bare)
        display_only = has_display and all(shown) and DISPLAY_FILLER.fullmatch(filler) is not None
        self.sections[-1][1].append(_Block(text, display_only, has_display))

    def math(self, node: LexborNode) -> None:
        record = self.by_position.get(self.position_of[node.mem_id])
        if record is None:
            return
        self.parts.append(f" {FORMULA_OPEN}{record.index}{FORMULA_CLOSE} ")

    def visit(self, node: LexborNode) -> None:
        for child in node.iter(include_text=True):
            tag = child.tag or ""
            if tag == "-text":
                self.parts.append(child.text())
                continue
            if tag.startswith("-") or tag in SKIP_TAGS or _skipped(child):
                continue
            typeof = (child.attributes.get("typeof") or "").split()
            if MATH_TYPEOF in typeof:
                self.math(child)
            elif REF_TYPEOF in typeof and tag == "sup":
                note_id = ref_note_id(child)
                if note_id is not None and note_id in self.pages:
                    self.parts.append(f"{CITE_OPEN}{note_id}{CITE_CLOSE}")
            elif tag in HEADING_LEVELS and node.tag == "section":
                self.heading(tag, child.text(strip=True))
            elif tag in BLOCK_TAGS:
                self.flush()
                self.visit(child)
                self.flush()
            else:
                self.visit(child)


def _skipped(node: LexborNode) -> bool:
    attrs = node.attributes
    if SKIP_CLASSES & set((attrs.get("class") or "").split()):
        return True
    if SKIP_TYPEOF & set((attrs.get("typeof") or "").split()):
        return True
    return "display:none" in (attrs.get("style") or "").replace(" ", "")


def _render(text: str, records: dict[int, FormulaRecord]) -> str:
    def formula(match: re.Match[str]) -> str:
        record = records[int(match.group(1))]
        if record.display:
            return f"\n$${record.latex}$$\n"
        return f"${' '.join(record.latex.split())}$"

    out = FORMULA_MARK.sub(formula, CITE_MARK.sub("", text))
    return re.sub(r"\s*\n\s*", "\n", out).strip()


def _pieces(block: _Block, records: dict[int, FormulaRecord]) -> list[str]:
    if block.has_display or count_tokens(_render(block.text, records)) <= SPLIT_TOKENS:
        return [block.text]
    out: list[str] = []
    current = ""
    for sentence in SENTENCE_END.split(block.text):
        joined = f"{current} {sentence}".strip()
        if current and count_tokens(_render(joined, records)) > PIECE_TOKENS:
            out.append(current)
            current = sentence
        else:
            current = joined
    out.append(current)
    return out


def _units(blocks: list[_Block], records: dict[int, FormulaRecord]) -> list[_Unit]:
    units: list[_Unit] = []
    for block in blocks:
        if block.display_only and units:
            previous = units[-1]
            text = f"{previous.text}\n{block.text}"
            units[-1] = _Unit(text, count_tokens(_render(text, records)), True)
            continue
        for piece in _pieces(block, records):
            units.append(_Unit(piece, count_tokens(_render(piece, records)), block.has_display))
    return [u for u in units if u.tokens]


def _pack(units: list[_Unit], cap: int, base: int) -> list[list[_Unit]]:
    chunks: list[list[_Unit]] = []
    i = 0
    while i < len(units):
        current = [chunks[-1][-1]] if chunks else []
        tokens = base + sum(u.tokens for u in current)
        added = 0
        while i < len(units) and (added == 0 or tokens + units[i].tokens <= cap):
            current.append(units[i])
            tokens += units[i].tokens
            added += 1
            i += 1
        chunks.append(current)
    return chunks


def _split(units: list[_Unit], base: int) -> list[list[_Unit]]:
    chunks = _pack(units, MAX_TOKENS, base)
    for cap in range(MAX_TOKENS, MIN_TOKENS - 1, -10):
        candidate = _pack(units, cap, base)
        last = base + sum(u.tokens for u in candidate[-1])
        if len(candidate) == 1 or last >= MIN_TOKENS:
            return candidate
    return chunks


def chunk_article(html: str) -> list[ChunkRecord]:
    """Splits an article into chunks that keep sections, equations and citations intact.

    The document is walked in order with the section stack of `extract_formulas`. Each chunk
    text starts with its heading path ("Article > Section > Subsection"), followed by
    paragraphs separated by blank lines. Inline math is written `$...$`, display math
    `$$...$$` on its own line, and reference markers are dropped.

    Within a section, chunks hold 300 to 500 tokens (see `count_tokens`), counting the heading
    path. A section shorter than 300 tokens is one chunk. Paragraphs over 200 tokens are cut at
    sentence ends. A display equation stays with the paragraph before it, and a paragraph that
    contains display math is never cut, so such a chunk may exceed 500 tokens. A chunk after the
    first begins with the last paragraph (with its equations) of the one before it.

    Skipped as non-content: tables (infoboxes, navboxes, data tables), figures, hatnotes,
    the references list, short-description and other hidden elements, and the sections
    "See also", "External links", "References", "Notes", "Further reading" and "Bibliography"
    with their subsections.

    Formula and citation ids are those of the text span, taken from `extract_formulas` and
    `parse_citations`, so they match what is stored for the article.

    Args:
        html: Parsoid HTML of an article.

    Returns:
        Chunks in document order.
    """
    formulas = extract_formulas(html)
    parsed = parse_citations(html, {f.position: f.display for f in formulas})
    tree = LexborHTMLParser(html)
    root = tree.root
    if root is None:
        return []
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else ""
    walk = _Walk(
        title=title,
        records={f.index: f for f in formulas},
        by_position={f.position: f for f in formulas},
        position_of={n.mem_id: i for i, n in enumerate(tree.css("*"))},
        pages={c.ref_key: c.page for c in parsed.citations},
    )
    walk.visit(root)
    walk.flush()

    records = walk.records
    chunks: list[ChunkRecord] = []
    for path, blocks in walk.sections:
        base = count_tokens(path)
        units = _units(blocks, records)
        if not units:
            continue
        for group in _split(units, base):
            marked = "\n\n".join(u.text for u in group)
            body = "\n\n".join(_render(u.text, records) for u in group)
            text = f"{path}\n\n{body}"
            formula_indices = tuple(dict.fromkeys(int(m) for m in FORMULA_MARK.findall(marked)))
            keys = tuple(dict.fromkeys((k, walk.pages[k]) for k in CITE_MARK.findall(marked)))
            chunks.append(
                ChunkRecord(len(chunks), path, text, count_tokens(text), formula_indices, keys)
            )
    return chunks

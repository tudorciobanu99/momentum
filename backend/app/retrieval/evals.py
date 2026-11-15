"""Retrieval evaluation dataset: models, validation and scoring helpers."""

import re
from collections import Counter
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, ValidationError, field_validator
from selectolax.lexbor import LexborHTMLParser
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Article, ArticleHtml, Chunk, Formula
from app.ingestion.titles import TOPICS

EVALS_PATH = Path(__file__).parents[2] / "evals" / "retrieval.jsonl"
MIN_QUESTIONS = 150
MAX_QUESTIONS = 250
MIN_PER_TOPIC = 15
MIN_FORMULA_SHARE = 0.3
MAX_ANCHOR_CHARS = 300
PARAPHRASE_RUN = 4
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


class GoldPassage(BaseModel):
    """A passage that answers a question.

    Attributes:
        article: Article title.
        rev_id: Pinned revision of the article.
        anchors: Short verbatim sentences or clauses from the article text, at least one. A
            chunk containing any of them answers the question.
    """

    article: str = Field(min_length=1)
    rev_id: int
    anchors: list[Annotated[str, Field(min_length=1, max_length=MAX_ANCHOR_CHARS)]] = Field(
        min_length=1
    )

    @field_validator("anchors")
    @classmethod
    def _not_blank(cls, value: list[str]) -> list[str]:
        if any(not a.strip() for a in value):
            raise ValueError("anchor is blank")
        return value


class GoldFormula(BaseModel):
    """A formula that answers a question.

    Attributes:
        article: Article title.
        rev_id: Pinned revision of the article.
        index: `Formula.index` within the article.
    """

    article: str = Field(min_length=1)
    rev_id: int
    index: int = Field(ge=0)


class EvalQuestion(BaseModel):
    """One retrieval evaluation question.

    Attributes:
        id: Unique id such as `q001`.
        question: Student-style question.
        topic: One of `TOPICS`.
        gold_passages: Passages that answer the question, at least one.
        gold_formulas: Formulas that answer the question, possibly none.
        note: Optional free text.
    """

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    topic: str
    gold_passages: list[GoldPassage] = Field(min_length=1)
    gold_formulas: list[GoldFormula] = []
    note: str | None = None

    @field_validator("topic")
    @classmethod
    def _known_topic(cls, value: str) -> str:
        if value not in TOPICS:
            raise ValueError(f"unknown topic {value!r}")
        return value


def normalise(text: str) -> str:
    """Collapses runs of whitespace to single spaces and trims the ends."""
    return " ".join(text.split())


def shared_run(first: str, second: str, length: int = PARAPHRASE_RUN) -> str | None:
    """Finds a run of consecutive non-stopwords that occurs in both texts.

    Args:
        first: First text.
        second: Second text.
        length: Number of consecutive words that counts as a copy.

    Returns:
        The first shared run in `first`, or None.
    """

    def words(text: str) -> list[str]:
        return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]

    a, b = words(first), words(second)
    runs = {tuple(b[i : i + length]) for i in range(len(b) - length + 1)}
    for i in range(len(a) - length + 1):
        if tuple(a[i : i + length]) in runs:
            return " ".join(a[i : i + length])
    return None


def is_hit(chunk_text: str, gold_passages: list[GoldPassage]) -> bool:
    """Tells whether a retrieved chunk contains any gold anchor.

    Args:
        chunk_text: Text of the retrieved chunk.
        gold_passages: Gold passages of the question.

    Returns:
        True if any anchor occurs in the chunk, ignoring whitespace differences.
    """
    text = normalise(chunk_text)
    return any(normalise(a) in text for p in gold_passages for a in p.anchors)


def parse_questions(content: str) -> tuple[list[EvalQuestion], list[str]]:
    """Parses JSON Lines into questions.

    Args:
        content: File content, one JSON object per line.

    Returns:
        The parsed questions and one problem per line that failed to parse.
    """
    questions: list[EvalQuestion] = []
    problems: list[str] = []
    for number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            questions.append(EvalQuestion.model_validate_json(line))
        except ValidationError as e:
            reasons = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
            problems.append(f"line {number}: {reasons}")
    return questions, problems


def check_schema(questions: list[EvalQuestion]) -> list[str]:
    """Checks dataset-level rules that need no database.

    Args:
        questions: Parsed questions.

    Returns:
        One message per broken rule, empty if the dataset is valid.
    """
    problems: list[str] = []
    if not MIN_QUESTIONS <= len(questions) <= MAX_QUESTIONS:
        problems.append(f"expected {MIN_QUESTIONS}-{MAX_QUESTIONS} questions, got {len(questions)}")
    for label, values in (
        ("id", [q.id for q in questions]),
        ("question", [normalise(q.question) for q in questions]),
    ):
        duplicates = sorted(v for v, n in Counter(values).items() if n > 1)
        if duplicates:
            problems.append(f"duplicate {label}s: {duplicates}")
    for q in questions:
        for p in q.gold_passages:
            for anchor in p.anchors:
                run = shared_run(q.question, anchor)
                if run:
                    problems.append(f"{q.id}: question copies anchor wording: {run!r}")
    per_topic = Counter(q.topic for q in questions)
    for topic in TOPICS:
        if per_topic[topic] < MIN_PER_TOPIC:
            problems.append(f"topic {topic!r} has {per_topic[topic]}, need {MIN_PER_TOPIC}")
    with_formulas = sum(bool(q.gold_formulas) for q in questions)
    if questions and with_formulas / len(questions) < MIN_FORMULA_SHARE:
        problems.append(
            f"{with_formulas} of {len(questions)} questions have gold formulas, "
            f"need {MIN_FORMULA_SHARE:.0%}"
        )
    return problems


def load_questions(path: Path = EVALS_PATH) -> tuple[list[EvalQuestion], list[str]]:
    """Reads the dataset and runs every check that needs no database.

    Args:
        path: JSON Lines file.

    Returns:
        The parsed questions and all schema problems.
    """
    questions, problems = parse_questions(path.read_text(encoding="utf-8"))
    return questions, problems + check_schema(questions)


def check_database(session: Session, questions: list[EvalQuestion]) -> list[str]:
    """Checks every gold passage and formula against the stored articles.

    An anchor must occur in the revision's chunk text or in its stored HTML text.

    Args:
        session: Open session on the database holding the articles.
        questions: Parsed questions.

    Returns:
        One message per missing article, anchor or formula.
    """
    articles = {(a.title, a.rev_id): a.id for a in session.scalars(select(Article))}
    texts: dict[int, tuple[str, str | None]] = {}
    headings: dict[int, list[tuple[str, str]]] = {}
    formulas = {(a, i) for a, i in session.execute(select(Formula.article_id, Formula.index))}

    def article_texts(article_id: int) -> tuple[str, str | None]:
        if article_id not in texts:
            rows = session.execute(
                select(Chunk.text, Chunk.heading_path)
                .where(Chunk.article_id == article_id)
                .order_by(Chunk.index)
            ).all()
            headings[article_id] = [(normalise(t), h) for t, h in rows]
            html = session.get(ArticleHtml, article_id)
            texts[article_id] = (
                normalise("\n".join(t for t, _ in rows)),
                normalise(LexborHTMLParser(html.html).text()) if html else None,
            )
        return texts[article_id]

    problems: list[str] = []
    for q in questions:
        for p in q.gold_passages:
            article_id = articles.get((p.article, p.rev_id))
            if article_id is None:
                problems.append(f"{q.id}: no article {p.article!r} at revision {p.rev_id}")
                continue
            chunk_text, html_text = article_texts(article_id)
            for raw in p.anchors:
                anchor = normalise(raw)
                if anchor not in chunk_text and (html_text is None or anchor not in html_text):
                    problems.append(f"{q.id}: anchor not found in {p.article!r}: {raw[:60]!r}")
                for text, heading in headings[article_id]:
                    run = anchor in text and shared_run(q.question, heading)
                    if run:
                        problems.append(f"{q.id}: question copies heading path: {run!r}")
        for f in q.gold_formulas:
            article_id = articles.get((f.article, f.rev_id))
            if article_id is None:
                problems.append(f"{q.id}: no article {f.article!r} at revision {f.rev_id}")
            elif (article_id, f.index) not in formulas:
                problems.append(f"{q.id}: no formula {f.index} in {f.article!r}")
    return problems

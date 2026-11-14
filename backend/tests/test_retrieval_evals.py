"""Tests for the retrieval evaluation dataset, its validators and the scoring helper."""

import json
from collections.abc import Iterator

import pytest
from sqlalchemy import delete, update
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, ArticleHtml, Chunk, Formula
from app.ingestion.titles import TOPICS
from app.retrieval.evals import (
    EVALS_PATH,
    EvalQuestion,
    GoldPassage,
    check_database,
    check_schema,
    is_hit,
    load_questions,
    parse_questions,
    shared_run,
)


def question(n: int, topic: str = "mechanics", formulas: bool = True, **overrides) -> dict:
    data = {
        "id": f"q{n:03d}",
        "question": f"Question number {n}?",
        "topic": topic,
        "gold_passages": [{"article": "Alpha", "rev_id": 100, "anchor": f"anchor {n}"}],
        "gold_formulas": [{"article": "Alpha", "rev_id": 100, "index": 0}] if formulas else [],
    }
    return data | overrides


def dataset(count: int = 150, formulas: bool = True) -> list[EvalQuestion]:
    return [
        EvalQuestion.model_validate(question(n, TOPICS[n % len(TOPICS)], formulas))
        for n in range(count)
    ]


def test_committed_dataset_is_valid() -> None:
    questions, problems = load_questions(EVALS_PATH)
    assert problems == []
    assert len(questions) >= 150


def test_valid_dataset_has_no_problems() -> None:
    assert check_schema(dataset()) == []


def test_size_bounds() -> None:
    assert "expected 150-250" in check_schema(dataset(149))[0]
    assert "expected 150-250" in check_schema(dataset(251))[0]


def test_duplicate_ids_and_questions() -> None:
    questions = dataset()
    questions[1].id = questions[0].id
    questions[3].question = f"  {questions[2].question.replace('number', 'number  ')}"
    problems = check_schema(questions)
    assert any("duplicate ids" in p for p in problems)
    assert any("duplicate questions" in p for p in problems)


def test_topic_minimum() -> None:
    questions = dataset()
    for q in questions[:120]:
        q.topic = "mechanics"
    assert any("topic 'quantum'" in p for p in check_schema(questions))


def test_formula_share() -> None:
    questions = dataset(formulas=False)
    for q in questions[:44]:
        q.gold_formulas = dataset(1)[0].gold_formulas
    assert any("gold formulas" in p for p in check_schema(questions))
    questions[44].gold_formulas = dataset(1)[0].gold_formulas
    assert check_schema(questions) == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"topic": "cooking"}, "unknown topic"),
        ({"gold_passages": []}, "gold_passages"),
        ({"question": ""}, "question"),
        (
            {"gold_passages": [{"article": "A", "rev_id": 1, "anchor": ""}]},
            "anchor",
        ),
        (
            {"gold_passages": [{"article": "A", "rev_id": 1, "anchor": "   "}]},
            "anchor is blank",
        ),
        (
            {"gold_passages": [{"article": "A", "rev_id": 1, "anchor": "x" * 301}]},
            "anchor",
        ),
    ],
)
def test_invalid_line_is_reported(overrides: dict, expected: str) -> None:
    questions, problems = parse_questions(json.dumps(question(1, **overrides)))
    assert questions == []
    assert len(problems) == 1
    assert problems[0].startswith("line 1:")
    assert expected in problems[0]


def test_malformed_json_is_reported() -> None:
    questions, problems = parse_questions(json.dumps(question(1)) + "\n{not json\n")
    assert len(questions) == 1
    assert len(problems) == 1
    assert problems[0].startswith("line 2:")


def test_anchor_at_limit_is_accepted() -> None:
    gold = [{"article": "A", "rev_id": 1, "anchor": "x" * 300}]
    questions, problems = parse_questions(json.dumps(question(1, gold_passages=gold)))
    assert problems == []
    assert len(questions) == 1


def test_shared_run_ignores_stopwords_and_case() -> None:
    anchor = "The force on a charged particle is proportional to its velocity."
    assert shared_run("Why is the FORCE on charged particle proportional to velocity?", anchor)
    assert (
        shared_run("force charged particle proportional", anchor)
        == "force charged particle proportional"
    )
    assert shared_run("How does force act on a charged body?", anchor) is None
    assert shared_run("", anchor) is None


def test_question_copying_anchor_is_an_error() -> None:
    questions = dataset()
    questions[0].question = "Why is anchor 0 repeated here?"
    questions[0].gold_passages[0].anchor = "one two three four five"
    questions[0].question = "Does two three four five hold?"
    problems = check_schema(questions)
    assert any("q000: question copies anchor wording" in p for p in problems)


PASSAGES = [
    GoldPassage(article="A", rev_id=1, anchor="the force is   proportional"),
    GoldPassage(article="B", rev_id=2, anchor="Energy is conserved"),
]


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        ("Here the force is\nproportional to the mass.", True),
        ("... Energy is conserved in a closed system.", True),
        ("energy is conserved", False),
        ("The force is not proportional", False),
        ("", False),
    ],
)
def test_is_hit(chunk: str, expected: bool) -> None:
    assert is_hit(chunk, PASSAGES) is expected


def test_is_hit_without_passages() -> None:
    assert not is_hit("anything", [])


@pytest.fixture
def db(sessions: sessionmaker[Session]) -> Iterator[sessionmaker[Session]]:
    def clean() -> None:
        with sessions.begin() as s:
            s.execute(delete(Article))

    clean()
    with sessions.begin() as s:
        article = Article(
            title="Alpha", rev_id=100, url="https://example.org/a", topic="mechanics", license="x"
        )
        s.add(article)
        s.flush()
        s.add(ArticleHtml(article_id=article.id, html="<p>Only in <b>the html</b> text.</p>"))
        s.add(
            Chunk(
                article_id=article.id,
                index=0,
                heading_path="Alpha",
                text="Alpha\n\nSecond   law holds here.",
                token_count=5,
            )
        )
        s.add(
            Formula(
                article_id=article.id,
                index=0,
                latex="F=ma",
                display=True,
                section_path="",
                position=1,
            )
        )
    yield sessions
    clean()


def check(db: sessionmaker[Session], **overrides) -> list[str]:
    q = EvalQuestion.model_validate(question(1, **overrides))
    with db() as s:
        return check_database(s, [q])


def test_database_check_accepts_known_gold(db: sessionmaker[Session]) -> None:
    passages = [
        {"article": "Alpha", "rev_id": 100, "anchor": "Second law holds"},
        {"article": "Alpha", "rev_id": 100, "anchor": "Only in the html text."},
    ]
    assert check(db, gold_passages=passages) == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"gold_passages": [{"article": "Alpha", "rev_id": 100, "anchor": "second law"}]},
            "anchor not found",
        ),
        (
            {"gold_passages": [{"article": "Alpha", "rev_id": 999, "anchor": "Second law"}]},
            "no article 'Alpha' at revision 999",
        ),
        (
            {
                "gold_passages": [{"article": "Alpha", "rev_id": 100, "anchor": "Second law"}],
                "gold_formulas": [{"article": "Alpha", "rev_id": 100, "index": 5}],
            },
            "no formula 5",
        ),
        (
            {
                "gold_passages": [{"article": "Alpha", "rev_id": 100, "anchor": "Second law"}],
                "gold_formulas": [{"article": "Gamma", "rev_id": 100, "index": 0}],
            },
            "no article 'Gamma'",
        ),
    ],
)
def test_database_check_reports_problems(
    db: sessionmaker[Session], overrides: dict, expected: str
) -> None:
    problems = check(db, **overrides)
    assert len(problems) == 1
    assert expected in problems[0]


def test_database_check_rejects_question_copying_heading(db: sessionmaker[Session]) -> None:
    passages = [{"article": "Alpha", "rev_id": 100, "anchor": "Second law holds"}]
    with db.begin() as s:
        s.execute(
            update(Chunk).values(heading_path="Alpha > Rotating frames and fictitious forces")
        )
    problems = check(
        db, gold_passages=passages, question="What are rotating frames fictitious forces?"
    )
    assert len(problems) == 1
    assert "copies heading path" in problems[0]

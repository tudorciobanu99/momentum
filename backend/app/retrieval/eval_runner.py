"""Run the retrieval evaluation: hybrid search, rerank, metrics and reports."""

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, Chunk, ChunkFormula, Formula
from app.retrieval.client import EmbeddingClient, Reranker
from app.retrieval.corpus import MERGED_SCHEMA, build_merged_corpus, corpus_session, drop_schema
from app.retrieval.evals import EvalQuestion, is_hit, normalise
from app.retrieval.hybrid import hybrid_search
from app.retrieval.langsmith_results import DATASET_NAME, ResultsClient
from app.retrieval.rerank import rerank_candidates

RECALL_K = 20
NDCG_K = 10
FORMULA_K = 5
SECTION_CREDIT = 0.5
WORST_COUNT = 10
TOP_PATHS = 3

FormulaKey = tuple[str, int, int]
SectionKey = tuple[int, str]


@dataclass(frozen=True)
class Retrieved:
    """A retrieved chunk as the metrics see it.

    Attributes:
        chunk_id: Chunk id.
        heading_path: Heading path of the chunk.
        text: Chunk text.
        rev_id: Revision of the chunk's article.
        formulas: `(article title, rev_id, formula index)` of the formulas linked to the chunk.
    """

    chunk_id: int
    heading_path: str
    text: str
    rev_id: int
    formulas: frozenset[FormulaKey] = frozenset()


@dataclass(frozen=True)
class QuestionOutcome:
    """What retrieval returned for one question.

    Attributes:
        question: The question.
        fused: Fused candidates, best first.
        ranked: Reranked chunks, best first.
        gold_sections: Per gold passage, the `(rev_id, heading_path)` of the stored chunks that
            contain one of its anchors. Passages beyond the list have no section.
    """

    question: EvalQuestion
    fused: list[Retrieved]
    ranked: list[Retrieved]
    gold_sections: list[frozenset[SectionKey]] = field(default_factory=list)


@dataclass(frozen=True)
class QuestionScore:
    """Scores of one question.

    Attributes:
        question: The question.
        recall: At least one gold passage is in the fused top `RECALL_K`.
        passages_found: Gold passages found in the fused top `RECALL_K`.
        passages_total: Gold passages of the question.
        ndcg: nDCG over the reranked top `NDCG_K`, with partial credit for the right section.
        ndcg_strict: nDCG over the reranked top `NDCG_K`, anchor hits only.
        reciprocal_rank: Best credit / rank in the reranked top `NDCG_K`, 0 if none.
        reciprocal_rank_strict: 1 / rank of the first anchor hit in the reranked top `NDCG_K`,
            0 if none.
        formula_hit: A gold formula is linked to the reranked top `FORMULA_K`, None if the
            question has no gold formula.
        ranked: Reranked chunks, best first.
        hits: Whether each reranked chunk contains a gold anchor.
    """

    question: EvalQuestion
    recall: bool
    passages_found: int
    passages_total: int
    ndcg: float
    ndcg_strict: float
    reciprocal_rank: float
    reciprocal_rank_strict: float
    formula_hit: bool | None
    ranked: list[Retrieved]
    hits: list[bool]


@dataclass(frozen=True)
class Metrics:
    """Metrics over a set of questions.

    Attributes:
        questions: Number of questions.
        recall_at_20: Share of questions with a gold passage in the fused top 20.
        passage_recall_at_20: Gold passages found in the fused top 20 over all gold passages.
        ndcg_at_10: Mean nDCG of the reranked top 10, with section credit.
        ndcg_at_10_strict: Mean nDCG of the reranked top 10, anchor hits only.
        mrr: Mean reciprocal rank of the reranked top 10, with section credit.
        mrr_strict: Mean reciprocal rank of the reranked top 10, anchor hits only.
        formula_hit_at_5: Share of questions with a gold formula among the top 5 reranked chunks'
            formulas, None if no question has a gold formula.
        formula_questions: Questions with a gold formula, the denominator of `formula_hit_at_5`.
    """

    questions: int
    recall_at_20: float
    passage_recall_at_20: float
    ndcg_at_10: float
    ndcg_at_10_strict: float
    mrr: float
    mrr_strict: float
    formula_hit_at_5: float | None
    formula_questions: int

    def as_dict(self) -> dict[str, float]:
        """Returns the metrics by name, leaving out an undefined Formula Hit@5."""
        out = {
            "recall@20": self.recall_at_20,
            "passage_recall@20": self.passage_recall_at_20,
            "ndcg@10": self.ndcg_at_10,
            "ndcg@10_strict": self.ndcg_at_10_strict,
            "mrr": self.mrr,
            "mrr_strict": self.mrr_strict,
        }
        if self.formula_hit_at_5 is not None:
            out["formula_hit@5"] = self.formula_hit_at_5
        return out


@dataclass(frozen=True)
class WorstQuestion:
    """A question that retrieval served badly.

    Attributes:
        id: Question id.
        question: Question text.
        anchors: Gold anchors.
        reciprocal_rank: Reciprocal rank of the first hit.
        top_paths: Heading paths of the top reranked chunks.
    """

    id: str
    question: str
    anchors: list[str]
    reciprocal_rank: float
    top_paths: list[str]


@dataclass(frozen=True)
class Report:
    """Metrics overall and by group, and the worst questions.

    Attributes:
        overall: Metrics over all questions.
        by_topic: Metrics per topic, sorted by topic.
        by_type: Metrics per question type, sorted by type.
        worst: The `WORST_COUNT` lowest-scoring questions.
    """

    overall: Metrics
    by_topic: dict[str, Metrics]
    by_type: dict[str, Metrics]
    worst: list[WorstQuestion]


def question_type(question: EvalQuestion) -> str:
    """Returns the type from the note, the text before the first semicolon."""
    if not question.note:
        return "unknown"
    return question.note.split(";")[0].strip() or "unknown"


def _credits(outcome: QuestionOutcome, ranked: list[Retrieved]) -> list[list[float]]:
    """Returns the credit of every ranked chunk for every gold passage."""
    credits: list[list[float]] = []
    for n, p in enumerate(outcome.question.gold_passages):
        sections = outcome.gold_sections[n] if n < len(outcome.gold_sections) else frozenset()
        anchors = [normalise(a) for a in p.anchors]
        row: list[float] = []
        for c in ranked:
            text = normalise(c.text)
            if any(a in text for a in anchors):
                row.append(1.0)
            elif (c.rev_id, c.heading_path) in sections:
                row.append(SECTION_CREDIT)
            else:
                row.append(0.0)
        credits.append(row)
    return credits


def _assigned_gains(credits: list[list[float]], hits: list[bool]) -> list[float]:
    """Credits each passage once at its best, then earliest, position.

    A position never gains less than 1 when it holds an anchor hit, so the result is never below
    the strict gains.
    """
    gains = [0.0] * len(hits)
    for row in credits:
        best = max(row, default=0.0)
        if best > 0:
            gains[row.index(best)] += best
    return [max(g, 1.0 if h else 0.0) for g, h in zip(gains, hits, strict=True)]


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, 1))


def score_question(outcome: QuestionOutcome) -> QuestionScore:
    """Scores the retrieval for one question.

    Strict relevance is binary: a chunk is relevant if it contains any gold anchor. The ideal DCG
    has one relevant chunk per gold passage, at most `NDCG_K`, and nDCG is capped at 1.

    Credited relevance gives a chunk 1 for an anchor of a gold passage and `SECTION_CREDIT` for
    sitting in the section of a gold passage without containing an anchor. Each gold passage is
    credited once, at its best then earliest position, and a chunk that hits several passages
    credits each. A position holding an anchor hit gains at least 1, so credited scores are never
    below strict ones. The reciprocal rank uses the best credit of any passage.

    Args:
        outcome: Retrieval results.

    Returns:
        The scores.
    """
    q = outcome.question
    gold = q.gold_passages
    fused = outcome.fused[:RECALL_K]
    found = sum(any(is_hit(c.text, [p]) for c in fused) for p in gold)
    ranked = outcome.ranked[:NDCG_K]
    hits = [is_hit(c.text, gold) for c in ranked]
    ideal = _dcg([1.0] * min(len(gold), NDCG_K))
    strict_dcg = _dcg([1.0 if h else 0.0 for h in hits])
    credits = _credits(outcome, ranked)
    gains = _assigned_gains(credits, hits)
    best = [max((row[pos] for row in credits), default=0.0) for pos in range(len(ranked))]
    gold_formulas = {(f.article, f.rev_id, f.index) for f in q.gold_formulas}
    formula_hit = (
        any(gold_formulas & c.formulas for c in ranked[:FORMULA_K]) if gold_formulas else None
    )
    return QuestionScore(
        question=q,
        recall=found > 0,
        passages_found=found,
        passages_total=len(gold),
        ndcg=min(1.0, _dcg(gains) / ideal),
        ndcg_strict=min(1.0, strict_dcg / ideal),
        reciprocal_rank=max((c / i for i, c in enumerate(best, 1)), default=0.0),
        reciprocal_rank_strict=next((1 / i for i, h in enumerate(hits, 1) if h), 0.0),
        formula_hit=formula_hit,
        ranked=ranked,
        hits=hits,
    )


def summarize(scores: list[QuestionScore]) -> Metrics:
    """Aggregates question scores.

    Args:
        scores: Scores to aggregate.

    Returns:
        The metrics, all zero for no scores. Questions without gold formulas are left out of the
        Formula Hit@5 denominator.
    """
    n = len(scores)
    with_formulas = [s for s in scores if s.formula_hit is not None]
    total_passages = sum(s.passages_total for s in scores)
    return Metrics(
        questions=n,
        recall_at_20=sum(s.recall for s in scores) / n if n else 0.0,
        passage_recall_at_20=(
            sum(s.passages_found for s in scores) / total_passages if total_passages else 0.0
        ),
        ndcg_at_10=sum(s.ndcg for s in scores) / n if n else 0.0,
        ndcg_at_10_strict=sum(s.ndcg_strict for s in scores) / n if n else 0.0,
        mrr=sum(s.reciprocal_rank for s in scores) / n if n else 0.0,
        mrr_strict=sum(s.reciprocal_rank_strict for s in scores) / n if n else 0.0,
        formula_hit_at_5=(
            sum(bool(s.formula_hit) for s in with_formulas) / len(with_formulas)
            if with_formulas
            else None
        ),
        formula_questions=len(with_formulas),
    )


def build_report(scores: list[QuestionScore]) -> Report:
    """Builds the report: overall, per topic, per type and the worst questions.

    Args:
        scores: Scores of all questions.

    Returns:
        The report, with groups sorted by name and worst questions by reciprocal rank, then id.
    """

    def group(key: Callable[[QuestionScore], str]) -> dict[str, Metrics]:
        grouped: dict[str, list[QuestionScore]] = defaultdict(list)
        for s in scores:
            grouped[key(s)].append(s)
        return {k: summarize(grouped[k]) for k in sorted(grouped)}

    worst = sorted(scores, key=lambda s: (s.reciprocal_rank, s.question.id))[:WORST_COUNT]
    return Report(
        overall=summarize(scores),
        by_topic=group(lambda s: s.question.topic),
        by_type=group(lambda s: question_type(s.question)),
        worst=[
            WorstQuestion(
                id=s.question.id,
                question=s.question.question,
                anchors=[a for p in s.question.gold_passages for a in p.anchors],
                reciprocal_rank=s.reciprocal_rank,
                top_paths=[c.heading_path for c in s.ranked[:TOP_PATHS]],
            )
            for s in worst
        ],
    )


def question_outputs(score: QuestionScore) -> dict[str, Any]:
    """Builds the per-question outputs logged to LangSmith."""
    return {
        "chunk_ids": [c.chunk_id for c in score.ranked],
        "heading_paths": [c.heading_path for c in score.ranked],
        "hits": score.hits,
        "recall@20": score.recall,
        "ndcg@10": score.ndcg,
        "ndcg@10_strict": score.ndcg_strict,
        "reciprocal_rank": score.reciprocal_rank,
        "reciprocal_rank_strict": score.reciprocal_rank_strict,
        "formula_hit@5": score.formula_hit,
    }


def _load(session: Session, ids: set[int]) -> dict[int, Retrieved]:
    chunk_ids = sorted(ids)
    rows = session.execute(
        select(Chunk.id, Chunk.heading_path, Chunk.text, Article.rev_id)
        .join(Article, Article.id == Chunk.article_id)
        .where(Chunk.id.in_(chunk_ids))
    )
    formulas: dict[int, set[FormulaKey]] = defaultdict(set)
    for chunk_id, title, rev_id, index in session.execute(
        select(ChunkFormula.chunk_id, Article.title, Article.rev_id, Formula.index)
        .join(Formula, Formula.id == ChunkFormula.formula_id)
        .join(Article, Article.id == Formula.article_id)
        .where(ChunkFormula.chunk_id.in_(chunk_ids))
    ):
        formulas[chunk_id].add((title, rev_id, index))
    return {
        chunk_id: Retrieved(chunk_id, path, text, rev_id, frozenset(formulas[chunk_id]))
        for chunk_id, path, text, rev_id in rows
    }


def _gold_sections(session: Session, question: EvalQuestion) -> list[frozenset[SectionKey]]:
    """Finds the sections of the stored chunks that contain an anchor, per gold passage."""
    sections: list[frozenset[SectionKey]] = []
    for p in question.gold_passages:
        anchors = [normalise(a) for a in p.anchors]
        rows = session.execute(
            select(Chunk.heading_path, Chunk.text)
            .join(Article, Article.id == Chunk.article_id)
            .where(Article.title == p.article, Article.rev_id == p.rev_id)
        )
        sections.append(
            frozenset(
                (p.rev_id, path)
                for path, text in rows
                if any(a in normalise(text) for a in anchors)
            )
        )
    return sections


def collect(
    session: Session,
    client: EmbeddingClient,
    reranker: Reranker,
    questions: list[EvalQuestion],
) -> list[QuestionOutcome]:
    """Runs hybrid search and reranking for every question.

    Args:
        session: Session on the corpus to evaluate.
        client: Embedding client.
        reranker: Reranking client.
        questions: Questions to run.

    Returns:
        One outcome per question, in order.

    Raises:
        RateLimitError: If a service is rate limiting.
        RerankError: If reranking fails.
    """
    outcomes: list[QuestionOutcome] = []
    for q in questions:
        candidates = hybrid_search(session, client, q.question)
        ranked = rerank_candidates(session, reranker, q.question, candidates, NDCG_K)
        chunks = _load(session, {c.chunk_id for c in candidates} | {r.chunk_id for r in ranked})
        outcomes.append(
            QuestionOutcome(
                question=q,
                fused=[chunks[c.chunk_id] for c in candidates],
                ranked=[chunks[r.chunk_id] for r in ranked],
                gold_sections=_gold_sections(session, q),
            )
        )
    return outcomes


def evaluate_corpus(
    sessions: sessionmaker[Session],
    client: EmbeddingClient,
    reranker: Reranker,
    questions: list[EvalQuestion],
    schema: str | None = None,
) -> list[QuestionScore]:
    """Scores every question against a corpus.

    Args:
        sessions: Session factory.
        client: Embedding client.
        reranker: Reranking client.
        questions: Questions to run.
        schema: Schema of the corpus, None for `public`.

    Returns:
        One score per question, in order.
    """
    with corpus_session(sessions, schema) as session:
        outcomes = collect(session, client, reranker, questions)
    return [score_question(o) for o in outcomes]


@dataclass(frozen=True)
class EvalRun:
    """Outcome of an evaluation run.

    Attributes:
        reports: Report per variant, in print order.
        uploaded: Whether the baseline experiment was uploaded.
    """

    reports: dict[str, Report]
    uploaded: bool


def run_evals(
    sessions: sessionmaker[Session],
    client: EmbeddingClient,
    reranker: Reranker,
    questions: list[EvalQuestion],
    results: ResultsClient | None = None,
    compare_merged: bool = False,
    experiment: str = "momentum",
) -> EvalRun:
    """Evaluates retrieval, optionally against a merged-chunks variant, and uploads the results.

    The merged variant is built in a temporary schema that is dropped afterwards, also on error.

    Args:
        sessions: Session factory.
        client: Embedding client.
        reranker: Reranking client.
        questions: Questions to run.
        results: Where to upload the experiments, None to skip uploading.
        compare_merged: Also evaluate a chunking with short chunks merged.
        experiment: Experiment name prefix.

    Returns:
        The reports, `baseline` first and `merged` second if requested.
    """
    variants: dict[str, list[QuestionScore]] = {
        "baseline": evaluate_corpus(sessions, client, reranker, questions)
    }
    if compare_merged:
        try:
            build_merged_corpus(sessions, client, MERGED_SCHEMA)
            variants["merged"] = evaluate_corpus(
                sessions, client, reranker, questions, MERGED_SCHEMA
            )
        finally:
            drop_schema(sessions, MERGED_SCHEMA)
    reports = {name: build_report(scores) for name, scores in variants.items()}
    if results is not None:
        results.sync_dataset(DATASET_NAME, questions)
        for name, scores in variants.items():
            results.log_experiment(
                DATASET_NAME,
                f"{experiment}-{name}",
                {s.question.id: question_outputs(s) for s in scores},
                reports[name].overall.as_dict(),
            )
    return EvalRun(reports, results is not None)


_COLUMNS = ("n", "R@20", "pR@20", "nDCG@10", "nDCG@10s", "MRR", "MRRs", "FH@5")


def _cells(m: Metrics) -> list[str]:
    fh = "-" if m.formula_hit_at_5 is None else f"{m.formula_hit_at_5:.3f}"
    return [
        str(m.questions),
        f"{m.recall_at_20:.3f}",
        f"{m.passage_recall_at_20:.3f}",
        f"{m.ndcg_at_10:.3f}",
        f"{m.ndcg_at_10_strict:.3f}",
        f"{m.mrr:.3f}",
        f"{m.mrr_strict:.3f}",
        fh,
    ]


def _table(title: str, rows: dict[str, dict[str, Metrics]]) -> list[str]:
    variants = list(next(iter(rows.values())))
    label_width = max([len(title), *(len(label) for label in rows)])
    header = [title.ljust(label_width)]
    for v in variants:
        header.append(
            " ".join(c.rjust(7) for c in _COLUMNS) + (f"  <- {v}" if len(variants) > 1 else "")
        )
    lines = [" | ".join(header)]
    for label, by_variant in rows.items():
        cells = [label.ljust(label_width)]
        for v in variants:
            cells.append(" ".join(c.rjust(7) for c in _cells(by_variant[v])))
        lines.append(" | ".join(cells))
    return lines


def format_reports(reports: dict[str, Report]) -> str:
    """Formats reports as text, variants side by side in each table.

    Args:
        reports: Report per variant name, at least one.

    Returns:
        The text to print.
    """
    names = list(reports)
    lines = _table("overall", {"all": {n: reports[n].overall for n in names}})
    for title, pick in (
        ("topic", lambda r: r.by_topic),
        ("type", lambda r: r.by_type),
    ):
        keys = sorted({k for n in names for k in pick(reports[n])})
        empty = summarize([])
        rows = {k: {n: pick(reports[n]).get(k, empty) for n in names} for k in keys}
        if rows:
            lines += ["", *_table(title, rows)]
    for name in names:
        lines += ["", f"worst questions ({name})"]
        for w in reports[name].worst:
            lines.append(f"{w.id}  rr={w.reciprocal_rank:.3f}  {w.question}")
            lines.append(f"    gold: {' | '.join(w.anchors)}")
            for path in w.top_paths:
                lines.append(f"    got: {path}")
    return "\n".join(lines)

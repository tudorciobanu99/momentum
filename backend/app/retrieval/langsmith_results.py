"""Upload retrieval evaluation results to LangSmith, with a fake for tests."""

from typing import Any, Protocol

from langsmith import Client, evaluate
from langsmith.evaluation import EvaluationResult, EvaluationResults

from app.retrieval.evals import EvalQuestion

DATASET_NAME = "momentum-retrieval"


class ResultsClient(Protocol):
    """Stores evaluation datasets and experiments."""

    def sync_dataset(self, name: str, questions: list[EvalQuestion]) -> None:
        """Creates the dataset if missing and adds the questions it does not hold yet.

        Args:
            name: Dataset name.
            questions: Questions, identified by `EvalQuestion.id`.
        """
        ...

    def log_experiment(
        self,
        dataset: str,
        experiment: str,
        outputs: dict[str, dict[str, Any]],
        metrics: dict[str, float],
    ) -> None:
        """Logs one experiment on a dataset.

        Args:
            dataset: Dataset name.
            experiment: Experiment name prefix.
            outputs: Per-question outputs by question id.
            metrics: Summary metrics by name.
        """
        ...


class LangSmithResults:
    """Uploads with the LangSmith SDK.

    Args:
        api_key: LangSmith API key.
    """

    def __init__(self, api_key: str) -> None:
        self._client = Client(api_key=api_key)

    def sync_dataset(self, name: str, questions: list[EvalQuestion]) -> None:
        if not self._client.has_dataset(dataset_name=name):
            self._client.create_dataset(name, description="Momentum retrieval evaluation questions")
        known = {
            str((e.inputs or {}).get("id")) for e in self._client.list_examples(dataset_name=name)
        }
        new = [q for q in questions if q.id not in known]
        if not new:
            return
        self._client.create_examples(
            dataset_name=name,
            examples=[
                {
                    "inputs": {"id": q.id, "question": q.question},
                    "outputs": {
                        "gold_passages": [p.model_dump() for p in q.gold_passages],
                        "gold_formulas": [f.model_dump() for f in q.gold_formulas],
                    },
                    "metadata": {"topic": q.topic, "note": q.note},
                }
                for q in new
            ],
        )

    def log_experiment(
        self,
        dataset: str,
        experiment: str,
        outputs: dict[str, dict[str, Any]],
        metrics: dict[str, float],
    ) -> None:
        def target(inputs: dict[str, Any]) -> dict[str, Any]:
            return outputs[inputs["id"]]

        def summary(runs: Any, examples: Any) -> EvaluationResults:
            return EvaluationResults(
                results=[EvaluationResult(key=k, score=v) for k, v in metrics.items()]
            )

        evaluate(
            target,
            data=dataset,
            summary_evaluators=[summary],
            experiment_prefix=experiment,
            client=self._client,
        )


class FakeResults:
    """In-memory results client that records its calls."""

    def __init__(self) -> None:
        self.datasets: dict[str, list[EvalQuestion]] = {}
        self.experiments: list[tuple[str, str, dict[str, dict[str, Any]], dict[str, float]]] = []

    def sync_dataset(self, name: str, questions: list[EvalQuestion]) -> None:
        held = self.datasets.setdefault(name, [])
        known = {q.id for q in held}
        held.extend(q for q in questions if q.id not in known)

    def log_experiment(
        self,
        dataset: str,
        experiment: str,
        outputs: dict[str, dict[str, Any]],
        metrics: dict[str, float],
    ) -> None:
        self.experiments.append((dataset, experiment, outputs, metrics))

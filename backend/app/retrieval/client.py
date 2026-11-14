"""Embedding and rerank client interfaces, the Voyage implementations and fakes for tests."""

from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

import voyageai.error
from voyageai.client import Client

InputType = Literal["document", "query"]


class RateLimitError(Exception):
    """The embedding service asked the caller to slow down.

    Attributes:
        retry_after: Seconds the service asked to wait, if it said.
    """

    def __init__(self, message: str = "rate limited", retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RerankError(Exception):
    """The reranking service failed or returned an unusable result."""


class EmbeddingClient(Protocol):
    """Turns texts into vectors."""

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        """Embeds texts.

        Args:
            texts: Texts to embed.
            input_type: Whether the texts are documents or search queries.

        Returns:
            One vector per text, in order.

        Raises:
            RateLimitError: If the service is rate limiting.
        """
        ...


class VoyageClient:
    """Embeds with the Voyage API.

    Args:
        api_key: Voyage API key.
        model: Voyage model name.
        dimension: Output dimension.
    """

    def __init__(self, api_key: str, model: str, dimension: int) -> None:
        self._client = Client(api_key=api_key, max_retries=0)
        self._model = model
        self._dimension = dimension

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        try:
            result = self._client.embed(
                texts,
                model=self._model,
                input_type=input_type,
                output_dimension=self._dimension,
            )
        except voyageai.error.RateLimitError as e:
            raise RateLimitError(str(e), _retry_after(e.headers)) from e
        return [[float(x) for x in v] for v in result.embeddings]


class Reranker(Protocol):
    """Scores documents against a query."""

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """Reranks documents.

        Args:
            query: Search query.
            documents: Documents to score.
            top_n: Number of best documents to return.

        Returns:
            `(index into documents, score)` pairs, best first, at most `top_n`.

        Raises:
            RateLimitError: If the service is rate limiting.
            RerankError: If the service fails.
        """
        ...


class VoyageReranker:
    """Reranks with the Voyage API.

    Args:
        api_key: Voyage API key.
        model: Voyage rerank model name.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._client = Client(api_key=api_key, max_retries=0)
        self._model = model

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        try:
            result = self._client.rerank(query, documents, model=self._model, top_k=top_n)
        except voyageai.error.RateLimitError as e:
            raise RateLimitError(str(e), _retry_after(e.headers)) from e
        except voyageai.error.VoyageError as e:
            raise RerankError(str(e)) from e
        return [(int(r.index), float(r.relevance_score)) for r in result.results]


def _retry_after(headers: Mapping[str, Any]) -> float | None:
    try:
        return max(0.0, float(headers["retry-after"]))
    except (KeyError, TypeError, ValueError):
        return None


class FakeClient:
    """Deterministic in-memory client that records its calls.

    Args:
        dimension: Length of each returned vector.
        failures: Errors to raise on the first calls, one per entry.
        vector: Builds the vector for a text, by default a function of its length.
    """

    def __init__(
        self,
        dimension: int,
        failures: list[RateLimitError] | None = None,
        vector: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.dimension = dimension
        self.failures = list(failures or [])
        self.vector = vector or (lambda t: [float(len(t) % 97 + 1)] * dimension)
        self.calls: list[tuple[list[str], InputType]] = []

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        if self.failures:
            raise self.failures.pop(0)
        self.calls.append((list(texts), input_type))
        return [self.vector(t) for t in texts]


class FakeReranker:
    """Deterministic in-memory reranker that records its calls.

    Args:
        score: Scores a query and a document, by default the number of shared words.
        error: Raised on every call if set.
    """

    def __init__(
        self,
        score: Callable[[str, str], float] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.score = score or _shared_words
        self.error = error
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if self.error:
            raise self.error
        self.calls.append((query, list(documents), top_n))
        scored = [(i, self.score(query, d)) for i, d in enumerate(documents)]
        scored.sort(key=lambda p: (-p[1], p[0]))
        return scored[:top_n]


def _shared_words(query: str, document: str) -> float:
    return float(len(set(query.lower().split()) & set(document.lower().split())))

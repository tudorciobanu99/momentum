"""Rate-limited HTTP client for the Wikipedia APIs."""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

BASE_URL = "https://en.wikipedia.org"
USER_AGENT = f"Momentum/0.1 (https://github.com/tudorciobanu99/momentum) httpx/{httpx.__version__}"


class RateLimitError(Exception):
    """Raised when a request is still answered with 429 after all retries."""


class WikiClient:
    """Async client capping concurrency and request rate, retrying on 429.

    Args:
        max_concurrent: Maximum requests in flight.
        max_per_minute: Maximum requests started in any 60 second window.
        max_retries: Retries after a 429 before giving up.
        transport: Optional httpx transport, used to mock HTTP in tests.
        clock: Monotonic clock in seconds.
        sleep: Awaitable sleep taking seconds.
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        max_per_minute: int = 200,
        max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"User-Agent": USER_AGENT},
            transport=transport,
            timeout=30,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_lock = asyncio.Lock()
        self._starts: deque[float] = deque()
        self._max_per_minute = max_per_minute
        self._max_retries = max_retries
        self._clock = clock
        self._sleep = sleep

    async def __aenter__(self) -> "WikiClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._http.aclose()

    async def _wait_for_slot(self) -> None:
        async with self._rate_lock:
            while True:
                now = self._clock()
                while self._starts and now - self._starts[0] >= 60:
                    self._starts.popleft()
                if len(self._starts) < self._max_per_minute:
                    self._starts.append(now)
                    return
                await self._sleep(60 - (now - self._starts[0]))

    async def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """Sends a GET, honouring `Retry-After` on 429 up to `max_retries` times.

        Args:
            url: Path relative to the Wikipedia base URL.
            params: Optional query parameters.

        Returns:
            The final response. Redirects are not followed.

        Raises:
            RateLimitError: If every attempt was answered with 429.
        """
        for attempt in range(self._max_retries + 1):
            async with self._semaphore:
                await self._wait_for_slot()
                response = await self._http.get(url, params=params)
            if response.status_code != 429:
                return response
            if attempt < self._max_retries:
                await self._sleep(_retry_after(response))
        raise RateLimitError(f"429 from {url} after {self._max_retries} retries")


def _retry_after(response: httpx.Response) -> float:
    try:
        return max(0.0, float(response.headers["Retry-After"]))
    except (KeyError, ValueError):
        return 1.0

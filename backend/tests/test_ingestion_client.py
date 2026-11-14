"""Tests for the rate-limited Wikipedia client, with HTTP mocked."""

import asyncio

import httpx
import pytest

from app.ingestion.client import USER_AGENT, RateLimitError, WikiClient


class FakeTime:
    """Virtual clock whose sleep advances time instantly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


def test_user_agent_header() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["User-Agent"])
        return httpx.Response(200)

    async def run() -> None:
        async with WikiClient(transport=httpx.MockTransport(handler)) as client:
            await client.get("/x")

    asyncio.run(run())
    assert seen == [USER_AGENT]
    assert USER_AGENT.startswith("Momentum/0.1 (https://github.com/tudorciobanu99/momentum) httpx/")
    assert USER_AGENT.split("httpx/")[1] == httpx.__version__


def test_concurrency_capped_at_three() -> None:
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200)

    async def run() -> None:
        async with WikiClient(transport=httpx.MockTransport(handler)) as client:
            await asyncio.gather(*(client.get(f"/p{i}") for i in range(12)))

    asyncio.run(run())
    assert peak == 3


def test_rate_limit_waits_for_window() -> None:
    fake = FakeTime()
    starts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(fake.now)
        return httpx.Response(200)

    async def run() -> None:
        async with WikiClient(
            max_per_minute=200,
            transport=httpx.MockTransport(handler),
            clock=fake.clock,
            sleep=fake.sleep,
        ) as client:
            await asyncio.gather(*(client.get(f"/p{i}") for i in range(250)))

    asyncio.run(run())
    assert starts.count(0.0) == 200
    assert len(starts) == 250
    assert min(s for s in starts if s > 0) == pytest.approx(60)


def test_429_honours_retry_after_then_succeeds() -> None:
    fake = FakeTime()
    responses = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(429, headers={"Retry-After": "2"}),
        httpx.Response(200, text="ok"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    async def run() -> httpx.Response:
        async with WikiClient(
            transport=httpx.MockTransport(handler), clock=fake.clock, sleep=fake.sleep
        ) as client:
            return await client.get("/x")

    response = asyncio.run(run())
    assert response.text == "ok"
    assert fake.sleeps == [7, 2]


def test_429_gives_up_after_bounded_retries() -> None:
    fake = FakeTime()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "1"})

    async def run() -> None:
        async with WikiClient(
            max_retries=3,
            transport=httpx.MockTransport(handler),
            clock=fake.clock,
            sleep=fake.sleep,
        ) as client:
            await client.get("/x")

    with pytest.raises(RateLimitError):
        asyncio.run(run())
    assert calls == 4

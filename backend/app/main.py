"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI( title = "Momentum" )


@app.get("/api/health")
def health() -> dict[str, str]:
    """Reports that the service is alive.

    Returns:
        A payload with status "ok".
    """
    return {"status": "ok"}

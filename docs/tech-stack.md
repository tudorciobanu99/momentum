# Tech stack

## Backend

| Area | Choice |
|---|---|
| Language | Python 3.13, managed with uv |
| Orchestration | LangGraph 1.2.x |
| API | FastAPI |
| Live updates | Server-sent events (SSE)|
| Database | Postgres with pgvector |
| ORM | SQLAlchemy 2 |
| Postgres driver | psycopg 3 |
| Migrations | Alembic |
| Local services | Docker Compose |
| Embeddings | Voyage `voyage-4` |
| Reranker | Voyage `rerank-3-lite` |
| LLMs | Provider agnostic |
| Tracing and evals | LangSmith |
| Sandbox | Modal Sandboxes |
| Symbolics and numerics | SymPy and SciPy |

## Frontend

| Area | Choice |
|---|---|
| Framework | Vite, React, TypeScript |
| Package manager | pnpm |
| Animation | Canvas 2D |
| Plots | uPlot |
| Agent graph | React Flow |
| Formulas | KaTeX |

## Dev tooling

| Side | Tools |
|---|---|
| Python | ruff (lint and format), pyright, pytest |
| TypeScript | Biome (lint and format), `tsc --noEmit`, Vitest |

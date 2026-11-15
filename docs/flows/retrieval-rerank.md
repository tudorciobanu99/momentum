# Retrieval: rerank

```mermaid
flowchart TD
    A[rerank_candidates query, candidates, top_n] --> B{Candidates empty}
    B -- yes --> C[Empty list, no reranker call]
    B -- no --> D[Load chunk text, article and heading path]
    D --> E[Voyage rerank-3-lite, top_k = all candidates]
    E --> F{Error}
    F -- rate limit --> G[RateLimitError]
    F -- other --> H[RerankError]
    F -- no --> I[Order by rerank score desc, chunk id]
    I --> K[Drop chunks sharing article and heading path with a better-ranked chunk]
    K --> L[Truncate to top_n, default 10]
    L --> J[Return chunk id, rerank score, fused score, ranks]
```

# Retrieval: eval runner

```mermaid
flowchart TD
    A[python -m app.retrieval evals run] --> B[Load and check questions, --limit N]
    B --> C[Per question: hybrid search, up to 30 fused candidates]
    C --> D[Rerank, top 10]
    D --> E[Hit = chunk text contains a gold anchor]
    E --> F[Recall@20 and passage recall on fused top 20]
    E --> G[nDCG@10 and MRR on reranked top 10, binary relevance]
    E --> H[Formula Hit@5: gold formula among formulas of reranked top 5, questions without gold formulas excluded]
    F --> I[Report: overall, per topic, per type, 10 worst questions]
    G --> I
    H --> I
    A --> J{--compare-merged}
    J -- yes --> K[Schema eval_merged: merge consecutive chunks under 300 tokens within a top-level section, union formula links, embed as documents, same indexes]
    K --> L[Run the same eval with search_path on eval_merged]
    L --> M[Drop schema, also on error]
    M --> I
    I --> N{LANGSMITH_API_KEY and not --no-upload}
    N -- yes --> O[Sync dataset momentum-retrieval by question id, log one experiment per variant]
    N -- no --> P[Warn, print only]
    I --> Q[Print report, variants side by side]
```

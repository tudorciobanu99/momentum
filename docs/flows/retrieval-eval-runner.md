# Retrieval: eval runner

```mermaid
flowchart TD
    A[python -m app.retrieval evals run] --> B[Load and check questions, --limit N; gold section = revision and heading path of stored chunks containing an anchor of the passage]
    B --> C[Per question: hybrid search, up to 30 fused candidates]
    C --> D[One rerank call of all candidates, same-section duplicates dropped, top 10]
    D --> E[Hit = chunk text contains any anchor of a gold passage]
    E --> F[Recall@20 and passage recall on fused top 20]
    E --> G[nDCG@10 and MRR on reranked top 10: hit 1.0, chunk in a gold passage's section without an anchor 0.5, each passage credited once at its best position; strict variants count hits only]
    E --> H[Formula Hit@5: gold formula among formulas of reranked top 5, questions without gold formulas excluded]
    F --> I[Report: overall, per topic, per type, 10 worst questions]
    G --> I
    H --> I
    A --> R{--compare-dedup}
    R -- yes --> S[From the same rerank call, variants dedup and nodedup: nodedup is the sorted list without dropping duplicates, top 10]
    S --> I
    A --> J{--compare-merged}
    J -- yes --> K[Schema eval_merged: merge consecutive chunks under 300 tokens within a top-level section, union formula links, embed as documents, same indexes]
    K --> L[Run the same eval with search_path on eval_merged]
    L --> M[Drop schema, also on error]
    M --> I
    I --> N{LANGSMITH_API_KEY and not --no-upload}
    N -- yes --> O[Sync dataset momentum-retrieval by question id, log one experiment per variant, name prefix-variant]
    N -- no --> P[Warn, print only]
    I --> Q[Print report, variants side by side]
```

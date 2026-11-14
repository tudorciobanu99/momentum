# Retrieval: lexical

```mermaid
flowchart TD
    A[lexical_search query, limit] --> B{Query blank}
    B -- yes --> C[Empty list]
    B -- no --> D[websearch_to_tsquery english]
    D --> E[Match chunks.search_vector, GIN index]
    E --> F[Rank with ts_rank_cd]
    F --> G[Order by rank desc, chunk id]
    G --> H[Return chunk id and rank, up to limit]
```

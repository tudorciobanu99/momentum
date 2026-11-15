# Retrieval: evals

```mermaid
flowchart TD
    A[python -m app.retrieval evals validate] --> B[Read evals/retrieval.jsonl]
    B --> C[Parse lines into EvalQuestion]
    C --> D[Check ids, topics, sizes, formula share, question not copying any anchor, each gold passage has 1 or more anchors]
    A --> E[Read articles, chunks, article_html, formulas]
    E --> F[Find every anchor in chunk text or HTML text, question not copying heading path]
    E --> G[Find each gold formula by article, revision, index]
    D --> H[Print problems]
    F --> H
    G --> H
    H --> I[Exit 1 if any problem]
```

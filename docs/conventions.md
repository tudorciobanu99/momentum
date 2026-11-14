# Conventions

## Comments

| Do | Don't |
|---|---|
| Explain why, not what | Restate the code |
| Note non-obvious intent | Leave commented-out code |
| State physics and maths assumptions | Record change history |
| State units | Leave TODOs, tasks live in Linear |
| Explain workarounds | |

## Docs in code

| Side | Choice |
|---|---|
| Python | Google-style docstrings on public modules, classes and functions |
| TypeScript | TSDoc `/** */` on exports, `//` for inline notes |

## Formatting

- Line length 100 on both sides
- Otherwise ruff and Biome defaults

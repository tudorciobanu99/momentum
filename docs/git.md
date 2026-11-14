# Git

## Types

| Type | Use |
|---|---|
| `feat` | New behaviour |
| `fix` | Bug fix |
| `chore` | Tooling, config, dependencies |
| `docs` | Documentation |

## Branching

Format: `type/id-slug`, for example `feat/mom-12-planner-node`

- All work happens on a branch
- Nothing is committed directly to `main` without the owner's approval
- One story per branch, branched from `main`

## Commits

Format: `type(surface): summary`, for example `feat(frontend): add planner node`

| Surface | Covers |
|---|---|
| `frontend` | Web app |
| `backend` | API and pipeline |
| `infra` | Docker, CI, tooling |

- Lowercase
- Committed under the owner's git identity
- No attribution or co-author lines
- Commits on a branch need no approval

## Pull requests

Title: same format as a commit

Body: [.github/pull_request_template.md](../.github/pull_request_template.md)

| Section | Holds |
|---|---|
| Story | `Closes MOM-12` |
| Changes | What changed |
| Acceptance criteria | Copied from the story, all ticked |
| How to test | Steps to see it working |

- Squash merge only
- Delete the branch after merging

## Protection on `main`

| Rule | Setting |
|---|---|
| Required checks | `backend`, `frontend` |
| Direct pushes | Blocked |
| Bypass | None |
| Up to date with `main` | Not required |

## Hooks

```sh
brew install lefthook
lefthook install
```

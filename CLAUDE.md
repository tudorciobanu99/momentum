# Momentum

A multi-agent physics companion that takes you from an idea to understanding, through simulations you can see and change.

## Read first

| Doc | Before |
|---|---|
| [docs/workflow.md](docs/workflow.md) | Starting any work |
| [docs/tech-stack.md](docs/tech-stack.md) | Adding a dependency or choosing a tool |
| [docs/git.md](docs/git.md) | Creating a branch, commit or pull request |
| [docs/conventions.md](docs/conventions.md) | Writing code or comments |

## Communication

- Answer first. No preamble, no restating the question, no closing recap.
- Short by default. Expand only when asked.
- One recommendation, not a survey of options, unless options are requested.
- Ask only when blocked on a decision that is genuinely the user's.

## Delegation

- The main session coordinates. It does not do bulk reading or large edits itself.
- Delegate anything that would dump large output into the main conversation.
- Subagent reports: ≤10 lines unless the agent defines otherwise, `file:line` refs, no raw logs.

## Tasks

- Tasks live in Linear, never in the repo. No TODO or plan files.

## Docs

- Check library docs via Context7 before using an API.
- Update [docs/api.md](docs/api.md) when an interface is added or changed.
- Add or update a flow in [docs/flows/](docs/flows/README.md) when a feature is added or changed.

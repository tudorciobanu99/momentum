#!/usr/bin/env python3
"""PreToolUse hook: a pull request to `main` must follow the template.

Checks the title format and that every section of
.github/pull_request_template.md is present and filled in. See docs/git.md.
"""
import json
import os
import re
import shlex
import sys

BASE = "main"
TEMPLATE = ".github/pull_request_template.md"
TITLE = re.compile(r"^(feat|fix|chore|docs)\([a-z][a-z-]*\): tud-\d+ \S.*$")
PLACEHOLDERS = {"", "-", "- [ ]", "1.", "closes tud-"}


def deny(reasons: list[str]) -> None:
    message = "Pull request does not follow docs/git.md:\n- " + "\n- ".join(reasons)
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": message,
        }
    }))
    sys.exit(0)


def option(tokens: list[str], *names: str) -> str | None:
    for i, token in enumerate(tokens):
        for name in names:
            if token == name and i + 1 < len(tokens):
                return tokens[i + 1]
            if name.startswith("--") and token.startswith(name + "="):
                return token[len(name) + 1:]
    return None


def sections(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current = heading.group(1)
            found[current] = ""
        elif current is not None:
            found[current] += line + "\n"
    return found


def is_empty(body: str) -> bool:
    lines = [line.strip().lower() for line in body.splitlines()]
    return all(line in PLACEHOLDERS for line in lines)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    command = (data.get("tool_input") or {}).get("command") or ""
    start = re.search(r"\bgh\s+pr\s+create\b", command)
    if not start:
        sys.exit(0)

    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    try:
        tokens = shlex.split(command[start.end():])
    except ValueError:
        deny(["the command could not be parsed; pass --title and --body-file"])

    base = option(tokens, "--base", "-B")
    if base and base != BASE:
        sys.exit(0)

    reasons: list[str] = []

    title = option(tokens, "--title", "-t")
    if not title:
        reasons.append("no --title given")
    elif not TITLE.match(title) or title != title.lower():
        reasons.append(f"title must be `type(surface): id summary`, lowercase; got `{title}`")

    body = option(tokens, "--body", "-b")
    body_file = option(tokens, "--body-file", "-F")
    if body is None and body_file and body_file != "-":
        path = body_file if os.path.isabs(body_file) else os.path.join(data.get("cwd") or root, body_file)
        try:
            with open(path, encoding="utf-8") as handle:
                body = handle.read()
        except OSError:
            reasons.append(f"body file not found: {body_file}")
    if body is None:
        reasons.append("no --body or --body-file given")
        deny(reasons)

    try:
        with open(os.path.join(root, TEMPLATE), encoding="utf-8") as handle:
            required = list(sections(handle.read()))
    except OSError:
        required = ["Story", "Changes", "Acceptance criteria", "How to test"]

    found = sections(body)
    for name in required:
        if name not in found:
            reasons.append(f"missing section `## {name}`")
        elif is_empty(found[name]):
            reasons.append(f"section `## {name}` is empty")

    if "Story" in found and not re.search(r"Closes TUD-\d+", found["Story"]):
        reasons.append("`## Story` must contain `Closes TUD-<number>`")

    criteria = found.get("Acceptance criteria", "")
    if re.search(r"^\s*- \[ \]", criteria, re.M):
        reasons.append("every acceptance criterion must be ticked")
    elif "Acceptance criteria" in found and not re.search(r"^\s*- \[[xX]\]", criteria, re.M):
        reasons.append("`## Acceptance criteria` needs at least one ticked item")

    if reasons:
        deny(reasons)
    sys.exit(0)


if __name__ == "__main__":
    main()

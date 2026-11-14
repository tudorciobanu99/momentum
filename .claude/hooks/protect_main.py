#!/usr/bin/env python3
"""PreToolUse hook: anything that changes `main` needs the owner's approval.

Asks before a commit, merge or push that lands on `main`, and before
`gh pr merge`. Denies a `gh pr merge` that is not a squash with branch deletion.
See docs/git.md.
"""
import json
import re
import subprocess
import sys

PROTECTED = "main"
WRITE_OPS = r"commit|merge|cherry-pick|revert|rebase|am|pull"


def decide(decision: str, reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def current_branch(cwd: str) -> str:
    try:
        out = subprocess.run(
            ["git", "symbolic-ref", "--short", "-q", "HEAD"],
            cwd=cwd or None, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def branch_after_switches(command: str, start: str) -> str:
    """Branch in effect once any `git checkout`/`git switch` in the command ran."""
    branch = start
    for match in re.finditer(r"\bgit\s+(?:checkout|switch)\b([^|;&\n]*)", command):
        args = [a for a in match.group(1).split() if not a.startswith("-")]
        if args:
            branch = args[0]
    return branch


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    command = (data.get("tool_input") or {}).get("command") or ""
    if not re.search(r"\b(git|gh)\b", command):
        sys.exit(0)

    merge = re.search(r"\bgh\s+pr\s+merge\b([^|;&\n]*)", command)
    if merge:
        flags = merge.group(1).split()
        squash = "--squash" in flags or "-s" in flags
        delete = "--delete-branch" in flags or "-d" in flags
        if not (squash and delete):
            decide("deny", "Pull requests merge by squash only, and the branch is "
                           "deleted: use `gh pr merge --squash --delete-branch`.")
        decide("ask", f"This merges a pull request into `{PROTECTED}`. "
                      "The owner's approval is required.")

    branch = branch_after_switches(command, current_branch(data.get("cwd", "")))

    if branch == PROTECTED and re.search(rf"\bgit\b[^|;&\n]*\s({WRITE_OPS})\b", command):
        decide("ask", f"This changes `{PROTECTED}` directly. "
                      "The owner's approval is required.")

    push = re.search(r"\bgit\b[^|;&\n]*\spush\b([^|;&\n]*)", command)
    if push:
        targets = [a for a in push.group(1).split() if not a.startswith("-")]
        refs = [t.split(":")[-1] for t in targets[1:]]
        to_main = PROTECTED in refs or (not refs and branch == PROTECTED)
        if to_main:
            decide("ask", f"This pushes to `{PROTECTED}`. "
                          "The owner's approval is required.")

    sys.exit(0)


if __name__ == "__main__":
    main()

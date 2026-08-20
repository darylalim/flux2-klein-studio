#!/usr/bin/env bash
# PostToolUse hook: flag that the test suite should run at end of turn.
#
# The full suite is expensive to run after every edit, so instead of running it
# here this hook just drops a marker when an edit touches something the suite
# asserts on. The Stop hook (run-tests.sh) consumes the marker and runs the
# gates once, at the end of the turn.
#
# The covered set is DERIVED, not transcribed. It used to be a hand-maintained
# allow-list of thirteen globs that had to be kept in step with what the nine
# test modules read — a coupling with two defects. It was fail-OPEN: any tracked
# path outside those globs armed nothing until someone remembered to edit this
# file, and the local loop went silently dark. (Verified against the old script:
# a new `assets/logo.svg` or `scripts/build.py` did not arm. A new module under
# `tests/` did, via the `*/tests/*` glob — the failure was new *directories*,
# not new files in known ones.) And its test checked the list against a
# hardcoded copy of the same list, so it could not fail except in lockstep with
# the script it mirrored.
#
# `git check-ignore` replaces it with external truth. tests/test_secrets.py
# scans `git ls-files`, so EVERY tracked file is already suite-relevant — the
# covered set is not a curated union of modules, it is simply "not gitignored".
# That makes the rule fail-CLOSED: an unrecognized path arms the suite rather
# than skipping it, so a new test module can never again land unrun locally.
# What stays excluded is what git already ignores: .venv/, __pycache__/, the
# caches, a local .env, .claude/settings.local.json, and the marker itself.
set -uo pipefail

command -v jq >/dev/null 2>&1 || exit 0

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
[[ -n "$file" ]] || exit 0

proj="${CLAUDE_PROJECT_DIR:-.}"

# Only in-project files.
case "$file" in
  "${proj%/}"/*) ;;
  *) exit 0 ;;
esac

# git's own plumbing is not project source; nothing asserts on it.
case "$file" in
  "${proj%/}"/.git/*) exit 0 ;;
esac

# Arm unless git ignores the path. Exit 0 means ignored (skip); 1 (not ignored)
# and 128 (no git, or no repo) both fall through and arm — fail closed.
if git -C "${proj%/}" check-ignore -q -- "$file" 2>/dev/null; then
  exit 0
fi

: > "${proj%/}/.claude/.tests-pending" 2>/dev/null || true
exit 0

#!/usr/bin/env bash
# PostToolUse hook: auto-format and lint-fix the edited Python file with ruff.
#
# Matcher fires on Edit|Write|MultiEdit for *every* file, so this script does the
# path filtering itself: it only touches real .py files inside the project.
# Non-blocking by design (always exits 0) — it corrects the file silently and
# leaves any non-autofixable lint for the ty/pytest gates or a manual `ruff check`.
set -uo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

# Only act on Python files that still exist on disk.
[[ "$file" == *.py && -f "$file" ]] || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

uv run ruff format "$file" >/dev/null 2>&1
uv run ruff check --fix "$file" >/dev/null 2>&1
exit 0

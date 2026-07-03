#!/usr/bin/env bash
# PostToolUse hook: auto-format and lint-fix the edited Python file with ruff.
#
# Matcher fires on Edit|Write|MultiEdit for *every* file, so this script does the
# path filtering itself: it only touches real .py files that live inside the
# project tree (an edit can target an absolute path anywhere; without the
# containment check this would reformat out-of-repo files). Non-blocking by
# design (always exits 0) — it corrects the file silently and leaves any
# non-autofixable lint for the ty/pytest gates or a manual `ruff check`.
set -uo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

# Only act on Python files that still exist on disk.
[[ "$file" == *.py && -f "$file" ]] || exit 0

# Only act on files inside the project tree (matches the comment above).
proj="${CLAUDE_PROJECT_DIR:-.}"
case "$file" in
  "${proj%/}"/*) ;;
  *) exit 0 ;;
esac

cd "$proj" || exit 0

uv run ruff format "$file" >/dev/null 2>&1
uv run ruff check --fix "$file" >/dev/null 2>&1
exit 0

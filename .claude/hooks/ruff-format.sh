#!/usr/bin/env bash
# PostToolUse hook: auto-format the edited file with ruff (and lint-fix Python).
#
# Matcher fires on Edit|Write|MultiEdit for *every* file, so this script does the
# path filtering itself: it only touches real .py/.md files that live inside the
# project tree (an edit can target an absolute path anywhere; without the
# containment check this would reformat out-of-repo files). Non-blocking by
# design (always exits 0) — it corrects the file silently and leaves any
# non-autofixable lint for the ty/pytest gates or a manual `ruff check`.
#
# Markdown is in scope because `ruff format` (>=0.16) formats Python fenced in
# .md and CI runs `ruff format --check .` from the repo root: without this, a
# mis-formatted block in README.md/CLAUDE.md would be auto-fixed by nothing
# locally and fail only in CI. `ruff check` has no Markdown support, so the
# lint-fix pass stays Python-only. Both globs are lowercase on purpose — ruff
# recognizes Markdown by a lowercase `.md` alone and parses any other extension
# handed to it as Python source, so routing a `README.MD` here would only make
# ruff fail to parse it (silently, since output is discarded).
set -uo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

# Only act on files ruff can format that still exist on disk.
[[ ( "$file" == *.py || "$file" == *.md ) && -f "$file" ]] || exit 0

# Only act on files inside the project tree (matches the comment above).
proj="${CLAUDE_PROJECT_DIR:-.}"
case "$file" in
  "${proj%/}"/*) ;;
  *) exit 0 ;;
esac

cd "$proj" || exit 0

uv run ruff format "$file" >/dev/null 2>&1
if [[ "$file" == *.py ]]; then
  uv run ruff check --fix "$file" >/dev/null 2>&1
fi
exit 0

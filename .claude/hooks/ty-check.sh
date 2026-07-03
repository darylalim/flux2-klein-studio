#!/usr/bin/env bash
# PostToolUse hook: type-check the project with ty after a Python edit.
#
# Runs only when a .py file inside the project is edited. On failure it exits 2,
# feeding ty's output back to Claude as an error to fix. A clean check exits 0
# silently. It type-checks the whole project (`ty check .`, matching CLAUDE.md),
# so it also catches cross-file breakage — the repo is expected to stay green.
# The surfaced header is exit-code-agnostic ("ty check failed") so an
# environmental failure (e.g. uv missing) is not mislabeled as a type error.
set -uo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

[[ "$file" == *.py ]] || exit 0

# Only run for edits inside the project tree.
proj="${CLAUDE_PROJECT_DIR:-.}"
case "$file" in
  "${proj%/}"/*) ;;
  *) exit 0 ;;
esac

cd "$proj" || exit 0

if ! out=$(uv run ty check . 2>&1); then
  printf 'ty check failed:\n%s\n' "$out" >&2
  exit 2
fi
exit 0

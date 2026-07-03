#!/usr/bin/env bash
# PostToolUse hook: type-check the project with ty after a Python edit.
#
# Runs only when a .py file was edited. On type errors it exits 2, which feeds
# ty's output back to Claude as an error so it fixes the type issue it just
# introduced. A clean check exits 0 silently. Because it type-checks the whole
# project (`ty check .`, matching CLAUDE.md), a pre-existing error anywhere would
# also surface — the repo is expected to stay green.
set -uo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

[[ "$file" == *.py ]] || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

if ! out=$(uv run ty check . 2>&1); then
  printf 'ty reported type errors:\n%s\n' "$out" >&2
  exit 2
fi
exit 0

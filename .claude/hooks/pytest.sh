#!/usr/bin/env bash
# PostToolUse hook: run the test suite after edits to the app or test code.
#
# Scoped narrowly (streamlit_app.py or anything under tests/) so it does not fire
# on docs/config edits. On failure it exits 2 to surface the failing tests back to
# Claude; a green run exits 0 silently.
set -uo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
[[ -n "$file" ]] || exit 0

case "$file" in
  *streamlit_app.py | */tests/*) ;;
  *) exit 0 ;;
esac

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

if ! out=$(uv run pytest -q 2>&1); then
  printf 'pytest failures after this edit:\n%s\n' "$out" >&2
  exit 2
fi
exit 0

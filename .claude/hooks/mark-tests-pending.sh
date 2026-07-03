#!/usr/bin/env bash
# PostToolUse hook: flag that the test suite should run at end of turn.
#
# The full suite is expensive to run after every edit, so instead of running it
# here this hook just drops a marker when an edit touches code the suite covers
# (streamlit_app.py, tests/, or the load-bearing .streamlit/config.toml whose
# WCAG contrast is asserted by TestThemeConfig). The Stop hook (run-tests.sh)
# consumes the marker and runs pytest once, at the end of the turn.
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

# Only files the suite actually covers.
case "$file" in
  *streamlit_app.py | */tests/* | */.streamlit/config.toml) ;;
  *) exit 0 ;;
esac

: > "${proj%/}/.claude/.tests-pending" 2>/dev/null || true
exit 0

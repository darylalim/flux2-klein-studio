#!/usr/bin/env bash
# PostToolUse hook: flag that the test suite should run at end of turn.
#
# The full suite is expensive to run after every edit, so instead of running it
# here this hook just drops a marker when an edit touches something the suite
# asserts on. The Stop hook (run-tests.sh) consumes the marker and runs pytest
# once, at the end of the turn.
#
# The covered set below is the union of what the test modules read, and must be
# kept in step with them or an edit lands with the suite never run locally (CI
# still catches it; the local loop just goes quiet):
#
#   streamlit_app.py, .streamlit/config.toml -> test_streamlit_app.py
#                                               (TestThemeConfig asserts the
#                                               theme's WCAG contrast)
#   tests/                                   -> all of them
#   .github/workflows/, .github/release.yml  -> test_ci / test_release /
#                                               test_workflows
#   .claude/settings.json, .claude/hooks/    -> test_hooks.py
#   README.md, docs/, examples/              -> test_readme.py (it asserts every
#                                               local image the README embeds
#                                               exists on disk)
#   pyproject.toml, LICENSE                  -> test_license.py, test_ci.py
#   .python-version                          -> test_ci.py
#
# CLAUDE.md is deliberately absent: no test reads it. It is covered by `ruff
# format --check`, which is ruff-format.sh's job, not pytest's.
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

# Only files the suite actually covers (see the map above).
case "$file" in
  *streamlit_app.py | */tests/* | */.streamlit/config.toml) ;;
  */.github/workflows/* | */.github/release.yml) ;;
  */.claude/settings.json | */.claude/hooks/*) ;;
  */README.md | */docs/* | */examples/*) ;;
  */pyproject.toml | */LICENSE | */.python-version) ;;
  *) exit 0 ;;
esac

: > "${proj%/}/.claude/.tests-pending" 2>/dev/null || true
exit 0

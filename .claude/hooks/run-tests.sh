#!/usr/bin/env bash
# Stop hook: run the test suite once per turn, if an edit flagged it pending.
#
# mark-tests-pending.sh drops a marker when relevant code is edited; this hook
# consumes it and runs the full suite. Exit 2 blocks the turn from ending and
# feeds the failure back to Claude to fix. Clearing the marker BEFORE running
# makes it loop-safe: a follow-up turn that edits nothing relevant leaves no
# marker, so this exits 0 and the conversation can end. (Stop hooks receive no
# tool payload and take no matcher.)
set -uo pipefail

proj="${CLAUDE_PROJECT_DIR:-.}"
marker="${proj%/}/.claude/.tests-pending"

[[ -f "$marker" ]] || exit 0
rm -f "$marker"

cd "$proj" || exit 0

if ! out=$(uv run pytest -q 2>&1); then
  printf 'pytest run failed (the suite is gated at end of turn):\n%s\n' "$out" >&2
  exit 2
fi
exit 0

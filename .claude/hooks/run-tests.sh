#!/usr/bin/env bash
# Stop hook: the turn's gate. Runs ruff always, then ty + pytest if pending.
#
# This is where every *whole-project* check lives, because each of them derives
# one fact about the whole tree and PostToolUse would re-derive it once per
# edit. An 8-edit turn paid `ty check .` eight times (0.58s each) to learn the
# same thing eight times, and surfaced a transient exit 2 on every intermediate
# edit of a multi-file refactor — an error Claude was already mid-way through
# fixing. Here each runs once per turn.
#
# Two tiers, deliberately:
#
#   * `ruff format .` runs UNCONDITIONALLY, above the marker gate. It is pure
#     layout — idempotent and incapable of losing work — so it is safe to run
#     unvalidated, and it is the only thing covering a file written by a Bash
#     heredoc or redirect, which fires no PostToolUse hook at all.
#   * `ruff check --fix .` runs BELOW the gate, because it is not safe in that
#     sense: it DELETES code (an unused import, an unreachable branch) across
#     the whole tree, including files this turn never touched. Above the gate it
#     would rewrite code on a turn where ty and pytest never run, so nothing
#     would ever validate what it did. Below the gate its output is checked by
#     the two gates that follow it.
#   * ty and pytest run only when an edit armed the marker, because together
#     they cost ~14s and most turns should not pay it.
#
# Lint-fix runs BEFORE format, and the order is load-bearing: `ruff check --fix`
# removing an unused import leaves behind the blank line that followed it, so
# the formatter has to run last or `ruff format --check .` (CI's gate) rejects
# the result. The reverse order was a real, verified bug.
#
# Exit 2 blocks the turn from ending and feeds the failure back to Claude to
# fix. Clearing the marker BEFORE running makes it loop-safe: a follow-up turn
# that edits nothing relevant leaves no marker, so this exits 0 and the
# conversation can end. (Stop hooks receive no tool payload and take no matcher.)
set -uo pipefail

proj="${CLAUDE_PROJECT_DIR:-.}"
marker="${proj%/}/.claude/.tests-pending"

cd "$proj" || exit 0

# Layout only: safe to run unvalidated, and covers Bash-written files.
uv run ruff format . >/dev/null 2>&1

[[ -f "$marker" ]] || exit 0
rm -f "$marker"

# Semantic: --fix deletes code, so it runs here where ty and pytest below will
# validate the result. Format again after it — removing an unused import strands
# the blank line that followed it, which `ruff format --check .` (CI's gate)
# rejects. Formatter last is the load-bearing half of this pair.
uv run ruff check --fix . >/dev/null 2>&1
uv run ruff format . >/dev/null 2>&1

# Fail fast: a type error is cheaper to report than a 14s suite run.
# The header is exit-code-agnostic ("ty check failed") so an environmental
# failure (e.g. uv missing) is not mislabeled as a type error.
if ! out=$(uv run ty check . 2>&1); then
  printf 'ty check failed:\n%s\n' "$out" >&2
  exit 2
fi

if ! out=$(uv run pytest -q 2>&1); then
  printf 'pytest run failed (the suite is gated at end of turn):\n%s\n' "$out" >&2
  exit 2
fi
exit 0

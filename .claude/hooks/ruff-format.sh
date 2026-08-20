#!/usr/bin/env bash
# PostToolUse hook: format the edited file with ruff.
#
# Formatting ONLY. The lint-fix pass (`ruff check --fix`) deliberately lives in
# the Stop hook instead, for two independent reasons:
#
#   * Ordering. `ruff check --fix` removing an unused import leaves behind the
#     blank line that followed it, which `ruff format --check .` — CI's gate —
#     then rejects. Verified: `import contextlib\n\nx = 1\n` run through format
#     then check --fix yields `\nx = 1\n`, which fails CI while this hook exits
#     0 silently. The formatter must run LAST, and pairing them here in the
#     wrong order was exactly that bug.
#   * Semantics. `--fix` deletes code. An import added in one edit and first
#     used several edits later is unused in between, so a per-edit `--fix`
#     races an in-progress refactor and removes work.
#
# `ruff format` has neither problem: it is pure layout, idempotent, and cannot
# lose work, so it is safe to run on every edit and keeps the file canonical for
# the next read. Non-blocking by design (always exits 0).
#
# Matcher fires on Edit|Write|MultiEdit for *every* file, so this script does the
# path filtering itself: it only touches real files inside the project tree (an
# edit can target an absolute path anywhere; without the containment check this
# would reformat out-of-repo files).
#
# The extension list mirrors what CI's `ruff format --check .` walks from the
# repo root — .py, .pyi, .md and .ipynb. (Markdown is in scope because ruff
# >=0.16 formats Python fenced inside it.)
#
# Every glob is lowercase on purpose, and for .ipynb that is a safety property
# rather than tidiness. Ruff picks its parser from the extension it is handed and
# parses anything it does not recognize as Python source — and JSON is valid
# Python expression syntax. On a case-insensitive filesystem (macOS's default)
# `ruff format Notebook.IPYNB` therefore opens the real notebook, parses its JSON
# as a dict literal and rewrites it as formatted Python, leaving a file that is
# no longer valid JSON. A mis-cased path must miss these globs entirely.
set -uo pipefail

command -v jq >/dev/null 2>&1 || exit 0

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

# Only act on files ruff can format...
case "$file" in
  *.py | *.pyi | *.md | *.ipynb) ;;
  *) exit 0 ;;
esac

# ...that still exist on disk.
[[ -f "$file" ]] || exit 0

# Only act on files inside the project tree (matches the comment above).
proj="${CLAUDE_PROJECT_DIR:-.}"
case "$file" in
  "${proj%/}"/*) ;;
  *) exit 0 ;;
esac

cd "$proj" || exit 0

uv run ruff format "$file" >/dev/null 2>&1
exit 0

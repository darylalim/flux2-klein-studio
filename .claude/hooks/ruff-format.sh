#!/usr/bin/env bash
# PostToolUse hook: auto-format the edited file with ruff (and lint-fix it).
#
# Matcher fires on Edit|Write|MultiEdit for *every* file, so this script does the
# path filtering itself: it only touches real files inside the project tree (an
# edit can target an absolute path anywhere; without the containment check this
# would reformat out-of-repo files). Non-blocking by design (always exits 0) — it
# corrects the file silently and leaves any non-autofixable lint for the
# ty/pytest gates or a manual `ruff check`.
#
# The extension list mirrors what CI's `ruff format --check .` walks from the
# repo root — .py, .pyi, .md and .ipynb — so a file CI would flag is auto-fixed
# here instead of failing only in CI. (Markdown is in scope because ruff >=0.16
# formats Python fenced inside it.) The lint-fix pass covers all of those except
# .md: `ruff check` has no Markdown support, but it does handle .pyi and .ipynb.
#
# Every glob is lowercase on purpose, and for .ipynb that is a safety property
# rather than tidiness. Ruff picks its parser from the extension it is handed and
# parses anything it does not recognize as Python source — and JSON is valid
# Python expression syntax. On a case-insensitive filesystem (macOS's default)
# `ruff format Notebook.IPYNB` therefore opens the real notebook, parses its JSON
# as a dict literal and rewrites it as formatted Python, leaving a file that is
# no longer valid JSON. A mis-cased path must miss these globs entirely.
set -uo pipefail

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
# `ruff check` handles .py/.pyi/.ipynb but has no Markdown support.
if [[ "$file" != *.md ]]; then
  uv run ruff check --fix "$file" >/dev/null 2>&1
fi
exit 0

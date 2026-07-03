#!/usr/bin/env bash
# PreToolUse hook: block edits to protected files.
#
#   .env / .env.*            — hold secrets (gitignored), EXCEPT committed
#                              templates (.env.example/.sample/.template/.dist),
#                              which are secret-free and allowed through
#   secrets.toml             — Streamlit's secrets file
#   uv.lock                  — must change only via uv (e.g. `uv add`, `uv lock`)
#
# Matching is case-insensitive because macOS (the primary target) uses a
# case-insensitive filesystem, so UV.LOCK would otherwise open the real file
# while slipping the guard. The hook FAILS CLOSED: if jq is missing it cannot
# parse the payload to tell protected from ordinary paths, so it denies rather
# than silently allowing a protected write through.
#
# Scope note: this is a best-effort guard on the Edit/Write/MultiEdit tools.
# A Bash command that writes these paths (redirect/tee/cp) is NOT intercepted —
# matchers match tool names, and Bash is intentionally unmatched (it is also the
# sanctioned path for uv to rewrite uv.lock).
set -uo pipefail

# Fail closed: without jq the payload cannot be parsed, so deny.
if ! command -v jq >/dev/null 2>&1; then
  reason="Blocked: this project's hook needs 'jq' to check protected paths, but jq is not installed. Install it (e.g. 'brew install jq') or disable the hook."
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}\n' "$reason"
  exit 0
fi

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
[[ -n "$file" ]] || exit 0

base=$(basename "$file")

shopt -s nocasematch

# Committed, secret-free templates are always editable.
case "$base" in
  .env.example | .env.sample | .env.template | .env.dist)
    exit 0
    ;;
esac

case "$base" in
  .env | .env.* | secrets.toml | uv.lock)
    reason="Blocked by project hook: '$base' is protected. .env/secrets.toml hold secrets and uv.lock must change only via uv (e.g. 'uv add'). Edit it manually if you really intend to."
    jq -n --arg r "$reason" \
      '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $r}}'
    exit 0
    ;;
esac
exit 0

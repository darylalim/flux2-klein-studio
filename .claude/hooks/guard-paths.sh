#!/usr/bin/env bash
# PreToolUse hook: block edits to protected files.
#
#   .env / .env.*  — hold secrets (gitignored)
#   uv.lock        — must change only via uv (e.g. `uv add`, `uv lock`)
#
# Emits a structured deny decision (permissionDecision: "deny") on stdout and
# exits 0; the JSON carries the block, so no non-zero exit is needed. Any other
# path is allowed through untouched.
set -uo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
[[ -n "$file" ]] || exit 0

base=$(basename "$file")
case "$base" in
  .env | .env.* | uv.lock)
    reason="Blocked by project hook: '$base' is protected. .env holds secrets and uv.lock must change only via uv (e.g. 'uv add'). Edit it manually if you really intend to."
    jq -n --arg r "$reason" \
      '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $r}}'
    exit 0
    ;;
esac
exit 0

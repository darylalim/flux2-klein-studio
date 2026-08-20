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
# while slipping the guard.
#
# Matching is on BASENAME ONLY, with no ${CLAUDE_PROJECT_DIR} containment check.
# That makes this the one hook that acts outside the project tree, and it is
# deliberate: an out-of-repo `.env` is still a secrets file git cannot restore,
# and the cost of a false deny (one explanatory message) is far below the cost
# of clobbering one. Adding containment here would only ever remove protection.
#
# Missing jq degrades but does not disable the guard. It used to deny *every*
# Edit/Write/MultiEdit in that state — a protection surface of four filenames
# with a failure blast radius of 100% of edits. The path is now extracted with
# bash string operations instead (no external tools at all, so it works on a
# stripped PATH), and only a payload that yields no path at all is denied
# outright, since a protected write cannot be ruled out.
#
# Scope note: this is a best-effort guard on the Edit/Write/MultiEdit tools.
# A Bash command that writes these paths (redirect/tee/cp) is NOT intercepted —
# matchers match tool names, and Bash is intentionally unmatched (it is also the
# sanctioned path for uv to rewrite uv.lock).
set -uo pipefail

_deny() {
  # Build the decision with jq when it is available (it escapes the reason
  # correctly); fall back to printf with a reason carrying no untrusted data.
  if command -v jq >/dev/null 2>&1; then
    jq -n --arg r "$1" \
      '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $r}}'
  else
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}\n' "$2"
  fi
}

input=$(cat)

if command -v jq >/dev/null 2>&1; then
  file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
else
  # Pure-bash extraction of the first "file_path": "..." value.
  file=""
  rest="${input#*\"file_path\"}"
  if [[ "$rest" != "$input" ]]; then
    rest="${rest#*:}"
    rest="${rest#*\"}"
    file="${rest%%\"*}"
  fi
  if [[ -z "$file" ]]; then
    msg="Blocked: this project's hook could not read the file path from the tool payload without 'jq', so a protected file (.env, secrets.toml, uv.lock) cannot be ruled out. Install jq (e.g. 'brew install jq') or disable the hook."
    _deny "$msg" "$msg"
    exit 0
  fi
fi

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
    _deny \
      "Blocked by project hook: '$base' is protected. .env/secrets.toml hold secrets and uv.lock must change only via uv (e.g. 'uv add'). Edit it manually if you really intend to." \
      "Blocked by project hook: that path is protected (.env / secrets.toml hold secrets; uv.lock must change only via uv). Edit it manually if you really intend to."
    exit 0
    ;;
esac
exit 0

#!/usr/bin/env bash
set -euo pipefail

# Dynamically resolve paths relative to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AUTO_DIR="$SCRIPT_DIR"
LOG_DIR="$AUTO_DIR/logs"
LOCK_FILE="$AUTO_DIR/monitor.lock"
ENV_FILE="$ROOT_DIR/.env"
DEFAULT_OPENAI_BASE_URL="https://us.api.openai.com/v1"

mkdir -p "$LOG_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "monitor already running"
  exit 0
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/monitor_${TIMESTAMP}.log"
PROMPT_FILE="$LOG_DIR/monitor_${TIMESTAMP}_prompt.md"
FINAL_FILE="$LOG_DIR/monitor_${TIMESTAMP}_final.md"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

TASK_FILE="${TASK_FILE:-$AUTO_DIR/automation_task.md}"
STATE_FILE="${STATE_FILE:-$AUTO_DIR/automation_state.md}"
OPENAI_BASE_URL_VALUE="${OPENAI_BASE_URL:-${OPENAI_API_BASE:-$DEFAULT_OPENAI_BASE_URL}}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.4}"

export ROOT_DIR AUTO_DIR TASK_FILE STATE_FILE RUN_LOG FINAL_FILE PROMPT_FILE

printf 'Codex monitor base URL: %s\n' "$OPENAI_BASE_URL_VALUE" >>"$RUN_LOG"
printf 'Codex monitor model: %s\n' "$CODEX_MODEL" >>"$RUN_LOG"

cat >"$PROMPT_FILE" <<EOF
You are running a periodic automation pass inside the repository at:
$ROOT_DIR

Task specification:
- Read and follow the instructions in $TASK_FILE.
- Use $STATE_FILE as the running state log.

Required behavior:
- Make small, auditable changes when justified.
- If no code or experiment change is justified, say so briefly in the final message.
- Append a short dated note to $STATE_FILE summarizing findings and next actions.

Output requirements:
- Write a concise final response.
- Keep the work bounded to a single automation pass, then exit.
EOF

if [[ $# -eq 0 ]]; then
  if ! command -v codex >/dev/null 2>&1; then
    echo "codex command not found" >&2
    exit 2
  fi

  CODEX_ARGS=(
    exec
    --full-auto
    --skip-git-repo-check
    --disable responses_websockets
    --disable responses_websockets_v2
    -m "$CODEX_MODEL"
    -C "$ROOT_DIR"
    -o "$FINAL_FILE"
    -
  )
  if [[ -n "$OPENAI_BASE_URL_VALUE" ]]; then
    CODEX_ARGS+=(-c "openai_base_url=\"$OPENAI_BASE_URL_VALUE\"")
  fi

  codex "${CODEX_ARGS[@]}" <"$PROMPT_FILE" >>"$RUN_LOG" 2>&1
  exit $?
fi

if [[ "$1" == "--help" ]]; then
  cat <<EOF
Usage:
  $0
  $0 <command> [args...]

Default behavior:
- acquires a non-blocking lock
- loads environment variables from $ENV_FILE if it exists
- builds a prompt file from the automation task/state context
- runs 'codex exec' in $ROOT_DIR
- writes the full execution log to $RUN_LOG
- writes the agent's last message to $FINAL_FILE

Fallback behavior:
- if a command is provided, runs that command instead and writes its output to $RUN_LOG

One-time setup:
- authenticate codex once, or provide OPENAI_API_KEY in .env

Artifacts:
- Prompt file: $PROMPT_FILE
- Full log: $RUN_LOG
- Final message: $FINAL_FILE
EOF
  exit 2
fi

"$@" >>"$RUN_LOG" 2>&1

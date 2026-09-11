#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ID="${1:-m1_dpo_canonical_v2_final}"
STATE_DIR="$ROOT/runs/queue"
STATE_FILE="$STATE_DIR/$STATE_ID.json"
PID_FILE="$STATE_DIR/$STATE_ID.pid"
LOCK_FILE="$STATE_DIR/$STATE_ID.lock"
LOG_FILE="$STATE_DIR/watchdog.log"

mkdir -p "$STATE_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

if [[ ! -f "$STATE_FILE" ]]; then
  echo "$(date -Is) state missing: $STATE_ID" >>"$LOG_FILE"
  exit 1
fi

.venv/bin/python - "$STATE_FILE" "$PID_FILE" "$LOG_FILE" <<'PY'
import json, os, sys, time
state_file, pid_file, log_file = sys.argv[1:]
payload = json.load(open(state_file, encoding="utf-8"))
state = payload.get("state")
pid = None
if os.path.isfile(pid_file):
    try:
        pid = int(open(pid_file, encoding="utf-8").read().strip())
    except ValueError:
        pid = None
alive = bool(pid and os.path.exists(f"/proc/{pid}"))
command = ""
if alive:
    try:
        command = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\\x00", b" ").decode()
    except OSError:
        alive = False
command_ok = alive and payload.get("config", "") in command
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
if state == "M1_RUNNING" and not command_ok:
    payload["state"] = "FAILED"
    payload["failure"] = "trainer process disappeared or config identity mismatched; no automatic restart was attempted"
    payload["updated_at"] = time.time()
    tmp = state_file + ".tmp"
    open(tmp, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, state_file)
    with open(log_file, "a", encoding="utf-8") as out:
        out.write(f"{now} {payload['state_id']} -> FAILED: process disappeared\n")
    raise SystemExit(1)
with open(log_file, "a", encoding="utf-8") as out:
    out.write(f"{now} {payload['state_id']} state={state} pid={pid} alive={alive}\n")
PY

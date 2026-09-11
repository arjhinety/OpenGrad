#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: run_experiment_queue.sh <config> <state-name> [state-id]}"
STATE_NAME="${2:?usage: run_experiment_queue.sh <config> <state-name> [state-id]}"
STATE_ID="${3:-m1_dpo_canonical_v2_final}"
STATE_DIR="$ROOT/runs/queue"
STATE_FILE="$STATE_DIR/$STATE_ID.json"
PID_FILE="$STATE_DIR/$STATE_ID.pid"
LOG_FILE="$STATE_DIR/$STATE_ID.log"
LOCK_FILE="$STATE_DIR/$STATE_ID.lock"

mkdir -p "$STATE_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "queue lock is already held: $STATE_ID" >&2; exit 2; }

if [[ -f "$STATE_FILE" ]]; then
  existing_state="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$STATE_FILE")"
  case "$existing_state" in
    M1_RUNNING|M1_TRAINED|M1_EVALUATING|M1_DONE|M2_DECISION|M2_RUNNING|M2_EVALUATING|COMPLETE)
      echo "queue refuses duplicate/terminal launch: $STATE_ID is $existing_state" >&2
      exit 3
      ;;
  esac
fi

config_abs="$ROOT/$CONFIG"
if [[ ! -f "$config_abs" ]]; then
  echo "missing config: $config_abs" >&2
  exit 4
fi

.venv/bin/python - "$STATE_FILE" "$PID_FILE" "$CONFIG" "$STATE_NAME" "$STATE_ID" "$$" <<'PY'
import hashlib, json, os, subprocess, sys, time
state_file, pid_file, config, state_name, state_id, shell_pid = sys.argv[1:]
root = os.path.dirname(os.path.dirname(os.path.dirname(state_file)))
config_path = os.path.join(root, config)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
config_hash = hashlib.sha256(open(config_path, "rb").read()).hexdigest()
payload = {
    "schema_version": 1,
    "state": state_name,
    "state_id": state_id,
    "config": config,
    "config_sha256": config_hash,
    "launch_commit": commit,
    "git_dirty_at_launch": dirty,
    "pid": int(shell_pid),
    "updated_at": time.time(),
}
tmp = state_file + ".tmp"
open(tmp, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, state_file)
open(pid_file, "w", encoding="utf-8").write(str(os.getpid()) + "\n")
PY

set +e
.venv/bin/opengrad train "$CONFIG" --json >>"$LOG_FILE" 2>&1
exit_code=$?
set -e

.venv/bin/python - "$STATE_FILE" "$PID_FILE" "$exit_code" <<'PY'
import json, os, sys, time
state_file, pid_file, code = sys.argv[1], sys.argv[2], int(sys.argv[3])
payload = json.load(open(state_file, encoding="utf-8"))
payload["state"] = "M1_TRAINED" if code == 0 else "FAILED"
payload["exit_code"] = code
payload["updated_at"] = time.time()
tmp = state_file + ".tmp"
open(tmp, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, state_file)
try:
    os.unlink(pid_file)
except FileNotFoundError:
    pass
PY
exit "$exit_code"

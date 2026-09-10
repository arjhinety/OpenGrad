#!/usr/bin/env bash
set -euo pipefail

# Register the harness-agnostic OpenGrad MCP server with Command Code.
# Usage: ./integrations/opengrad-mcp/install.sh [project|user]
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCOPE="${1:-project}"
SERVER="$ROOT/integrations/opengrad-mcp/src/index.js"

[[ "$SCOPE" =~ ^(project|user|local)$ ]] || {
  echo "error: scope must be project, user, or local" >&2
  exit 1
}
command -v node >/dev/null 2>&1 || { echo "error: node is not on PATH" >&2; exit 1; }
command -v cmd >/dev/null 2>&1 || { echo "error: cmd is not on PATH" >&2; exit 1; }
[[ -f "$SERVER" ]] || { echo "error: MCP server entrypoint not found: $SERVER" >&2; exit 1; }

cmd mcp remove opengrad --scope "$SCOPE" >/dev/null 2>&1 || true
cmd mcp add --scope "$SCOPE" --env "OPENGRAD_ROOT=$ROOT" opengrad -- node "$SERVER"

cat <<EOF

OpenGrad MCP server registered (scope: $SCOPE).
  Server:        $SERVER
  OpenGrad root: $ROOT

Verify, then restart or start a session:
  cmd mcp list
  /mcp
EOF

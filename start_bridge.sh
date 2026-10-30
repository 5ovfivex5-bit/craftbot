#!/usr/bin/env bash
# Keep the original entry point, forwarding arguments without shell expansion.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/scripts/craftbot.sh" bridge "$@"

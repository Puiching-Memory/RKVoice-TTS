#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOAD_FILE="/sys/kernel/debug/rknpu/load"
PROFILE_LOG="${RKVOICE_RKNN_PROFILE_LOG:-$RUNTIME_DIR/output/rknn_profile.log}"
INTERVAL_SECONDS="${RKVOICE_RKNN_PROFILE_INTERVAL:-0.1}"

mkdir -p "$RUNTIME_DIR/output"

if [ ! -r "$LOAD_FILE" ]; then
    echo "Cannot read $LOAD_FILE"
    exit 1
fi

monitor_load() {
    while true; do
        printf '=== %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')"
        cat "$LOAD_FILE"
        sleep "$INTERVAL_SECONDS"
    done
}

monitor_load > "$PROFILE_LOG" &
monitor_pid="$!"
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

RKVOICE_ASR_PROVIDER=rknn "$RUNTIME_DIR/run_asr.sh" "$@"

kill "$monitor_pid" 2>/dev/null || true
trap - EXIT
echo "RKNN load profile saved to $PROFILE_LOG"
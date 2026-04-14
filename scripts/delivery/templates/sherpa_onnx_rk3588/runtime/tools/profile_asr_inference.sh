#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOAD_FILE="/sys/kernel/debug/rknpu/load"
LEGACY_PROFILE_LOG="${RKVOICE_RKNN_PROFILE_LOG:-$RUNTIME_DIR/output/rknn_profile.log}"
RKNPU_LOAD_LOG="${RKVOICE_RKNPU_LOAD_LOG:-$RUNTIME_DIR/output/rknpu_load.log}"
RKNN_RUNTIME_LOG="${RKVOICE_RKNN_RUNTIME_LOG:-$RUNTIME_DIR/output/rknn_runtime.log}"
INTERVAL_SECONDS="${RKVOICE_RKNN_PROFILE_INTERVAL:-0.1}"

mkdir -p "$RUNTIME_DIR/output"

if [ "$#" -eq 0 ]; then
    audios_dir="$RUNTIME_DIR/audios"
    first_wav="$(find "$audios_dir" -maxdepth 1 -name '*.wav' -type f 2>/dev/null | sort | head -1)"
    if [ -n "$first_wav" ]; then
        set -- "$first_wav"
    fi
fi

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

monitor_load > "$RKNPU_LOAD_LOG" &
monitor_pid="$!"
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

status=0
if RKNN_LOG_LEVEL="${RKNN_LOG_LEVEL:-4}" "$RUNTIME_DIR/run_asr.sh" "$@" 2>&1 | tee "$RKNN_RUNTIME_LOG"; then
    status=0
else
    status=$?
fi

kill "$monitor_pid" 2>/dev/null || true
trap - EXIT
if [ "$RKNPU_LOAD_LOG" != "$LEGACY_PROFILE_LOG" ]; then
    cp "$RKNPU_LOAD_LOG" "$LEGACY_PROFILE_LOG"
fi

echo "RKNN runtime log saved to $RKNN_RUNTIME_LOG"
echo "RKNPU load profile saved to $RKNPU_LOAD_LOG"
echo "Legacy compatibility copy saved to $LEGACY_PROFILE_LOG"
exit "$status"
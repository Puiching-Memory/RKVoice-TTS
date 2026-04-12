#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_BIN="$RUNTIME_DIR/bin/sherpa-onnx-offline"

if ! command -v ldd >/dev/null 2>&1; then
    echo "ldd is not available on the board"
    exit 1
fi

echo "Checking RKNN runtime linkage for $TARGET_BIN"
ldd_output="$(ldd "$TARGET_BIN" || true)"
printf '%s\n' "$ldd_output"

rknn_path="$(printf '%s\n' "$ldd_output" | awk '/librknnrt.so/ {print $3; exit}')"
if [ -z "$rknn_path" ]; then
    echo "librknnrt.so is not resolved for sherpa-onnx-offline"
    exit 1
fi

echo "Resolved librknnrt.so: $rknn_path"
if command -v strings >/dev/null 2>&1; then
    strings "$rknn_path" | grep "librknnrt version" || true
fi

if [ -r /sys/kernel/debug/rknpu/load ]; then
    echo "Current /sys/kernel/debug/rknpu/load snapshot:"
    cat /sys/kernel/debug/rknpu/load
else
    echo "/sys/kernel/debug/rknpu/load is not readable"
fi
#!/bin/bash
set -euo pipefail

echo "== board profile capabilities =="
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "kernel=$(uname -a)"
echo "perf=$(command -v perf || echo missing)"
echo "strace=$(command -v strace || echo missing)"
echo "gdb=$(command -v gdb || echo missing)"
echo "rknn_server=$(command -v rknn_server || echo missing)"
echo "python3=$(command -v python3 || echo missing)"
echo "bash=$(command -v bash || echo missing)"
if command -v ldconfig >/dev/null 2>&1; then
    echo "librknnrt=$(ldconfig -p 2>/dev/null | grep -m1 'librknnrt.so' || echo missing)"
else
    echo "librknnrt=unknown"
fi
if [ -r /sys/kernel/debug/rknpu/load ]; then
    echo "rknpu_load=available"
    cat /sys/kernel/debug/rknpu/load
else
    echo "rknpu_load=unavailable"
fi

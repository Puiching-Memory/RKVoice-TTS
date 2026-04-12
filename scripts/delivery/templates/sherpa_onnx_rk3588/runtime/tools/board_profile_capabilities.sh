#!/bin/bash
set -euo pipefail

echo "== system =="
uname -a || true

echo
echo "== cpu =="
if command -v lscpu >/dev/null 2>&1; then
    lscpu || true
else
    cat /proc/cpuinfo || true
fi

echo
echo "== memory =="
if command -v free >/dev/null 2>&1; then
    free -h || true
fi

echo
echo "== rknn runtime =="
ls -l /lib/librknnrt.so* 2>/dev/null || echo "librknnrt.so not found under /lib"

if [ -r /lib/librknnrt.so ]; then
    if command -v strings >/dev/null 2>&1; then
        strings /lib/librknnrt.so | grep "librknnrt version" || true
    fi
fi

echo
echo "== rknpu load interface =="
if [ -r /sys/kernel/debug/rknpu/load ]; then
    cat /sys/kernel/debug/rknpu/load
else
    echo "/sys/kernel/debug/rknpu/load is not readable"
fi
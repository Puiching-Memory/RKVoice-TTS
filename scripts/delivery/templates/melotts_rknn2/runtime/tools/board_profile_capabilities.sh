#!/bin/bash
set -euo pipefail

echo "== board profile capabilities =="
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "kernel=$(uname -a)"
echo "python3=$(command -v python3 || echo missing)"
echo "rknn_server=$(command -v rknn_server || echo missing)"
echo "perf=$(command -v perf || echo missing)"
echo "strace=$(command -v strace || echo missing)"
echo "gdb=$(command -v gdb || echo missing)"
echo "bash=$(command -v bash || echo missing)"

if command -v python3 >/dev/null 2>&1; then
    if python3 -c "import onnxruntime" >/dev/null 2>&1; then
        echo "onnxruntime_python=available"
    else
        echo "onnxruntime_python=missing"
    fi
    if python3 -c "from rknnlite.api import RKNNLite" >/dev/null 2>&1; then
        echo "rknn_lite2_python=available"
    else
        echo "rknn_lite2_python=missing"
    fi
else
    echo "onnxruntime_python=missing"
    echo "rknn_lite2_python=missing"
fi

if command -v ldconfig >/dev/null 2>&1; then
    echo "librknnrt=$(ldconfig -p 2>/dev/null | grep -m1 'librknnrt.so' || echo missing)"
else
    echo "librknnrt=unknown"
fi

echo "rknn_runtime_layer_log=use profile_tts_inference.sh and board_profile_capabilities.txt together"
echo "recommended_profile_order=python env check -> run_tts -> profile_tts_inference -> rknpu/load"

echo
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
    echo "rknpu_load=available"
    cat /sys/kernel/debug/rknpu/load
else
    echo "rknpu_load=unavailable"
fi
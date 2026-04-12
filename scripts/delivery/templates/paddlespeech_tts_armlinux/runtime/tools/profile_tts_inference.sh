#!/bin/bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
RUNTIME_DIR="$(cd "$TOOLS_DIR/.." && pwd)"

sentence=""
output_wav=""
log_path=""
samples_csv=""
sample_interval_ms="20"

while [ $# -gt 0 ]; do
    case "$1" in
        --sentence)
            sentence="$2"
            shift 2
            ;;
        --output_wav)
            output_wav="$2"
            shift 2
            ;;
        --log)
            log_path="$2"
            shift 2
            ;;
        --samples-csv)
            samples_csv="$2"
            shift 2
            ;;
        --sample-interval-ms)
            sample_interval_ms="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$sentence" ] || [ -z "$output_wav" ] || [ -z "$log_path" ] || [ -z "$samples_csv" ]; then
    echo "Missing required arguments" >&2
    exit 2
fi

mkdir -p "$(dirname "$output_wav")" "$(dirname "$log_path")" "$(dirname "$samples_csv")"

monotonic_ms() {
    awk '{printf "%.0f", $1 * 1000}' /proc/uptime
}

parse_rknpu_load() {
    local load_line
    load_line="$(cat /sys/kernel/debug/rknpu/load 2>/dev/null || true)"
    if [ -z "$load_line" ]; then
        echo "0,0,0"
        return
    fi
    local core0 core1 core2
    core0="$(printf '%s\n' "$load_line" | sed -n 's/.*Core0:[[:space:]]*\([0-9][0-9]*\)%.*/\1/p')"
    core1="$(printf '%s\n' "$load_line" | sed -n 's/.*Core1:[[:space:]]*\([0-9][0-9]*\)%.*/\1/p')"
    core2="$(printf '%s\n' "$load_line" | sed -n 's/.*Core2:[[:space:]]*\([0-9][0-9]*\)%.*/\1/p')"
    echo "${core0:-0},${core1:-0},${core2:-0}"
}

sample_process() {
    local target_pid="$1"
    local started_ms="$2"
    printf 'elapsed_ms,rss_kb,vm_size_kb,threads,state,utime_ticks,stime_ticks,npu_core0_percent,npu_core1_percent,npu_core2_percent\n' > "$samples_csv"
    while kill -0 "$target_pid" 2>/dev/null; do
        local now_ms elapsed_ms rss_kb vm_size_kb threads state utime_ticks stime_ticks npu_values
        now_ms="$(monotonic_ms)"
        elapsed_ms="$((now_ms - started_ms))"
        rss_kb="$(awk '/^VmRSS:/ {print $2}' "/proc/$target_pid/status" 2>/dev/null || true)"
        vm_size_kb="$(awk '/^VmSize:/ {print $2}' "/proc/$target_pid/status" 2>/dev/null || true)"
        threads="$(awk '/^Threads:/ {print $2}' "/proc/$target_pid/status" 2>/dev/null || true)"
        state="$(awk '/^State:/ {print $2}' "/proc/$target_pid/status" 2>/dev/null || true)"
        read -r _ _ _ _ _ _ _ _ _ _ _ _ _ utime_ticks stime_ticks _ < "/proc/$target_pid/stat" 2>/dev/null || true
        npu_values="$(parse_rknpu_load)"
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${elapsed_ms:-0}" \
            "${rss_kb:-0}" \
            "${vm_size_kb:-0}" \
            "${threads:-0}" \
            "${state:-?}" \
            "${utime_ticks:-0}" \
            "${stime_ticks:-0}" \
            "$npu_values" >> "$samples_csv"
        sleep "$(awk -v ms="$sample_interval_ms" 'BEGIN { printf "%.3f", ms / 1000.0 }')"
    done
}

pipe_path="$RUNTIME_DIR/output/.profile_pipe_$$"
rm -f "$pipe_path"
mkfifo "$pipe_path"

started_ms="$(monotonic_ms)"
tee "$log_path" < "$pipe_path" &
tee_pid=$!
"$RUNTIME_DIR/run_tts.sh" --sentence "$sentence" --output_wav "$output_wav" > "$pipe_path" 2>&1 &
tts_pid=$!
sample_process "$tts_pid" "$started_ms" &
sampler_pid=$!

status=0
if ! wait "$tts_pid"; then
    status=$?
fi
wait "$sampler_pid" || true
wait "$tee_pid" || true
rm -f "$pipe_path"
exit "$status"

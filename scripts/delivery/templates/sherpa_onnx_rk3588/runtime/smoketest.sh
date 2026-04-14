#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

summary_log="${RKVOICE_SMOKE_TEST_LOG:-./output/smoke_test_summary.log}"

mkdir -p ./output
exec > >(tee "$summary_log") 2>&1

audios_dir="$SCRIPT_DIR/audios"
run_args=("$@")
if [ "${#run_args[@]}" -eq 0 ] && [ -d "$audios_dir" ]; then
    first_wav="$(find "$audios_dir" -maxdepth 1 -name '*.wav' -type f | sort | head -1)"
    if [ -n "$first_wav" ]; then
        run_args=("$first_wav")
    fi
fi

echo "[0/1] Board capability snapshot"
./tools/board_profile_capabilities.sh > ./output/board_profile_capabilities.txt || true
echo "Capability snapshot saved to ./output/board_profile_capabilities.txt"

echo "[1/1] Streaming ASR (RKNN) smoke test"
./tools/check_rknn_env.sh

if [ "${#run_args[@]}" -gt 0 ]; then
    echo "Using ${#run_args[@]} input wav(s):"
    printf ' - %s\n' "${run_args[@]}"
else
    echo "No explicit wav provided and ./audios is empty; run_asr.sh will resolve its built-in fallback sample."
fi

audio_duration_s=""
if [ "${#run_args[@]}" -gt 0 ] && command -v python3 >/dev/null 2>&1; then
    audio_duration_s="$(python3 - "${run_args[@]}" <<'PY'
import sys
import wave

duration = 0.0
for wav_path in sys.argv[1:]:
    with wave.open(wav_path, "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        duration += frame_count / frame_rate if frame_rate else 0.0
print(f"{duration:.3f}")
PY
    )" || true
fi

start_ts="$(date +%s.%N)"
status=0
set +e
./run_asr.sh "${run_args[@]}"
status=$?
set -e
end_ts="$(date +%s.%N)"

elapsed_s="$(awk -v start="$start_ts" -v end="$end_ts" 'BEGIN { printf "%.3f", (end - start) }')"
echo "Elapsed seconds: $elapsed_s s"

if [ -n "$audio_duration_s" ]; then
    echo "Audio duration: $audio_duration_s s"
    rtf="$(awk -v elapsed="$elapsed_s" -v duration="$audio_duration_s" 'BEGIN { if (duration > 0) printf "%.3f", elapsed / duration; else printf "0.000" }')"
    echo "Real time factor (RTF): $elapsed_s / $audio_duration_s = $rtf"
fi

if [ "$status" -ne 0 ]; then
    echo "ASR smoke test failed with exit code: $status"
    exit "$status"
fi

echo "Smoke test log saved to: $summary_log"
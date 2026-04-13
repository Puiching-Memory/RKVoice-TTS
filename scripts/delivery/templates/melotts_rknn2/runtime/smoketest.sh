#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

tts_text="${1:-你好，欢迎使用 RKVoice MeloTTS-RKNN2 离线语音服务。}"
tts_output_wav="${2:-./output/smoke_test_tts.wav}"
summary_log="${RKVOICE_SMOKE_TEST_LOG:-./output/smoke_test_summary.log}"
profile_output_wav="${RKVOICE_PROFILE_OUTPUT_WAV:-./output/profile_tts.wav}"

mkdir -p ./output
exec > >(tee "$summary_log") 2>&1

echo "[0/4] Board capability snapshot"
./tools/board_profile_capabilities.sh > ./output/board_profile_capabilities.txt || true
echo "Capability snapshot saved to ./output/board_profile_capabilities.txt"

echo "[1/4] Python runtime check"
./tools/check_python_env.sh

echo "[2/4] RKNN TTS smoke test"
RKVOICE_TTS_OUTPUT_WAV="$tts_output_wav" ./run_tts.sh "$tts_text"
ls -lh "$tts_output_wav"

echo "[3/4] RKNN TTS profile capture"
./tools/profile_tts_inference.sh \
    --sentence "$tts_text" \
    --output_wav "$profile_output_wav" \
    --log ./output/rknn_runtime.log \
    --samples-csv ./output/profile-samples.csv
ls -lh ./output/profile-samples.csv ./output/rknn_runtime.log

echo "[4/4] Completed"
echo "Smoke test WAV saved to: $tts_output_wav"
echo "Smoke test log saved to: $summary_log"
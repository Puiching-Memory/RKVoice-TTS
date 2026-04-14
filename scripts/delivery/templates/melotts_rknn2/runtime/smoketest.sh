#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

tts_text="${1:-你好，欢迎使用 RKVoice MeloTTS-RKNN2 离线语音服务。}"
tts_output_wav="${2:-./output/smoke_test_tts.wav}"
summary_log="${RKVOICE_SMOKE_TEST_LOG:-./output/smoke_test_summary.log}"
warm_output_wav="${RKVOICE_WARM_TTS_OUTPUT_WAV:-./output/warm_run_tts.wav}"
profile_output_wav="${RKVOICE_PROFILE_OUTPUT_WAV:-./output/profile_tts.wav}"

mkdir -p ./output
exec > >(tee "$summary_log") 2>&1

echo "[0/5] Board capability snapshot"
./tools/board_profile_capabilities.sh > ./output/board_profile_capabilities.txt || true
echo "Capability snapshot saved to ./output/board_profile_capabilities.txt"

echo "[1/5] Python runtime check"
./tools/check_python_env.sh

echo "[2/5] RKNN TTS cold start"
RKVOICE_TTS_OUTPUT_WAV="$tts_output_wav" ./run_tts.sh "$tts_text"
ls -lh "$tts_output_wav"

echo "[3/5] RKNN TTS warm run"
RKVOICE_TTS_OUTPUT_WAV="$warm_output_wav" ./run_tts.sh "$tts_text"
ls -lh "$warm_output_wav"

echo "[4/5] RKNN TTS profile run"
./tools/profile_tts_inference.sh \
    --sentence "$tts_text" \
    --output_wav "$profile_output_wav" \
    --log ./output/rknn_runtime.log \
    --samples-csv ./output/profile-samples.csv
ls -lh ./output/profile-samples.csv ./output/rknn_runtime.log

echo "[5/5] Completed"
echo "Cold start WAV saved to: $tts_output_wav"
echo "Warm run WAV saved to: $warm_output_wav"
echo "Smoke test log saved to: $summary_log"

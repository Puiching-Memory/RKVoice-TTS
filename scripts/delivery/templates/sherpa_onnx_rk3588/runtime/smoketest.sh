#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

tts_text="${1:-你好，欢迎使用 RKVoice sherpa-onnx 离线语音服务。}"
tts_output_wav="${2:-./output/smoke_test_tts.wav}"
summary_log="${RKVOICE_SMOKE_TEST_LOG:-./output/smoke_test_summary.log}"

mkdir -p ./output
exec > >(tee "$summary_log") 2>&1

echo "[0/4] Board capability snapshot"
./tools/board_profile_capabilities.sh > ./output/board_profile_capabilities.txt || true
echo "Capability snapshot saved to ./output/board_profile_capabilities.txt"

echo "[1/4] CPU TTS smoke test"
RKVOICE_TTS_OUTPUT_WAV="$tts_output_wav" ./run_tts.sh "$tts_text"
ls -lh "$tts_output_wav"

echo "[2/4] Streaming ASR smoke test"
RKVOICE_ASR_MODE=streaming ./run_asr.sh

echo "[3/4] CPU offline ASR smoke test"
RKVOICE_ASR_MODE=offline RKVOICE_ASR_PROVIDER=cpu ./run_asr.sh

if [ "${RKVOICE_ENABLE_RKNN_SMOKETEST:-1}" = "1" ]; then
    echo "[4/4] RKNN ASR smoke test"
    ./tools/check_rknn_env.sh
    ./tools/profile_asr_inference.sh
else
    echo "[4/4] RKNN ASR smoke test skipped"
fi

echo "Smoke test WAV saved to: $tts_output_wav"
echo "Smoke test log saved to: $summary_log"
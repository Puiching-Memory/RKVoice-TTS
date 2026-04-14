#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

summary_log="${RKVOICE_SMOKE_TEST_LOG:-./output/smoke_test_summary.log}"

mkdir -p ./output
exec > >(tee "$summary_log") 2>&1

audios_dir="$SCRIPT_DIR/audios"
wav_files=()
if [ -d "$audios_dir" ]; then
    while IFS= read -r f; do wav_files+=("$f"); done < <(find "$audios_dir" -maxdepth 1 -name '*.wav' -type f | sort)
fi

echo "[0/1] Board capability snapshot"
./tools/board_profile_capabilities.sh > ./output/board_profile_capabilities.txt || true
echo "Capability snapshot saved to ./output/board_profile_capabilities.txt"

echo "[1/1] Streaming ASR (RKNN) smoke test"
./tools/check_rknn_env.sh
./run_asr.sh "${wav_files[@]}"

echo "Smoke test log saved to: $summary_log"
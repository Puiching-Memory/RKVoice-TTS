#!/bin/bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
. ./offline_env.sh

sentence="${1:-你好，欢迎使用离线语音合成服务。}"
output_wav="${2:-./output/smoke_test.wav}"

./build.sh
./run.sh --sentence "$sentence" --output_wav "$output_wav"
echo "Smoke test WAV saved to: $output_wav"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

sentence="${1:-}"
output_wav="${2:-./output/smoke_test.wav}"

if [ -z "$sentence" ]; then
    echo "Missing sentence argument"
    exit 1
fi

./run_tts.sh --sentence "$sentence" --output_wav "$output_wav"
ls -lh "$output_wav"
echo "Smoke test WAV saved to: $output_wav"

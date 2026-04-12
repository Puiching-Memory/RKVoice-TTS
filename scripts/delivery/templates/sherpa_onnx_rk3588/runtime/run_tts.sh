#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="$SCRIPT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$SCRIPT_DIR/output"

tts_model_dir="$SCRIPT_DIR/models/tts/vits-icefall-zh-aishell3"
speaker_id="${RKVOICE_TTS_SPEAKER_ID:-66}"
num_threads="${RKVOICE_TTS_NUM_THREADS:-2}"
length_scale="${RKVOICE_TTS_LENGTH_SCALE:-1.0}"
output_wav="${RKVOICE_TTS_OUTPUT_WAV:-$SCRIPT_DIR/output/demo.wav}"
default_text="${RKVOICE_TTS_TEXT:-你好，欢迎使用 RKVoice sherpa-onnx 离线语音服务。}"

if [ "$#" -eq 0 ]; then
    set -- "$default_text"
fi

exec "$SCRIPT_DIR/bin/sherpa-onnx-offline-tts" \
    --vits-model="$tts_model_dir/model.onnx" \
    --vits-lexicon="$tts_model_dir/lexicon.txt" \
    --vits-tokens="$tts_model_dir/tokens.txt" \
    --tts-rule-fsts="$tts_model_dir/phone.fst,$tts_model_dir/date.fst,$tts_model_dir/number.fst" \
    --sid="$speaker_id" \
    --vits-length-scale="$length_scale" \
    --num-threads="$num_threads" \
    --output-filename="$output_wav" \
    "$1"
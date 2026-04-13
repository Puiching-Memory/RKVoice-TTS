#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/output" "$SCRIPT_DIR/bin"

python_bin="${RKVOICE_PYTHON_BIN:-python3}"
pydeps_dir="${RKVOICE_PYDEPS_DIR:-$SCRIPT_DIR/pydeps}"
output_wav="${RKVOICE_TTS_OUTPUT_WAV:-$SCRIPT_DIR/output/demo.wav}"
default_text="${RKVOICE_TTS_TEXT:-你好，欢迎使用 RKVoice MeloTTS-RKNN2 离线语音服务。}"
sample_rate="${RKVOICE_TTS_SAMPLE_RATE:-44100}"
speed="${RKVOICE_TTS_SPEED:-0.8}"
encoder_model="${RKVOICE_TTS_ENCODER_MODEL:-$SCRIPT_DIR/encoder.onnx}"
decoder_model="${RKVOICE_TTS_DECODER_MODEL:-$SCRIPT_DIR/decoder.rknn}"
lexicon_path="${RKVOICE_TTS_LEXICON:-$SCRIPT_DIR/lexicon.txt}"
token_path="${RKVOICE_TTS_TOKEN:-$SCRIPT_DIR/tokens.txt}"

if [ "$#" -eq 0 ]; then
    set -- "$default_text"
fi

if [ -d "$pydeps_dir" ]; then
    export PYTHONPATH="$pydeps_dir${PYTHONPATH:+:$PYTHONPATH}"
fi

exec "$python_bin" "$SCRIPT_DIR/melotts_rknn.py" \
    --sentence "$1" \
    --wav "$output_wav" \
    --encoder "$encoder_model" \
    --decoder "$decoder_model" \
    --sample_rate "$sample_rate" \
    --speed "$speed" \
    --lexicon "$lexicon_path" \
    --token "$token_path"
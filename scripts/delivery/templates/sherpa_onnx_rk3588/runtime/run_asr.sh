#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="$SCRIPT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

num_threads="${RKVOICE_ASR_NUM_THREADS:-2}"
rknn_num_threads="${RKVOICE_ASR_RKNN_NUM_THREADS:-$num_threads}"

streaming_rknn_model_dir="$SCRIPT_DIR/models/asr/streaming-rknn/streaming-zipformer-rk3588-small"

if [ "$#" -eq 0 ]; then
    audios_dir="$SCRIPT_DIR/audios"
    first_wav="$(find "$audios_dir" -maxdepth 1 -name '*.wav' -type f 2>/dev/null | sort | head -1)"
    if [ -n "$first_wav" ]; then
        set -- "$first_wav"
    else
        set -- "$streaming_rknn_model_dir/test_wavs/DEV_T0000000000.wav"
    fi
fi

case "$rknn_num_threads" in
    1|0|-1|-2|-3|-4)
        ;;
    *)
        rknn_num_threads=1
        ;;
esac

exec "$SCRIPT_DIR/bin/sherpa-onnx" \
    --provider=rknn \
    --tokens="$streaming_rknn_model_dir/tokens.txt" \
    --encoder="$streaming_rknn_model_dir/encoder.rknn" \
    --decoder="$streaming_rknn_model_dir/decoder.rknn" \
    --joiner="$streaming_rknn_model_dir/joiner.rknn" \
    --num-threads="$rknn_num_threads" \
    "$@"
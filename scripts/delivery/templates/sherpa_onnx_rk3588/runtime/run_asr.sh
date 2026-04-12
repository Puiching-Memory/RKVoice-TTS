#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="$SCRIPT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

provider="${RKVOICE_ASR_PROVIDER:-cpu}"
num_threads="${RKVOICE_ASR_NUM_THREADS:-2}"
language="${RKVOICE_ASR_LANGUAGE:-zh}"
use_itn="${RKVOICE_ASR_USE_ITN:-1}"
rknn_num_threads="${RKVOICE_ASR_RKNN_NUM_THREADS:-$num_threads}"

cpu_model_dir="$SCRIPT_DIR/models/asr/cpu/sense-voice"
rknn_model_dir="$SCRIPT_DIR/models/asr/rknn/sense-voice-rk3588-20s"

if [ "$#" -eq 0 ]; then
    if [ "$provider" = "rknn" ]; then
        set -- "$rknn_model_dir/test_wavs/zh.wav"
    else
        set -- "$cpu_model_dir/test_wavs/zh.wav"
    fi
fi

case "$rknn_num_threads" in
    1|0|-1|-2|-3|-4)
        ;;
    *)
        rknn_num_threads=1
        ;;
esac

case "$provider" in
    cpu)
        exec "$SCRIPT_DIR/bin/sherpa-onnx-offline" \
            --sense-voice-model="$cpu_model_dir/model.int8.onnx" \
            --tokens="$cpu_model_dir/tokens.txt" \
            --sense-voice-language="$language" \
            --sense-voice-use-itn="$use_itn" \
            --num-threads="$num_threads" \
            "$@"
        ;;
    rknn)
        exec "$SCRIPT_DIR/bin/sherpa-onnx-offline" \
            --provider=rknn \
            --sense-voice-model="$rknn_model_dir/model.rknn" \
            --tokens="$rknn_model_dir/tokens.txt" \
            --sense-voice-language="$language" \
            --sense-voice-use-itn="$use_itn" \
            --num-threads="$rknn_num_threads" \
            "$@"
        ;;
    *)
        echo "Unsupported RKVOICE_ASR_PROVIDER: $provider"
        exit 1
        ;;
esac
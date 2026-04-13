#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="$SCRIPT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mode="${RKVOICE_ASR_MODE:-streaming}"
provider="${RKVOICE_ASR_PROVIDER:-cpu}"
num_threads="${RKVOICE_ASR_NUM_THREADS:-2}"
language="${RKVOICE_ASR_LANGUAGE:-zh}"
use_itn="${RKVOICE_ASR_USE_ITN:-1}"
rknn_num_threads="${RKVOICE_ASR_RKNN_NUM_THREADS:-$num_threads}"

cpu_model_dir="$SCRIPT_DIR/models/asr/cpu/sense-voice"
rknn_model_dir="$SCRIPT_DIR/models/asr/rknn/sense-voice-rk3588-20s"
streaming_model_dir="$SCRIPT_DIR/models/asr/streaming/streaming-zipformer-multi-zh-hans"

if [ "$#" -eq 0 ]; then
    case "$mode" in
        streaming)
            set -- "$streaming_model_dir/test_wavs/DEV_T0000000000.wav"
            ;;
        *)
            if [ "$provider" = "rknn" ]; then
                set -- "$rknn_model_dir/test_wavs/zh.wav"
            else
                set -- "$cpu_model_dir/test_wavs/zh.wav"
            fi
            ;;
    esac
fi

case "$rknn_num_threads" in
    1|0|-1|-2|-3|-4)
        ;;
    *)
        rknn_num_threads=1
        ;;
esac

case "$mode" in
    streaming)
        exec "$SCRIPT_DIR/bin/sherpa-onnx" \
            --tokens="$streaming_model_dir/tokens.txt" \
            --encoder="$streaming_model_dir/encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx" \
            --decoder="$streaming_model_dir/decoder-epoch-20-avg-1-chunk-16-left-128.onnx" \
            --joiner="$streaming_model_dir/joiner-epoch-20-avg-1-chunk-16-left-128.int8.onnx" \
            --num-threads="$num_threads" \
            "$@"
        ;;
    offline)
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
        ;;
    *)
        echo "Unsupported RKVOICE_ASR_MODE: $mode (use 'streaming' or 'offline')"
        exit 1
        ;;
esac
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="$SCRIPT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$SCRIPT_DIR/output"

backend="${RKVOICE_TTS_BACKEND:-cpu}"
acoustic_model="${RKVOICE_TTS_ACOUSTIC_MODEL:-$SCRIPT_DIR/models/cpu/fastspeech2_csmsc_arm.nb}"
vocoder_model="${RKVOICE_TTS_VOCODER_MODEL:-$SCRIPT_DIR/models/cpu/mb_melgan_csmsc_arm.nb}"
cpu_thread="${RKVOICE_TTS_CPU_THREAD:-1}"

exec "$SCRIPT_DIR/bin/rkvoice_tts_demo" \
    --backend "$backend" \
    --front_conf "$SCRIPT_DIR/front.conf" \
    --acoustic_model "$acoustic_model" \
    --vocoder "$vocoder_model" \
    --cpu_thread "$cpu_thread" \
    "$@"

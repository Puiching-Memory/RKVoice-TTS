#!/bin/bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
RUNTIME_DIR="$(cd "$TOOLS_DIR/.." && pwd)"
OUTPUT_DIR="$RUNTIME_DIR/output"
WHEEL_DIR="$RUNTIME_DIR/wheels"
PYDEPS_DIR="${RKVOICE_PYDEPS_DIR:-$RUNTIME_DIR/pydeps}"
PYTHON_BIN="${RKVOICE_PYTHON_BIN:-python3}"
LOG_PATH="${RKVOICE_PYTHON_DEPS_LOG:-$OUTPUT_DIR/python_deps_install.log}"

mkdir -p "$OUTPUT_DIR"
exec > >(tee "$LOG_PATH") 2>&1

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "$PYTHON_BIN is required but was not found in PATH" >&2
    exit 1
fi

if [ ! -d "$WHEEL_DIR" ]; then
    echo "Wheelhouse is missing: $WHEEL_DIR" >&2
    exit 1
fi

pip_wheel="$(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'pip-*.whl' | sort | tail -n 1)"
if [ -z "$pip_wheel" ]; then
    echo "pip wheel is missing under $WHEEL_DIR" >&2
    exit 1
fi

echo "python_bin=$PYTHON_BIN"
echo "pydeps_dir=$PYDEPS_DIR"
echo "wheel_dir=$WHEEL_DIR"
echo "pip_wheel=$pip_wheel"

mkdir -p "$PYDEPS_DIR"
rm -rf "$PYDEPS_DIR"/*

PYTHONPATH="$pip_wheel" "$PYTHON_BIN" -m pip install \
    --no-index \
    --find-links="$WHEEL_DIR" \
    --target "$PYDEPS_DIR" \
    --upgrade \
    numpy==1.24.4 \
    onnxruntime==1.16.0 \
    soundfile \
    cn2an \
    inflect \
    psutil \
    ruamel.yaml

PYTHONPATH="$pip_wheel" "$PYTHON_BIN" -m pip install \
    --no-index \
    --find-links="$WHEEL_DIR" \
    --target "$PYDEPS_DIR" \
    --upgrade \
    --no-deps \
    rknn-toolkit-lite2==2.3.2

PYTHONPATH="$PYDEPS_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import cn2an
import inflect
import numpy
import onnxruntime
import soundfile
from rknnlite.api import RKNNLite

print("IMPORT_OK")
PY

printf '%s\n' "$PYDEPS_DIR" > "$OUTPUT_DIR/python_deps_path.txt"
echo "Dependency installation finished"
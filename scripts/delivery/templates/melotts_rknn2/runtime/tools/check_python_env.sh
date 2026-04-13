#!/bin/bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
RUNTIME_DIR="$(cd "$TOOLS_DIR/.." && pwd)"
python_bin="${RKVOICE_PYTHON_BIN:-python3}"
pydeps_dir="${RKVOICE_PYDEPS_DIR:-$RUNTIME_DIR/pydeps}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "$python_bin is required but was not found in PATH" >&2
    exit 1
fi

if [ -d "$pydeps_dir" ]; then
    export PYTHONPATH="$pydeps_dir${PYTHONPATH:+:$PYTHONPATH}"
fi

"$python_bin" - <<'PY'
import importlib
import sys

print(f"python_version={sys.version.split()[0]}")

required_modules = ["numpy", "soundfile", "onnxruntime"]
missing = []
for module_name in required_modules:
    try:
        importlib.import_module(module_name)
    except Exception:
        missing.append(module_name)

try:
    from rknnlite.api import RKNNLite  # noqa: F401
except Exception:
    missing.append("rknnlite.api")

if missing:
    print("missing_python_modules=" + ",".join(missing), file=sys.stderr)
    raise SystemExit(1)

print("python_imports=ok")
PY
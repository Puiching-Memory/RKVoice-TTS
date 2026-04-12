#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$SCRIPT_DIR/tools/cmake/bin:$PATH"

THIRD_PARTY_LIB_DIR="$SCRIPT_DIR/src/TTSCppFrontend/third-party/build/lib"
THIRD_PARTY_LIB64_DIR="$SCRIPT_DIR/src/TTSCppFrontend/third-party/build/lib64"
PADDLE_LITE_LIB_DIR="$SCRIPT_DIR/libs/inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv/cxx/lib"

export LD_LIBRARY_PATH="$PADDLE_LITE_LIB_DIR:$THIRD_PARTY_LIB_DIR:$THIRD_PARTY_LIB64_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

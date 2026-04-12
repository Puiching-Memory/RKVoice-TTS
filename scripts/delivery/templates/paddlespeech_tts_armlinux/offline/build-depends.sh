#!/bin/bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
./src/TTSCppFrontend/build-depends.sh "$@"

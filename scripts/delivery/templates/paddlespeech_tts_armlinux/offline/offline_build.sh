#!/bin/bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
. ./offline_env.sh
./build.sh "$@"

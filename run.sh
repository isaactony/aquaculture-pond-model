#!/usr/bin/env bash
# Run a project script with the project's virtual environment, e.g. ./run.sh src/train_v1.py
# On macOS without Homebrew's libomp, LightGBM is pointed at the OpenMP runtime that ships with scikit-learn.
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname)" == "Darwin" && ! -e /opt/homebrew/opt/libomp/lib/libomp.dylib && ! -e /usr/local/opt/libomp/lib/libomp.dylib ]]; then
  export DYLD_LIBRARY_PATH="$(echo "$PWD"/.venv/lib/python*/site-packages/sklearn/.dylibs)${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
exec .venv/bin/python "$@"

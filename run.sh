#!/usr/bin/env bash
# Greenium pipeline — local end-to-end runner.
# Sources the determinism env, then runs the Python pipeline (greenium.main)
# followed by the R post-processing (R/main.R).
set -euo pipefail

cd "$(dirname "$0")"

# Determinism / portability env (vendored in this repo at scripts/env.sh).
source scripts/env.sh

echo "=== Running greenium.main (Python) ==="
uv run python -u -m greenium.main

echo "=== Running R/main.R (R post-processing) ==="
Rscript R/main.R

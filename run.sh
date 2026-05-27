#!/usr/bin/env bash
# Greenium pipeline — local end-to-end runner.
# Sources the determinism env from the planning repo, then runs the Python
# pipeline (greenium.main) followed by the R post-processing (R/main.R).
set -euo pipefail

ENV_SH="${ENV_SH:-$HOME/projects/planning/greenium-codebase-separation/scripts/env.sh}"
if [[ -f "$ENV_SH" ]]; then
    source "$ENV_SH"
else
    echo "warn: $ENV_SH not found; pipeline will run without pinned determinism env" >&2
fi

cd "$(dirname "$0")"

echo "=== Running greenium.main (Python) ==="
uv run python -u -m greenium.main

echo "=== Running R/main.R (R post-processing) ==="
Rscript R/main.R

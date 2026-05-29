#!/usr/bin/env bash
# scripts/env.sh — determinism / portability env vars for the pipeline.
#
# Sourced by run.sh before the pipeline runs. Source it manually if you invoke
# the Python or R steps directly.
#
# Usage:  source scripts/env.sh

# Deterministic timestamps in any embedded metadata (matplotlib PDF, etc.).
export SOURCE_DATE_EPOCH=0

# Deterministic Python hash randomization → reproducible dict iteration order.
export PYTHONHASHSEED=0

# matplotlib must use a headless backend.
export MPLBACKEND=Agg

# Localize the matplotlib font cache to the project, so two clean clones
# produce identical font-subset metadata.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PWD}/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

# matplotlib rcParams overrides (deterministic PDF output), shipped alongside
# this script.
export MATPLOTLIBRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/matplotlibrc"

# Deterministic locale for any locale-sensitive string sort.
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "[env.sh] SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH"
echo "[env.sh] PYTHONHASHSEED=$PYTHONHASHSEED"
echo "[env.sh] MPLBACKEND=$MPLBACKEND"
echo "[env.sh] MPLCONFIGDIR=$MPLCONFIGDIR"
echo "[env.sh] MATPLOTLIBRC=$MATPLOTLIBRC"
echo "[env.sh] LC_ALL=$LC_ALL"

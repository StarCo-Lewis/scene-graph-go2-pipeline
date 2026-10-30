#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
CG_PYTHON=/home/lewis/miniforge3/envs/conceptgraph/bin/python
if [[ ! -x "$CG_PYTHON" ]]; then
  printf '%s\n' "ConceptGraphs interpreter missing: $CG_PYTHON" >&2
  exit 2
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  if [[ ! -t 0 ]]; then
    printf '%s\n' 'Set DASHSCOPE_API_KEY in this shell before running.' >&2
    exit 2
  fi
  read -r -s -p 'DashScope API key: ' DASHSCOPE_API_KEY
  printf '\n'
  export DASHSCOPE_API_KEY
fi
PREPARED="$PROJECT_ROOT/outputs/d3_619_prepared_20261021/prepared"
if [[ -f "${PREPARED%/prepared}/input_audit.json" ]]; then
  exec "$CG_PYTHON" -B -m embodied_pipeline real-scene run --prepared-input "$PREPARED" "$@"
else
  exec "$CG_PYTHON" -B -m embodied_pipeline real-scene run "$@"
fi

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
    printf '%s\n' 'Set DASHSCOPE_API_KEY in this shell before continuing.' >&2
    exit 2
  fi
  read -r -s -p 'DashScope API key: ' DASHSCOPE_API_KEY
  printf '\n'
  export DASHSCOPE_API_KEY
fi
exec "$CG_PYTHON" -B -m embodied_pipeline real-scene resume \
  --from-summary "$PROJECT_ROOT/outputs/real_scene_runs/d3_619_20261021_191531_691432/run_summary.json" "$@"

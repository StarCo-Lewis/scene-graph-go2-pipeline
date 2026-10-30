#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
CG_PYTHON=/home/lewis/miniforge3/envs/conceptgraph/bin/python
preflight=false
for arg in "$@"; do
  [[ "$arg" != --preflight-only ]] || preflight=true
done
if [[ "$preflight" == false && -z "${DASHSCOPE_API_KEY:-}" ]]; then
  if [[ ! -t 0 ]]; then
    printf '%s\n' 'Set DASHSCOPE_API_KEY before continuing.' >&2
    exit 2
  fi
  read -r -s -p 'DashScope API key: ' DASHSCOPE_API_KEY
  printf '\n'
  export DASHSCOPE_API_KEY
fi
exec "$CG_PYTHON" -B -u -m embodied_pipeline real-scene-plan "$@"

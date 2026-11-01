#!/usr/bin/env bash
# Run manually in WSL. Never start a robot, SSH session or physical dispatcher.
set +x
set -euo pipefail
pipeline_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$pipeline_root"
pilot_python="${PIPELINE_PYTHON:-/home/lewis/miniforge3/envs/conceptgraph/bin/python}"
pilot_root="${1:-outputs/symbolic_pilot_12_20261031_ready}"
"$pilot_python" -B -m embodied_pipeline.data_collection.symbolic_pilot check --root "$pilot_root"
if [[ -d "$pilot_root/results" ]]; then
  echo 'This pilot already started. Review its results; automatic replay is disabled.'
  exit 1
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  read -r -s -p 'DashScope API key (hidden): ' DASHSCOPE_API_KEY
  printf '\n'
  export DASHSCOPE_API_KEY
fi
echo 'Qwen 3.8 Flash; 12 tasks; at most 36 requests; no automatic retry.'
echo 'Budget: CNY 10. This is NOT a provider billing cap; configure provider limits and monitor billing.'
pilot_log="$pilot_root/console_$(date +%Y%m%d_%H%M%S)_$$.log"
PYTHONUNBUFFERED=1 "$pilot_python" -B -m embodied_pipeline.data_collection.symbolic_pilot run \
  --root "$pilot_root" --acknowledge-paid-api 2>&1 | tee "$pilot_log"

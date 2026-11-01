#!/usr/bin/env bash
set +x
set -euo pipefail
pipeline_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$pipeline_root"
pilot_python="${PIPELINE_PYTHON:-/home/lewis/miniforge3/envs/conceptgraph/bin/python}"
pilot_root="${1:-outputs/symbolic_extension18_20261031_ready_v2}"
"$pilot_python" -B tools/extend_symbolic_pilot.py check --root "$pilot_root"
if [[ -d "$pilot_root/results" ]]; then
  echo 'This extension already started. Automatic replay is disabled.'
  exit 1
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  read -r -s -p 'DashScope API key (hidden): ' DASHSCOPE_API_KEY
  printf '\n'
  export DASHSCOPE_API_KEY
fi
echo '18 NEW tasks / at most 54 requests / Qwen 3.8 Flash / no automatic retry / no robot execution.'
echo 'Budget allowance CNY 10 is NOT a hard billing cap. Record provider billing before/after this batch.'
pilot_log="$pilot_root/console_$(date +%Y%m%d_%H%M%S)_$$.log"
PYTHONUNBUFFERED=1 "$pilot_python" -B tools/extend_symbolic_pilot.py run \
  --root "$pilot_root" --acknowledge-paid-api 2>&1 | tee "$pilot_log"

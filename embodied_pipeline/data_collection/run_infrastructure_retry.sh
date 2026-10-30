#!/usr/bin/env bash
set -euo pipefail
scene_id="${1:-}"
case "$scene_id" in scene200) ceiling=384;; scene202) ceiling=336;; d3_619) ceiling=270;; *) printf 'Usage: bash run_infrastructure_retry.sh scene200|scene202|d3_619 --approve-new-api-calls\n'; exit 2;; esac
shift
approved=false
for arg in "$@"; do if [[ "$arg" == '--approve-new-api-calls' ]]; then approved=true; fi; done
if [[ "$approved" != true ]]; then printf 'Explicit new API authorization is required.\n'; exit 2; fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
retry_root='/mnt/c/Users/1/Documents/Codex/natural_retry_330_20261024_v1'
pipeline_python="${PIPELINE_PYTHON:-/home/lewis/miniforge3/envs/conceptgraph/bin/python}"
if [[ ! -f "$retry_root/collection_config.json" ]]; then printf 'Prepared retry bundle missing.\n'; exit 2; fi
if [[ -f "$retry_root/account_blocked.json" ]]; then printf 'Account stop marker is present. Resolve account status before resuming; do not rerun repeatedly.\n'; exit 2; fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then read -r -s -p 'DashScope API key (hidden): ' DASHSCOPE_API_KEY; printf '\n'; export DASHSCOPE_API_KEY; fi
export PYTHONUNBUFFERED=1
log_path="$retry_root/cases/$scene_id/console_$(date -u +%Y%m%dT%H%M%S%NZ)_$$.log"
printf 'Retry scene: %s | request ceiling: %s | log: %s\n' "$scene_id" "$ceiling" "$log_path"
"$pipeline_python" -B -m embodied_pipeline.data_collection.natural_batch worker --output "$retry_root" --scene "$scene_id" --max-api-calls "$ceiling" "$@" 2>&1 | tee "$log_path"

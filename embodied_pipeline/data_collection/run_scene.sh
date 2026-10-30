#!/usr/bin/env bash
# One scene per terminal; no robot connection, automatic retries or key files.
set -euo pipefail

scene_id="${1:-}"
case "$scene_id" in
  scene200|scene201|scene202|d3_619) shift ;;
  *) printf 'Usage: bash embodied_pipeline/data_collection/run_scene.sh scene200|scene201|scene202|d3_619 --approve-new-api-calls [--limit-per-scene 2]\n'; exit 2 ;;
esac
approved=false
for arg in "$@"; do
  if [[ "$arg" == '--approve-new-api-calls' ]]; then approved=true; fi
done
if [[ "$approved" != true ]]; then
  printf 'New API calls require --approve-new-api-calls. Maximum 600 requests per full scene; this is not a money budget.\n'
  exit 2
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
collection_root="${COLLECTION_OUTPUT_DIR:-/mnt/c/Users/1/Documents/Codex/task_collection_800_20261023_v1}"
pipeline_python="${PIPELINE_PYTHON:-/home/lewis/miniforge3/envs/conceptgraph/bin/python}"
if [[ ! -f "$collection_root/collection_config.json" ]]; then
  printf 'Prepared collection not found: %s\n' "$collection_root"
  exit 2
fi
if [[ ! -x "$pipeline_python" ]]; then
  printf 'Python not found: %s\n' "$pipeline_python"
  exit 2
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  read -r -s -p 'DashScope API key (hidden): ' DASHSCOPE_API_KEY
  printf '\n'
  export DASHSCOPE_API_KEY
fi
export PYTHONUNBUFFERED=1
log_path="$collection_root/cases/$scene_id/console_$(date -u +%Y%m%dT%H%M%S%NZ)_$$.log"
printf 'Scene: %s | Log: %s\n' "$scene_id" "$log_path"
printf 'English tasks; one pass; decomposition + object binding + skill binding; no robot execution.\n'
"$pipeline_python" -B -m embodied_pipeline.data_collection.batch worker \
  --output "$collection_root" --scene "$scene_id" --max-api-calls 600 "$@" 2>&1 | tee "$log_path"

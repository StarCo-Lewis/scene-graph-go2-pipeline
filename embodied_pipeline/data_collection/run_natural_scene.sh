#!/usr/bin/env bash
# One reviewed scene per terminal. No ROS/robot connection.
set -euo pipefail
scene_id="${1:-}"
case "$scene_id" in scene200|scene202|d3_619) shift;; *) printf 'Usage: bash run_natural_scene.sh scene200|scene202|d3_619 --approve-new-api-calls\n'; exit 2;; esac
approved=false
for arg in "$@"; do if [[ "$arg" == '--approve-new-api-calls' ]]; then approved=true; fi; done
if [[ "$approved" != true ]]; then printf 'Pass --approve-new-api-calls for this NEW collection. Up to 600 model requests per scene; not a monetary budget.\n'; exit 2; fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
collection_root="${NATURAL_COLLECTION_OUTPUT_DIR:-/mnt/c/Users/1/Documents/Codex/natural_collection_600_20261024_v1}"
pipeline_python="${PIPELINE_PYTHON:-/home/lewis/miniforge3/envs/conceptgraph/bin/python}"
if [[ ! -f "$collection_root/collection_config.json" ]]; then printf 'Prepared collection missing: %s\n' "$collection_root"; exit 2; fi
if [[ ! -x "$pipeline_python" ]]; then printf 'Python not found: %s\n' "$pipeline_python"; exit 2; fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then read -r -s -p 'DashScope API key (hidden): ' DASHSCOPE_API_KEY; printf '\n'; export DASHSCOPE_API_KEY; fi
export PYTHONUNBUFFERED=1
log_path="$collection_root/cases/$scene_id/console_$(date -u +%Y%m%dT%H%M%S%NZ)_$$.log"
printf 'Scene: %s | Log: %s\n' "$scene_id" "$log_path"
printf 'Reviewed English instructions + user location + dialogue. Single pass, no automatic retries, no robot execution.\n'
"$pipeline_python" -B -m embodied_pipeline.data_collection.natural_batch worker --output "$collection_root" --scene "$scene_id" --max-api-calls 600 "$@" 2>&1 | tee "$log_path"

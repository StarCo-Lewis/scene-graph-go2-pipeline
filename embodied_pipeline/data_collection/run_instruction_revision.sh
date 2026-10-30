#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != '--approve-new-api-calls' ]]; then
  printf 'Usage: bash run_instruction_revision.sh --approve-new-api-calls\n'
  exit 2
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
revision_root="$project_root/outputs/instruction_audit_74_20261024_v1/corrected_run"
pipeline_python="${PIPELINE_PYTHON:-/home/lewis/miniforge3/envs/conceptgraph/bin/python}"
if [[ ! -f "$revision_root/collection_config.json" ]]; then printf 'Prepared revision bundle missing.\n'; exit 2; fi
if [[ -f "$revision_root/account_blocked.json" ]]; then printf 'Account stop marker present; inspect account before resuming.\n'; exit 2; fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then read -r -s -p 'DashScope API key (hidden): ' DASHSCOPE_API_KEY; printf '\n'; export DASHSCOPE_API_KEY; fi
export PYTHONUNBUFFERED=1
failed=0
for scene_id in scene200 scene202 d3_619; do
  case "$scene_id" in scene200) ceiling=39;; scene202) ceiling=30;; d3_619) ceiling=33;; esac
  if [[ -f "$revision_root/account_blocked.json" ]]; then printf 'Arrearage: remaining scenes not started.\n'; exit 2; fi
  log_path="$revision_root/cases/$scene_id/console_$(date -u +%Y%m%dT%H%M%S%NZ)_$$.log"
  printf 'Input revision scene: %s | request ceiling: %s | log: %s\n' "$scene_id" "$ceiling" "$log_path"
  if "$pipeline_python" -B -m embodied_pipeline.data_collection.natural_batch worker --output "$revision_root" --scene "$scene_id" --max-api-calls "$ceiling" --approve-new-api-calls 2>&1 | tee "$log_path"; then
    printf 'Scene finished: %s\n' "$scene_id"
  else
    failed=1
    printf 'Scene worker reported an error; evidence retained: %s\n' "$scene_id"
  fi
done
"$pipeline_python" -B -m embodied_pipeline.data_collection.natural_batch summarize --output "$revision_root"
exit "$failed"

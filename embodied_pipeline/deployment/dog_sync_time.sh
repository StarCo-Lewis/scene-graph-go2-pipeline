#!/usr/bin/env bash
# Adapted from the user-provided dog_sync_time.sh (2026-10-30).
# Run on the laptop in Git Bash, from the pipeline root:
#   bash embodied_pipeline/deployment/dog_sync_time.sh
# Interactive SSH/sudo authentication; no external askpass script required.
# Updates the system clock only; does not start controllers or send goals.

set -euo pipefail

DOG=unitree@192.168.123.18
NOW=$(date -u +"%Y-%m-%d %H:%M:%S")
printf '笔记本 UTC：%s\n' "$NOW"
printf '连接狗并同步系统时间；请按提示输入 SSH / sudo 密码。\n'

env LD_LIBRARY_PATH= ssh -tt -o ConnectTimeout=10 "$DOG" "
  set -e
  printf '狗同步前 UTC：'
  date -u '+%F %T'
  sudo date -u -s '$NOW'
  printf '狗同步后 UTC：'
  date -u '+%F %T'
"

printf '\n系统时间同步命令已完成。按手册，狗每次重新开机后需再次同步。\n'

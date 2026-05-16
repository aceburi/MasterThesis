#!/usr/bin/env bash
# Aborta el comando si la RAM residente (RSS) del proceso + hijos supera LIMIT_GB.
# Uso: ./run_with_mem_limit.sh <limite_GB> <comando...>
# Ejemplo: ./run_with_mem_limit.sh 400 python -u modelQ_ultralight.py
set -euo pipefail

LIMIT_GB="${1:?Uso: $0 <limite_GB> <comando...>}"
shift
LIMIT_KB=$((LIMIT_GB * 1024 * 1024))
POLL_SEC="${MEM_LIMIT_POLL_SEC:-15}"

rss_tree_kb() {
  local root="$1" total=0 rss pid
  for pid in $(pgrep -P "$root" 2>/dev/null; echo "$root"); do
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    [[ -n "$rss" ]] && total=$((total + rss))
  done
  echo "$total"
}

"$@" &
MAIN_PID=$!
echo "[memlimit] PID=$MAIN_PID limite=${LIMIT_GB} GB poll=${POLL_SEC}s"

while kill -0 "$MAIN_PID" 2>/dev/null; do
  RSS=$(rss_tree_kb "$MAIN_PID")
  RSS_GB=$(awk -v r="$RSS" 'BEGIN{printf "%.1f", r/1024/1024}')
  if (( RSS > LIMIT_KB )); then
    echo "[memlimit] ABORT: RSS ~${RSS_GB} GB > ${LIMIT_GB} GB"
    pkill -TERM -P "$MAIN_PID" 2>/dev/null || true
    kill -TERM "$MAIN_PID" 2>/dev/null || true
    sleep 5
    pkill -KILL -P "$MAIN_PID" 2>/dev/null || true
    kill -KILL "$MAIN_PID" 2>/dev/null || true
    exit 137
  fi
  sleep "$POLL_SEC"
done

wait "$MAIN_PID"

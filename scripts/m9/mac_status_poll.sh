#!/bin/bash
# Mac-side read-only status poll. Never restarts training.
# First success: write a full snapshot. Then every 30 minutes.
set -u
LOG="${HOME}/.cursor/m9_status_poll.log"
STATUS="${HOME}/.cursor/m9_last_status.txt"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

ssh_ok() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 ancoda-server 'echo up' >/dev/null 2>&1
}

snapshot() {
  ssh -o BatchMode=yes -o ConnectTimeout=40 ancoda-server 'bash -s' <<'REMOTE'
set -u
echo "===UPTIME==="
uptime
echo "===SCREEN==="
screen -ls 2>/dev/null || echo none
echo "===QUEUE==="
tail -5 "$HOME/m9/queue.log" 2>/dev/null || echo no-queue
echo "===CRON==="
crontab -l 2>/dev/null || echo no-crontab
echo "===DISK==="
df -h "$HOME"
echo "===MEM==="
free -h
echo "===XPU==="
"$HOME/m9env/bin/python" - <<'PY'
import torch
print("xpu", torch.xpu.is_available())
if torch.xpu.is_available():
    print("alloc_bytes", int(torch.xpu.memory_allocated()))
    print("reserved_bytes", int(torch.xpu.memory_reserved()))
    try:
        print("max_alloc_bytes", int(torch.xpu.max_memory_allocated()))
    except Exception as e:
        print("max_alloc_err", str(e))
PY
echo "===TRAIN_PS==="
ps -u "$(id -un)" -o pid=,etime=,pcpu=,rss=,cmd= | grep -E "ancoda_tok.m9 train" | grep -v grep || echo none
echo "===RUN_LOGS==="
for d in "$HOME"/m9/runs/*/; do
  [ -d "$d" ] || continue
  base=$(basename "$d")
  case "$base" in
    probe-*|compile-*|pilot-*) continue ;;
  esac
  if [ -f "$d/log.jsonl" ]; then
    echo "RUN_DIR $base"
    echo -n "LAST "
    tail -1 "$d/log.jsonl" 2>/dev/null || true
    ls -1 "$d"/latest.pt "$d"/checkpoint*.pt 2>/dev/null | sed "s|^|CKPT |" || true
  fi
done
echo "===BSTAR==="
python3 - <<'PY'
from pathlib import Path
p = Path.home() / "m9" / "bstar.json"
print(p.read_text() if p.is_file() else "missing")
PY
echo "===DRIVER_HEAD==="
if [ -f "$HOME/m9/run_full.sh" ]; then
  ls -l "$HOME/m9/run_full.sh"
  grep -nE "resume|latest|screen|crontab|@reboot|checkpoint" "$HOME/m9/run_full.sh" | head -40
else
  echo no-run_full
fi
REMOTE
}

echo "==== waiter start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
# Until first success: retry every 5 minutes (failed polls are normal).
while ! ssh_ok; do
  echo "ssh down $(date -u +%H:%M:%S)"
  sleep 300
done
echo "ssh up first $(date -u +%Y-%m-%dT%H:%M:%SZ)"
snapshot | tee "$STATUS"
echo "FIRST_SNAPSHOT_OK" | tee -a "$STATUS"

# Steady cadence.
while true; do
  sleep 1800
  if ssh_ok; then
    echo "==== poll $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
    snapshot | tee "$STATUS"
  else
    echo "ssh down $(date -u +%H:%M:%S) (normal)"
  fi
done

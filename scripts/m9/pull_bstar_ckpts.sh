#!/usr/bin/env bash
# B* checkpoints only. Run with: bash pull_bstar_ckpts.sh
set -u
LOG=/Users/jarvis/Downloads/ancoda-tokenizer/artifacts/m9/server_copy/ckpt_rsync.log
CKPT=/Users/jarvis/Downloads/ancoda-tokenizer/artifacts/m9/checkpoints
# Array so bash word-splits correctly; do not use a single-string RSH under zsh.
SSH=(ssh -o BatchMode=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=30
     -o ConnectTimeout=30 -o TCPKeepAlive=yes)
RSYNC_RSH='ssh -o BatchMode=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=30 -o ConnectTimeout=30 -o TCPKeepAlive=yes'

echo "bstar_only $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

pull() {
  local src="$1" dest="$2" label="$3"
  local a=1
  while [ "$a" -le 30 ]; do
    echo "TRY $label a=$a $(date -u +%H:%M:%S)" | tee -a "$LOG"
    if rsync -az --partial --timeout=600 -e "$RSYNC_RSH" "$src" "$dest"; then
      echo "OK $label $(date -u +%H:%M:%S)" | tee -a "$LOG"
      return 0
    fi
    echo "retry $label a=$a $(date -u +%H:%M:%S)" | tee -a "$LOG"
    sleep 45
    a=$((a + 1))
  done
  echo "FAIL $label" | tee -a "$LOG"
  return 1
}

for run in ancoda-64k-s1 ancoda-64k-s2 ancoda-64k-s3 \
           yak-64k-v1-s1 yak-64k-v1-s2 yak-64k-v1-s3 \
           sidskarki-64k-s1 sidskarki-64k-s2 sidskarki-64k-s3 \
           yak-64k-v2.0-s1; do
  mkdir -p "$CKPT/$run"
  if ls "$CKPT/$run"/checkpoint-bstar-*.pt >/dev/null 2>&1; then
    echo "SKIP_BSTAR $run" | tee -a "$LOG"
    continue
  fi
  remote=$("${SSH[@]}" ancoda-server "ls /home/ancoda/m9/runs/$run/checkpoint-bstar-*.pt" | head -1) || remote=""
  if [ -z "$remote" ]; then
    echo "FAIL_LIST BSTAR:$run" | tee -a "$LOG"
    continue
  fi
  echo "BSTAR $run $(basename "$remote") $(date -u +%H:%M:%S)" | tee -a "$LOG"
  pull "ancoda-server:$remote" "$CKPT/$run/" "BSTAR:$run" || true
done

echo BSTAR_CKPTS_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) | tee -a "$LOG"
python3 - <<'PY'
from pathlib import Path
root = Path("/Users/jarvis/Downloads/ancoda-tokenizer/artifacts/m9/checkpoints")
n = 0
for p in sorted(root.rglob("checkpoint-bstar-*.pt")):
    if p.name.startswith("."):
        continue
    n += 1
    print(f"{p.stat().st_size/1e6:7.1f} MB  {p.relative_to(root)}")
print(f"bstar_files={n}")
PY

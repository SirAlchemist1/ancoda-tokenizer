#!/bin/bash
# One-at-a-time M9 full-run driver. Resumes from latest.pt. Never tokenizes while training.
set -uo pipefail
source "$HOME/m9env/bin/activate"
export PYTHONPATH="$HOME/m9/code/python"
PY="$HOME/m9env/bin/python"
BUILD="$HOME/m9/code/BUILD.json"
EVAL="$HOME/m9/data/eval"
SID_TOK="$HOME/m9/tokenizers/sidskarki-64k/nepali_bpe_64k.model"
BSTAR_JSON="$HOME/m9/bstar.json"
QUEUE_LOG="$HOME/m9/queue.log"
LOCK="$HOME/m9/driver.lock"
CRASH_DIR="$HOME/m9/crashes"
mkdir -p "$CRASH_DIR" "$HOME/m9/evals" "$HOME/m9/runs"

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "STOP: another driver holds $LOCK" | tee -a "$QUEUE_LOG"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

free_gb() {
  df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9'
}

check_disk() {
  avail=$(free_gb)
  if [ "$avail" -lt 10 ]; then
    echo "STOP: free disk ${avail}G under 10 GB" | tee -a "$QUEUE_LOG"
    exit 1
  fi
}

tokenizer_for() {
  case "$1" in
    ancoda-64k) echo "$HOME/m9/tokenizers/ancoda-64k/tokenizer.json" ;;
    yak-64k-v1) echo "$HOME/m9/tokenizers/yak-64k-v1/tokenizer.json" ;;
    yak-64k-v2.0) echo "$HOME/m9/tokenizers/yak-64k-v2.0/tokenizer.json" ;;
    plain-64k) echo "$HOME/m9/tokenizers/plain-64k/tokenizer.json" ;;
    ancoda-64k-nojoiner) echo "$HOME/m9/tokenizers/ancoda-64k-nojoiner/tokenizer.json" ;;
    sidskarki-64k) echo "$SID_TOK" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

vocab_for() {
  case "$1" in
    sidskarki-64k) echo 64001 ;;
    plain-64k) echo 65537 ;;
    *) echo 65536 ;;
  esac
}

bstar_for() {
  local arm="$1" seed="$2"
  "$PY" - << PY
import json
payload = json.load(open("$BSTAR_JSON"))
print(int(payload[str($seed)]["bstar"]))
PY
}

run_bpb() {
  local arm="$1" seed="$2" ckpt="$3" tag="$4"
  local out="$HOME/m9/evals/${arm}-s${seed}-${tag}.json"
  if [ -f "$out" ]; then
    echo "eval exists $out"
    return 0
  fi
  check_disk
  "$PY" -u -m ancoda_tok.m9 bpb \
    --arm "$arm" \
    --tokenizer "$(tokenizer_for "$arm")" \
    --checkpoint "$ckpt" \
    --eval-dir "$EVAL" \
    --out "$out" \
    --build "$BUILD" \
    --sid-tokenizer "$SID_TOK"
}

run_one() {
  local arm="$1" seed="$2"
  local run_dir="$HOME/m9/runs/${arm}-s${seed}"
  local tokens="$HOME/m9/tokens/${arm}"
  local vocab
  vocab=$(vocab_for "$arm")
  local bstar
  bstar=$(bstar_for "$arm" "$seed")
  local crashes="$CRASH_DIR/${arm}-s${seed}.count"
  mkdir -p "$run_dir"
  check_disk
  echo "START $arm seed=$seed bstar=$bstar $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$QUEUE_LOG"
  set +e
  "$PY" -u -m ancoda_tok.m9 train \
    --tokens "$tokens" \
    --arm "$arm" \
    --seed "$seed" \
    --vocab "$vocab" \
    --steps 7630 \
    --run-dir "$run_dir" \
    --micro-batch 4 \
    --checkpoint-every 500 \
    --bstar "$bstar" \
    --resume \
    --build "$BUILD" \
    > "$run_dir/train.log" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    local n=0
    [ -f "$crashes" ] && n=$(cat "$crashes")
    n=$((n + 1))
    echo "$n" > "$crashes"
    echo "CRASH $arm seed=$seed count=$n rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$QUEUE_LOG"
    if [ "$n" -ge 3 ]; then
      echo "STOP: $arm seed=$seed crashed 3 times" | tee -a "$QUEUE_LOG"
      exit 1
    fi
    return 1
  fi
  echo 0 > "$crashes"
  local final="$run_dir/checkpoint-007630.pt"
  local bstar_ckpt
  bstar_ckpt=$(ls "$run_dir"/checkpoint-bstar-*.pt 2>/dev/null | head -1 || true)
  if [ ! -f "$final" ]; then
    echo "STOP: missing final checkpoint for $arm seed=$seed" | tee -a "$QUEUE_LOG"
    exit 1
  fi
  run_bpb "$arm" "$seed" "$final" "final"
  if [ -n "$bstar_ckpt" ] && [ -f "$bstar_ckpt" ]; then
    run_bpb "$arm" "$seed" "$bstar_ckpt" "bstar"
  else
    echo "WARN: no B* checkpoint for $arm seed=$seed" | tee -a "$QUEUE_LOG"
  fi
  echo "DONE $arm seed=$seed $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$QUEUE_LOG"
  return 0
}

JOBS=(
  "ancoda-64k 1"
  "yak-64k-v1 1"
  "sidskarki-64k 1"
  "ancoda-64k 2"
  "yak-64k-v1 2"
  "sidskarki-64k 2"
  "ancoda-64k 3"
  "yak-64k-v1 3"
  "sidskarki-64k 3"
  "yak-64k-v2.0 1"
  "plain-64k 1"
  "yak-64k-v2.0 2"
  "plain-64k 2"
  "yak-64k-v2.0 3"
  "plain-64k 3"
  "ancoda-64k-nojoiner 1"
  "ancoda-64k-nojoiner 2"
  "ancoda-64k-nojoiner 3"
)

echo "DRIVER_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$QUEUE_LOG"
for job in "${JOBS[@]}"; do
  set -- $job
  arm=$1
  seed=$2
  done_mark="$HOME/m9/runs/${arm}-s${seed}/DONE"
  if [ -f "$done_mark" ]; then
    echo "SKIP $arm seed=$seed already DONE" | tee -a "$QUEUE_LOG"
    continue
  fi
  until run_one "$arm" "$seed"; do
    echo "RETRY $arm seed=$seed after crash $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$QUEUE_LOG"
    sleep 5
  done
  date -u +%Y-%m-%dT%H:%M:%SZ > "$HOME/m9/runs/${arm}-s${seed}/DONE"
done
echo "DRIVER_ALL_OK $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$QUEUE_LOG"

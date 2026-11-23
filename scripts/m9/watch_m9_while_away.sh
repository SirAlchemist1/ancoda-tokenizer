#!/bin/bash
# Mac-side M9 watcher: poll every 30 minutes. Never restart/kill/edit the server.
# Safe across sleep — simply resumes on wake. Set M9_HOST and M9_ROOT.
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
: "${M9_HOST:?set M9_HOST to the M9 SSH host alias}"
: "${M9_ROOT:?set M9_ROOT to the remote account home directory on the M9 server}"
DEST="$REPO_ROOT/artifacts/m9/server_copy"
PROG="$REPO_ROOT/docs/PROGRESS.md"
INTERVAL=1800
mkdir -p "$DEST/runs" "$DEST/results" "$DEST/evals"
cd "$REPO_ROOT"

rsync_once() {
  local ok=0
  local i
  for i in 1 2 3; do
    if rsync -az --timeout=60 \
      "$M9_HOST:$M9_ROOT/m9/queue.log" \
      "$M9_HOST:$M9_ROOT/m9/bstar.json" \
      "$DEST/" 2>/dev/null
    then
      ok=1
      break
    fi
    sleep 20
  done
  rsync -az --timeout=60 --ignore-missing-args \
    "$M9_HOST:$M9_ROOT/m9/results/" "$DEST/results/" 2>/dev/null || true
  rsync -az --timeout=60 --ignore-missing-args \
    "$M9_HOST:$M9_ROOT/m9/evals/" "$DEST/evals/" 2>/dev/null || true
  rsync -az --timeout=120 \
    --include='*/' --include='train.log' --include='log.jsonl' --include='DONE' --exclude='*' \
    "$M9_HOST:$M9_ROOT/m9/runs/" "$DEST/runs/" 2>/dev/null || true
  return $((1 - ok))
}

commit_as_author() {
  local msg="$1"
  shift
  local author_name author_email
  author_name=$(git config user.name)
  author_email=$(git config user.email)
  git add -- "$@"
  if git diff --cached --quiet; then
    return 0
  fi
  # commit-tree so hooks cannot inject Co-authored-by.
  local tree parent new
  tree=$(git write-tree)
  parent=$(git rev-parse HEAD)
  new=$(GIT_AUTHOR_NAME="$author_name" \
    GIT_AUTHOR_EMAIL="$author_email" \
    GIT_COMMITTER_NAME="$author_name" \
    GIT_COMMITTER_EMAIL="$author_email" \
    git commit-tree "$tree" -p "$parent" -m "$msg")
  git update-ref HEAD "$new"
  git reset --mixed HEAD >/dev/null
}

append_done_progress() {
  local arm="$1" seed="$2" done_ts="$3"
  local mark="M9 DONE ${arm}-s${seed}"
  if grep -qF "$mark" "$PROG" 2>/dev/null; then
    return 0
  fi
  {
    echo ""
    echo "- [x] ${mark} at ${done_ts} UTC (watcher). Queue continues on $M9_HOST."
  } >> "$PROG"
  # Mirror small DONE marker + logs into docs snapshot when under 5 MB
  local snap="$REPO_ROOT/docs/m9_server_copy/runs/${arm}-s${seed}"
  mkdir -p "$snap"
  local src="$DEST/runs/${arm}-s${seed}"
  cp "$src/DONE" "$snap/DONE" 2>/dev/null || true
  for f in train.log log.jsonl; do
    if [ -f "$src/$f" ]; then
      sz=$(stat -f%z "$src/$f" 2>/dev/null || echo 99999999)
      if [ "$sz" -lt 5242880 ]; then
        cp "$src/$f" "$snap/$f"
      fi
    fi
  done
  rsync -az "$DEST/queue.log" "$DEST/bstar.json" "$REPO_ROOT/docs/m9_server_copy/" 2>/dev/null || true
  commit_as_author \
    "Record ${arm} seed ${seed} finished on the M9 server queue." \
    "$PROG" \
    "docs/m9_server_copy/queue.log" \
    "docs/m9_server_copy/bstar.json" \
    "docs/m9_server_copy/runs/${arm}-s${seed}"
}

echo "M9 watcher start $(date -u +%Y-%m-%dT%H:%M:%SZ) root=$REPO_ROOT"
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%S)
  if rsync_once; then
    echo "[$ts] rsync ok"
    # Detect newly finished runs
    for done in "$DEST"/runs/*/DONE; do
      [ -f "$done" ] || continue
      run_dir=$(dirname "$done")
      base=$(basename "$run_dir")
      # ancoda-64k-s1 -> arm=ancoda-64k seed=1
      arm=${base%-s*}
      seed=${base##*-s}
      done_ts=$(tr -d '\n' < "$done")
      append_done_progress "$arm" "$seed" "$done_ts" || echo "[$ts] commit failed for $base"
    done
  else
    echo "[$ts] rsync failed (will retry next cycle; server untouched)"
  fi
  sleep "$INTERVAL"
done
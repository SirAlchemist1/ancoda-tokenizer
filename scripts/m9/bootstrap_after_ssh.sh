#!/bin/bash
# Run on the Mac once SSH works. Syncs code, finishes pilot, then starts full runs.
# Writes a Mac-side report to data/m9/report_step100.txt when A-s1 passes step 100.
set -euo pipefail
ROOT="/Users/jarvis/Downloads/ancoda-tokenizer"
cd "$ROOT"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 ancoda-server)

echo "=== STATE ==="
"${SSH[@]}" 'echo up; uptime; echo ---; screen -ls || true; echo ---; free -h; echo ---; df -h ~; echo ---; echo PILOT_A_TAIL; tail -20 ~/m9/pilot-a.log 2>/dev/null || echo no-pilot-a; echo ---; ls -la ~/m9/runs/ancoda-64k-s1/ 2>/dev/null || echo no-a-s1-run'

echo "=== SYNC CODE ==="
# Refresh BUILD from current HEAD (prereg amendment already at 49a0437).
uv run python -m ancoda_tok.m9 build-record --out data/m9/BUILD.json
rsync -a --exclude __pycache__ \
  "$ROOT/python/ancoda_tok/__init__.py" \
  "$ROOT/python/ancoda_tok/m9" \
  ancoda-server:~/m9/code/python/ancoda_tok/
rsync -a "$ROOT/data/m9/BUILD.json" ancoda-server:~/m9/code/BUILD.json
rsync -a "$ROOT/data/m9/run_pilot.sh" "$ROOT/data/m9/run_full.sh" "$ROOT/data/m9/install_reboot_driver.sh" \
  ancoda-server:~/m9/
"${SSH[@]}" 'chmod +x ~/m9/run_pilot.sh ~/m9/run_full.sh ~/m9/install_reboot_driver.sh; python3 -c "import json; print(json.load(open(\"/home/ancoda/m9/code/BUILD.json\"))[\"git_commit\"])"'

echo "=== START PILOT ==="
"${SSH[@]}" 'screen -S m9pilot -X quit 2>/dev/null || true; screen -S m9full -X quit 2>/dev/null || true; rm -f ~/m9/driver.lock 2>/dev/null || true; rmdir ~/m9/driver.lock 2>/dev/null || true; : > ~/m9/pilot-finish.log; screen -dmS m9pilot bash -lc "bash ~/m9/run_pilot.sh"; sleep 3; screen -ls; tail -30 ~/m9/pilot-finish.log'

echo "=== WAIT PILOT ==="
# Pilot A ~301 steps at ~5s/step ≈ 25–40 min if from 0; less if resumed. Cap 3h.
for i in $(seq 1 360); do
  status=$("${SSH[@]}" 'if grep -q PILOT_FINISH_OK ~/m9/pilot-finish.log 2>/dev/null; then echo OK; elif grep -q "^STOP:" ~/m9/pilot-finish.log 2>/dev/null; then echo STOP; elif screen -ls 2>/dev/null | grep -q m9pilot; then echo RUN; else echo DEAD; fi' || echo SSHFAIL)
  echo "pilot_poll=$i status=$status $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ "$status" = "OK" ]; then
    break
  fi
  if [ "$status" = "STOP" ] || [ "$status" = "DEAD" ]; then
    "${SSH[@]}" 'tail -80 ~/m9/pilot-finish.log; ls -la ~/m9/runs/ancoda-64k-s1/ 2>/dev/null; ls -la ~/m9/runs/sidskarki-64k-s1-pilot/ 2>/dev/null'
    echo "STOP: pilot did not finish cleanly ($status)"
    exit 1
  fi
  sleep 30
done
status=$("${SSH[@]}" 'grep -q PILOT_FINISH_OK ~/m9/pilot-finish.log && echo OK || echo FAIL')
if [ "$status" != "OK" ]; then
  echo "STOP: pilot wait timed out"
  exit 1
fi

echo "=== VALIDATE PILOT NUMBERS ==="
"${SSH[@]}" 'PYTHONPATH=$HOME/m9/code/python ~/m9env/bin/python - << "PY"
import json, math, sys
from pathlib import Path
home = Path.home() / "m9"
# Arm A losses from log.jsonl
log_a = home / "runs/ancoda-64k-s1/log.jsonl"
rows = [json.loads(l) for l in log_a.read_text().splitlines() if l.strip()]
by_step = {r["step"]: r for r in rows}
need = [0, 50, 100, 150, 200, 250, 300]
missing = [s for s in need if s not in by_step]
if missing:
    # tolerate missing mid points if 0 and 300 exist
    if 0 not in by_step or 300 not in by_step:
        raise SystemExit(f"STOP: arm A missing steps {missing}")
step0 = by_step[0]["loss"]
step300 = by_step[300]["loss"]
print(f"2a A step0_loss={step0:.4f} step300_loss={step300:.4f} seconds={by_step[300].get(\"step_seconds\")}")
if not (step300 < 11.183 and step300 < step0 - 0.5):
    raise SystemExit(f"STOP: arm A loss did not fall clearly below 11.183 (got {step300})")
# resume
resume = (home / "runs/ancoda-64k-s1/RESUME_PROVEN").read_text().strip()
print("2b", resume)
# Arm C
log_c = home / "runs/sidskarki-64k-s1-pilot/log.jsonl"
crows = [json.loads(l) for l in log_c.read_text().splitlines() if l.strip()]
c0 = crows[0]["loss"]
c_last = crows[-1]["loss"]
print(f"2c C first_loss={c0:.4f} last_step={crows[-1][\"step\"]} last_loss={c_last:.4f}")
if not (c_last < c0):
    raise SystemExit("STOP: arm C loss did not fall")
# BPB
bpb_path = home / "evals/pilot-ancoda-64k-s1-step301.json"
bpb = json.loads(bpb_path.read_text())
groups = bpb.get("groups") or {}
print("2d bpb_groups", sorted(groups.keys())[:30], "n=", len(groups))
if len(groups) < 1:
    raise SystemExit("STOP: BPB eval empty")
print("PILOT_NUMBERS_OK")
# dump compact summary for Mac report
summary = {
    "a_step0": step0,
    "a_step300": step300,
    "a_step_seconds": by_step[300].get("step_seconds"),
    "a_tokens_per_s": 131072 / by_step[300]["step_seconds"] if by_step[300].get("step_seconds") else None,
    "a_peak_cpu": by_step[300].get("peak_cpu_temp_c"),
    "resume": resume,
    "c_first": c0,
    "c_last": c_last,
    "c_last_step": crows[-1]["step"],
    "bpb_path": str(bpb_path),
    "bpb_ngroups": len(groups),
    "bpb_group_names": sorted(groups.keys()),
}
(home / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
PY'

echo "=== FULL RUN PREP ==="
"${SSH[@]}" 'bash -s' << 'REMOTE'
set -euo pipefail
source "$HOME/m9env/bin/activate"
export PYTHONPATH="$HOME/m9/code/python"
# Delete pilot dirs so full A-s1 starts from 0
rm -rf "$HOME/m9/runs/ancoda-64k-s1" \
       "$HOME/m9/runs/sidskarki-64k-s1-pilot" \
       "$HOME/m9/runs/resume-probe" \
       "$HOME/m9/runs/compile-a-s1" \
       "$HOME/m9/runs"/probe-* 2>/dev/null || true
# Recompute B*
~/m9env/bin/python - << 'PY'
import json
from pathlib import Path
import numpy as np
from ancoda_tok.m9.perm import first_step_reaching
from ancoda_tok.m9.common import FULL_STEPS, TOKENS_PER_STEP
from ancoda_tok.m9.reader import TokenStream

POS = FULL_STEPS * TOKENS_PER_STEP  # 1_000_079_360
core = ["ancoda-64k", "yak-64k-v1", "sidskarki-64k"]
out = {}
for seed in (1, 2, 3):
    bytes_at = {}
    for arm in core:
        stream = TokenStream(Path.home() / "m9" / "tokens" / arm, seed)
        b = int(stream.bytes_covered(POS))
        bytes_at[arm] = b
        print(f"seed={seed} arm={arm} bytes_at_full={b}")
    bstar = min(bytes_at.values())
    steps = {}
    for arm in core + ["yak-64k-v2.0"]:
        stream = TokenStream(Path.home() / "m9" / "tokens" / arm, seed)
        step = first_step_reaching(stream.bytes_covered, bstar, TOKENS_PER_STEP, FULL_STEPS)
        steps[arm] = step
        print(f"seed={seed} arm={arm} bstar_step={step}")
    out[str(seed)] = {"bstar": bstar, "bytes_at_full": bytes_at, "bstar_step": steps}
path = Path.home() / "m9" / "bstar.json"
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print("wrote", path)
print(json.dumps(out, indent=2, sort_keys=True))
PY
bash "$HOME/m9/install_reboot_driver.sh"
: > "$HOME/m9/queue.log"
rmdir "$HOME/m9/driver.lock" 2>/dev/null || true
screen -S m9full -X quit 2>/dev/null || true
screen -dmS m9full bash -lc "bash $HOME/m9/run_full.sh >> $HOME/m9/queue.log 2>&1"
sleep 3
screen -ls
echo QUEUE_HEAD
head -20 "$HOME/m9/queue.log" || true
cat "$HOME/m9/bstar.json"
REMOTE

echo "=== WAIT A-s1 PAST STEP 100 ==="
for i in $(seq 1 240); do
  info=$("${SSH[@]}" 'python3 - << "PY"
import json
from pathlib import Path
p = Path.home()/"m9/runs/ancoda-64k-s1/log.jsonl"
if not p.exists():
    print("NOLOG")
else:
    rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not rows:
        print("EMPTY")
    else:
        last=rows[-1]
        print(f"OK step={last[\"step\"]} loss={last[\"loss\"]:.4f} sec={last.get(\"step_seconds\")} tok_s={131072/last[\"step_seconds\"] if last.get(\"step_seconds\") else None}")
        if last["step"] >= 100:
            print("PASSED100")
PY
tail -3 ~/m9/queue.log 2>/dev/null || true
df -BG --output=avail ~ | tail -1
' || echo SSHFAIL)
  echo "as1_poll=$i $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$info"
  if echo "$info" | grep -q PASSED100; then
    break
  fi
  if echo "$info" | grep -q 'STOP:'; then
    echo "STOP: full-run queue halted"
    exit 1
  fi
  sleep 30
done
if ! echo "$info" | grep -q PASSED100; then
  echo "STOP: A-s1 did not reach step 100 in time"
  exit 1
fi

echo "=== COLLECT REPORT ==="
"${SSH[@]}" 'python3 - << "PY"
import json, time
from pathlib import Path
home = Path.home()/"m9"
pilot = json.loads((home/"pilot_summary.json").read_text())
bstar = json.loads((home/"bstar.json").read_text())
rows=[json.loads(l) for l in (home/"runs/ancoda-64k-s1/log.jsonl").read_text().splitlines() if l.strip()]
by={r["step"]:r for r in rows}
r100=by.get(100) or by[max(k for k in by if k<=100)]
sec=r100.get("step_seconds") or 5.0
tok_s=131072/sec
# 10 runs × 7630 steps; core 9 + E
core_steps=9*7630
e_steps=7630
core_hours=(core_steps*sec)/3600
e_hours=(e_steps*sec)/3600
import subprocess, shutil
df=subprocess.check_output(["df","-h",str(Path.home())], text=True)
avail=subprocess.check_output(["bash","-lc","df -BG --output=avail $HOME | tail -1"], text=True).strip()
queue=Path.home().joinpath("m9/queue.log").read_text()[-500:]
build=json.loads((home/"code/BUILD.json").read_text())
report={
  "pilot": pilot,
  "bstar": bstar,
  "a_s1_step100": r100,
  "tokens_per_s": tok_s,
  "projected_core_hours": core_hours,
  "projected_e_hours": e_hours,
  "free_disk_line": df.strip().splitlines()[-1],
  "free_gb": avail,
  "build_git": build.get("git_commit"),
  "queue_tail": queue,
  "mac_progress_cmd": "ssh -o BatchMode=yes ancoda-server '\''tail -20 ~/m9/queue.log; python3 -c \"import json;from pathlib import Path;p=Path.home()/\\\"m9/runs/ancoda-64k-s1/log.jsonl\\\";rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()];print(rows[-1] if rows else None)\"'\''",
}
(home/"report_step100.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
PY' | tee "$ROOT/data/m9/report_step100.json"

echo "BOOTSTRAP_ALL_OK $(date -u +%Y-%m-%dT%H:%M:%SZ)"

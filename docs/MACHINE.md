# Machine check (M0 step 2)

Recorded: 4 Nov 2026 (local). Repo path: `/Users/jarvis/Downloads/ancoda-tokenizer`.
Re-checked after PLAN.md disk threshold update (100 GB free minimum).

## `sw_vers`

```
ProductName:		macOS
ProductVersion:		26.6
BuildVersion:		25G72
```

## `uname -m`

```
arm64
```

## `sysctl -n hw.memsize` (GB)

- Raw bytes: `19327352832`
- GiB (÷ 1024³): **18.0 GB**
- Threshold: ≥ 16 GB → **PASS**

## Performance cores

- Command: `sysctl -n hw.perflevel0.physicalcpu` (Apple Silicon)
- Value: **5**
- Use as rayon / datatrove worker count (`ANCODA_WORKERS=5`).

## `df -h .`

```
Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s5   460Gi   280Gi   156Gi    65%    2.9M  1.6G    0%   /System/Volumes/Data
```

- Available: **156 GiB**
- Threshold: ≥ 100 GB free → **PASS**
- Note: M2 measures real storage need before downloads; STOP later only if under 50 GB would remain after a >20 GB write.

## Verdict

**PASS.** Continue M0 from step 3.

# Machine check (setup step 2)

Recorded: 4 Nov 2026 (local). Repo path: `$REPO_ROOT`.
Re-checked after docs/SPEC.md disk threshold update (100 GB free minimum).

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
- Note: data-plan measures real storage need before downloads; STOP later only if under 50 GB would remain after a >20 GB write.

## Verdict

**PASS.** Continue setup from step 3.

## trainer benchmark (8 Nov 2026)

1 GB slice: `artifacts/counts/n50_1gb/pretoken_counts.bin` (1,950,607 unique pre-tokens, 20,046,751 key bytes, 1,000,000,996 text bytes). Config `configs/bench_64k_1gb.yaml` (block0, block1, bans, vocab 65536). Release binary, `RAYON_NUM_THREADS=5`.

```
peak_rss_bytes=1303707648
unique_pretokens=1950607
initial_symbols=17248686
pair_entries=65796
learned=60847
duplicates=0
banned=113
stopped=filled
tokens=64512
elapsed_secs=22
```

Peak RSS 1,303,707,648 bytes = 1.214 GiB. Wall clock of the `cargo run --release` command, including the release compile, was 33.5 s; the trainer itself reported 22 s.

Full n50 table: 9,041,738 unique, 227,794,543 key bytes. Projection uses the larger ratio, key bytes: 227794543 / 20046751 = 11.363165. Projected full-size peak = 1,303,707,648 × 11.363165 = 14,814,245,355 bytes = 13.797 GiB. Limit is 14 × 1024³ = 15,032,385,536 bytes. Projection is under the limit by 218,140,181 bytes.

Full n50 train (`configs/build_64k.yaml`, release, 5 threads) then measured:

```
peak_rss_bytes=4078977024
learned=60847
duplicates=0
banned=93
stopped=filled
tokens=64512
elapsed_secs=68
```

Actual peak 4,078,977,024 bytes = 3.800 GiB, under 14 GiB. Plain 1 GB train peaked at 1,448,984,576 bytes (1.350 GiB) in 22 s and filled 65,536 tokens.

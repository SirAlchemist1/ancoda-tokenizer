# Arm F, seed 3

Analysis only. The server queue was not changed. Eval JSON and run
manifests were pulled with rsync. The decision rule is the 16 Nov 2026
amendment in `docs/MODEL_EVAL_PREREG.md`. Difference is BPB(F) − BPB(A). Lower BPB
is better. **F better** requires every seed difference strictly negative and
|mean| > sample SD (divisor n−1). **F worse** requires every difference
strictly positive and the same mean test. Otherwise **no measurable
difference**. This is not the one-sided arm-A rule.

Equal tokens is step 7630. Equal bytes is each seed's existing B* from
`bstar.json` (the core A/B/C byte target). B* was not recomputed. The F
checkpoint step is the first step whose covered bytes reach that target,
read from the first line of that run's `train.log` and checked against the
eval JSON.

## Log check

| Seed | Registered B* bytes | F step | A step | F final eval | F manifest |
|---:|---:|---:|---:|---|---|
| 1 | 4813230411 | 5395 | 5397 | 2026-11-18T00:48:39Z | 2026-11-18T00:41:14Z |
| 2 | 4795818967 | 5400 | 5402 | 2026-11-18T11:04:31Z | 2026-11-18T10:56:59Z |
| 3 | 4805817305 | 5399 | 5401 | 2026-11-18T21:20:03Z | 2026-11-18T21:12:36Z |

Each final eval is `checkpoint-007630.pt` with `completed_steps` 7630.
Each B* eval's checkpoint name and `completed_steps` match that run's
`first_step`. Both arms' train logs carry the registered B* byte count.
Manifest `tokens_seen` is 1,000,079,360 and `completed_steps` is 7630.
Each final eval's `finished_at` is after that run's manifest `finished_at`.

## Main comparison

Pre-registered prediction: F is better than A on both joiner sets. If it
is not, the joiner-copy explanation is weakened.

### equal tokens

| Set | d1 | d2 | d3 | mean | sample SD | verdict |
|---|---:|---:|---:|---:|---:|---|
| `sealed_npi_zwj` | -0.103889 | -0.098407 | -0.089378 | -0.097225 | 0.007328 | **F better** |
| `sealed_npi_zwnj` | 0.015424 | -0.007443 | -0.022655 | -0.004891 | 0.019167 | **no measurable difference** |

| Seed | F ZWJ | A ZWJ | F ZWNJ | A ZWNJ |
|---:|---:|---:|---:|---:|
| 1 | 0.941454 | 1.045344 | 1.130445 | 1.115021 |
| 2 | 0.947438 | 1.045845 | 1.120598 | 1.128041 |
| 3 | 0.969378 | 1.058755 | 1.118145 | 1.140800 |

### equal bytes

| Set | d1 | d2 | d3 | mean | sample SD | verdict |
|---|---:|---:|---:|---:|---:|---|
| `sealed_npi_zwj` | -0.108212 | -0.095610 | -0.097938 | -0.100586 | 0.006706 | **F better** |
| `sealed_npi_zwnj` | 0.010396 | 0.001907 | -0.026866 | -0.004854 | 0.019529 | **no measurable difference** |

| Seed | F ZWJ | A ZWJ | F ZWNJ | A ZWNJ |
|---:|---:|---:|---:|---:|
| 1 | 0.955801 | 1.064013 | 1.144666 | 1.134270 |
| 2 | 0.967380 | 1.062990 | 1.138833 | 1.136925 |
| 3 | 0.979197 | 1.077135 | 1.138394 | 1.165260 |

The prediction is not met, so the joiner-copy explanation is weakened.

## Secondary comparison

F minus A on sealed Nepali and the unweighted mean of the 11 sealed
Devanagari groups, at both checkpoints. The same decision rule is applied.
No prediction is attached.

### equal tokens

| Set | d1 | d2 | d3 | mean | sample SD | verdict |
|---|---:|---:|---:|---:|---:|---|
| Nepali | -0.000674 | 0.001486 | 0.000743 | 0.000519 | 0.001098 | **no measurable difference** |
| Dev11 | -0.001830 | 0.002710 | 0.000908 | 0.000596 | 0.002286 | **no measurable difference** |

### equal bytes

| Set | d1 | d2 | d3 | mean | sample SD | verdict |
|---|---:|---:|---:|---:|---:|---|
| Nepali | -0.000556 | 0.001362 | 0.001142 | 0.000649 | 0.001050 | **no measurable difference** |
| Dev11 | -0.002467 | 0.002902 | 0.001579 | 0.000671 | 0.002798 | **no measurable difference** |

## Training hours

Elapsed time from `started_at` to `finished_at` in each run manifest.
This is wall time for that training process, including whatever the
manifest recorded between those two timestamps.

| Arm | Seed | Started | Finished | Hours |
|---|---:|---|---|---:|
| A ancoda-64k | 1 | 2026-11-10T18:01:08Z | 2026-11-11T04:00:53Z | 10.00 |
| A ancoda-64k | 2 | 2026-11-12T00:49:58Z | 2026-11-12T10:51:31Z | 10.03 |
| A ancoda-64k | 3 | 2026-11-13T07:40:13Z | 2026-11-13T17:41:14Z | 10.02 |
| B yak-64k-v1 | 1 | 2026-11-11T04:15:46Z | 2026-11-11T14:17:10Z | 10.02 |
| B yak-64k-v1 | 2 | 2026-11-12T11:06:15Z | 2026-11-12T21:07:59Z | 10.03 |
| B yak-64k-v1 | 3 | 2026-11-13T17:56:02Z | 2026-11-14T03:57:53Z | 10.03 |
| C sidskarki-64k | 1 | 2026-11-11T14:32:29Z | 2026-11-12T00:33:46Z | 10.02 |
| C sidskarki-64k | 2 | 2026-11-12T21:23:13Z | 2026-11-13T07:23:58Z | 10.01 |
| C sidskarki-64k | 3 | 2026-11-14T04:13:18Z | 2026-11-14T14:13:26Z | 10.00 |
| D plain-64k | 1 | 2026-11-15T10:42:57Z | 2026-11-15T20:55:22Z | 10.21 |
| D plain-64k | 2 | 2026-11-16T07:27:22Z | 2026-11-16T17:39:46Z | 10.21 |
| D plain-64k | 3 | 2026-11-17T04:11:14Z | 2026-11-17T14:24:29Z | 10.22 |
| E yak-64k-v2.0 | 1 | 2026-11-14T14:29:44Z | 2026-11-15T00:31:07Z | 10.02 |
| E yak-64k-v2.0 | 2 | 2026-11-15T21:10:23Z | 2026-11-16T07:12:03Z | 10.03 |
| E yak-64k-v2.0 | 3 | 2026-11-16T17:54:45Z | 2026-11-17T03:56:05Z | 10.02 |
| F ancoda-64k-nojoiner | 1 | 2026-11-17T14:39:52Z | 2026-11-18T00:41:14Z | 10.02 |
| F ancoda-64k-nojoiner | 2 | 2026-11-18T00:56:09Z | 2026-11-18T10:56:59Z | 10.01 |
| F ancoda-64k-nojoiner | 3 | 2026-11-18T11:12:07Z | 2026-11-18T21:12:36Z | 10.01 |
| sum of the 18 runs |  |  |  | 180.91 |

JSON: `artifacts/analysis/arm_f.json`.

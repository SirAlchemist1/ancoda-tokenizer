"""Joiner loss split on sealed_npi_{zwj,zwnj,2000}. Analysis only; MPS/CPU.

Does not train and does not touch the server queue. Uses the same scoring path
as ancoda_tok.m9.bpb (EOS context-only, original_bytes denominator).
"""

from __future__ import annotations

import io
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
import zstandard as zstd

from ancoda_tok.export_hf import bytes_to_unicode
from ancoda_tok.m9.arms import ARMS, load_arm, mac_path
from ancoda_tok.m9.common import REPO_ROOT, bpb, now_utc, text_windows
from ancoda_tok.m9.model import Llama

ZWJ = "\u200d"
ZWNJ = "\u200c"
LN2 = math.log(2)
EVAL = REPO_ROOT / "data" / "m9" / "eval"
SETS = (
    ("sealed_npi_2000", EVAL / "sealed_npi_2000.jsonl.zst", False),
    ("sealed_npi_zwj", EVAL / "perturb" / "sealed_npi_zwj.jsonl.zst", True),
    ("sealed_npi_zwnj", EVAL / "perturb" / "sealed_npi_zwnj.jsonl.zst", True),
)
ARMS_CORE = (
    ("A", "ancoda-64k"),
    ("B", "yak-64k-v1"),
    ("C", "sidskarki-64k"),
)
# Optional follow-up arm (M9 amendment 16 Nov 2026). Pass --include-f to score F.
ARMS_F = (("F", "ancoda-64k-nojoiner"),)
SEEDS = (1, 2, 3)
CKPT = REPO_ROOT / "artifacts" / "m9" / "checkpoints"
TOKENS_BIN = REPO_ROOT / "artifacts" / "m9" / "tokens" / "ancoda-64k" / "tokens.bin"
TOKENS_MAN = REPO_ROOT / "artifacts" / "m9" / "tokens" / "ancoda-64k" / "manifest.json"
OUT_JSON = REPO_ROOT / "artifacts" / "analysis" / "joiner_loss.json"
OUT_MD = REPO_ROOT / "docs" / "ANALYSIS_JOINER_LOSS.md"
PROGRESS = REPO_ROOT / "artifacts" / "analysis" / "joiner_loss_progress.json"
EVAL_BATCH = 2


def _log(msg: str) -> None:
    print(msg, flush=True)


def read_jsonl_zst(path: Path) -> list[dict]:
    rows: list[dict] = []
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as raw, dctx.stream_reader(raw) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def piece_text_hf(tok, tid: int, inverse: dict[str, int]) -> str:
    piece = tok.id_to_token(tid)
    if piece is None:
        return ""
    try:
        raw = bytes(inverse[ch] for ch in piece)
    except KeyError:
        return piece
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def joiner_id_set(arm) -> set[int]:
    out: set[int] = set()
    if arm._hf is not None:
        inverse = {ch: b for b, ch in bytes_to_unicode().items()}
        vocab = int(arm._hf.get_vocab_size(with_added_tokens=True))
        for tid in range(vocab):
            text = piece_text_hf(arm._hf, tid, inverse)
            if ZWJ in text or ZWNJ in text:
                out.add(tid)
    else:
        assert arm._sp is not None
        for tid in range(arm.vocab):
            piece = arm._sp.id_to_piece(tid)
            # SentencePiece keeps joiners as literal characters in the piece.
            if ZWJ in piece or ZWNJ in piece:
                out.add(tid)
    return out


def score_token_nll(
    model: Llama, ids: list[int], eos: int, device: torch.device
) -> np.ndarray:
    """Per-token NLL in nats, one value per content token (same windows as bpb)."""
    if not ids:
        return np.zeros(0, dtype=np.float64)
    losses = np.zeros(len(ids), dtype=np.float64)
    spans = [ids[a:b] for a, b in text_windows(len(ids))]
    offset = 0
    for i in range(0, len(spans), EVAL_BATCH):
        batch_spans = spans[i : i + EVAL_BATCH]
        width = max(len(span) for span in batch_spans) + 1
        batch = torch.zeros(len(batch_spans), width, dtype=torch.long, device=device)
        batch[:, 0] = eos
        for j, span in enumerate(batch_spans):
            if span:
                batch[j, 1 : 1 + len(span)] = torch.tensor(
                    span, dtype=torch.long, device=device
                )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(batch)
        for j, span in enumerate(batch_spans):
            n = len(span)
            if n == 0:
                continue
            nll = F.cross_entropy(
                logits[j, :n].float(),
                batch[j, 1 : 1 + n],
                reduction="none",
            )
            losses[offset : offset + n] = nll.detach().cpu().numpy()
            offset += n
    return losses


def classify_mask(ids: list[int], joiner_ids: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(ids)
    is_joiner = np.array([tid in joiner_ids for tid in ids], dtype=bool)
    is_neighbor = np.zeros(n, dtype=bool)
    for i, flag in enumerate(is_joiner):
        if not flag:
            continue
        if i > 0 and not is_joiner[i - 1]:
            is_neighbor[i - 1] = True
        if i + 1 < n and not is_joiner[i + 1]:
            is_neighbor[i + 1] = True
    is_rest = ~(is_joiner | is_neighbor)
    return is_joiner, is_neighbor, is_rest


def bits(nats: float) -> float:
    return float(nats) / LN2


def load_progress() -> dict:
    if PROGRESS.is_file():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {}


def save_progress(data: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_one(
    letter: str,
    arm_name: str,
    seed: int,
    device: torch.device,
    rows_by_set: dict[str, list[dict]],
    progress: dict,
) -> dict:
    key = f"{arm_name}-s{seed}"
    if key in progress and progress[key].get("complete"):
        _log(f"resume skip {key}")
        return progress[key]

    ckpt = CKPT / f"{arm_name}-s{seed}" / "checkpoint-007630.pt"
    if not ckpt.is_file():
        raise SystemExit(f"STOP: missing checkpoint {ckpt}")
    arm = load_arm(arm_name)
    joiner_ids = joiner_id_set(arm)
    _log(f"load {key} joiner_vocab_ids={len(joiner_ids)} device={device}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = Llama(int(payload["vocab"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    clean_bits_by_id: dict[str, float] = {}
    sets_out: dict[str, dict] = {}
    # Arm A: accumulate per-joiner-id loss on joiner sets.
    type_nll: dict[int, list[float]] = defaultdict(list)

    with torch.inference_mode():
        for set_name, _path, is_joiner_set in SETS:
            rows = rows_by_set[set_name]
            total_nll = 0.0
            total_bytes = 0
            joiner_nll = 0.0
            other_nll = 0.0
            neighbor_nll = 0.0
            rest_nll = 0.0
            n_joiner_tok = 0
            n_neighbor_tok = 0
            n_rest_tok = 0
            extra_bits_sum = 0.0
            paired = 0
            t0 = time.time()
            for i, row in enumerate(rows):
                ids = arm.encode(row["text"])
                nlls = score_token_nll(model, ids, arm.eos_id, device)
                doc_nll = float(nlls.sum()) if len(nlls) else 0.0
                original = int(row["original_bytes"])
                total_nll += doc_nll
                total_bytes += original
                jmask, nmask, rmask = classify_mask(ids, joiner_ids)
                j_nll = float(nlls[jmask].sum()) if jmask.any() else 0.0
                n_nll = float(nlls[nmask].sum()) if nmask.any() else 0.0
                r_nll = float(nlls[rmask].sum()) if rmask.any() else 0.0
                joiner_nll += j_nll
                neighbor_nll += n_nll
                rest_nll += r_nll
                other_nll += n_nll + r_nll
                n_joiner_tok += int(jmask.sum())
                n_neighbor_tok += int(nmask.sum())
                n_rest_tok += int(rmask.sum())

                doc_bits = bits(doc_nll)
                if not is_joiner_set:
                    clean_bits_by_id[row["doc_id"]] = doc_bits
                else:
                    clean_b = clean_bits_by_id.get(row["doc_id"])
                    if clean_b is not None:
                        extra_bits_sum += doc_bits - clean_b
                        paired += 1
                    if letter == "A":
                        for tid, nll in zip(ids, nlls):
                            if tid in joiner_ids:
                                type_nll[int(tid)].append(float(nll))

                if (i + 1) % 100 == 0:
                    _log(
                        f"  {key} {set_name} {i + 1}/{len(rows)} "
                        f"{time.time() - t0:.0f}s"
                    )

            sets_out[set_name] = {
                "docs": len(rows),
                "bytes": total_bytes,
                "nll_nats": total_nll,
                "total_bits": bits(total_nll),
                "bpb": bpb(total_nll, total_bytes) if total_bytes else None,
                "bits_joiner_tokens": bits(joiner_nll),
                "bits_other_tokens": bits(other_nll),
                "bits_neighbor_tokens": bits(neighbor_nll),
                "bits_rest_tokens": bits(rest_nll),
                "n_joiner_tokens": n_joiner_tok,
                "n_neighbor_tokens": n_neighbor_tok,
                "n_rest_tokens": n_rest_tok,
                "extra": None,
            }
            if is_joiner_set:
                # (c) Attribute extra = Δbits by token class using clean vs perturbed
                # class totals. Clean sealed Nepali has (near) zero joiner/neighbor
                # tokens, so this partitions extra_bits exactly:
                #   extra_class = bits_class(pert) − bits_class(clean).
                clean = sets_out["sealed_npi_2000"]
                ej = bits(joiner_nll) - clean["bits_joiner_tokens"]
                en = bits(neighbor_nll) - clean["bits_neighbor_tokens"]
                er = bits(rest_nll) - clean["bits_rest_tokens"]
                sets_out[set_name]["extra"] = {
                    "paired_docs": paired,
                    "extra_bits": extra_bits_sum,
                    "extra_bits_on_joiner_tokens": ej,
                    "extra_bits_on_neighbor_tokens": en,
                    "extra_bits_on_rest_tokens": er,
                    "perturbed_bits_on_joiner_tokens": bits(joiner_nll),
                    "perturbed_bits_on_neighbor_tokens": bits(neighbor_nll),
                    "perturbed_bits_on_rest_tokens": bits(rest_nll),
                    "note": (
                        "extra_bits = total_bits(perturbed) − total_bits(clean), "
                        "paired by doc_id. Class extras are bits_class(pert) − "
                        "bits_class(clean) and sum to extra_bits."
                    ),
                }
            _log(
                f"done {key} {set_name} bpb={sets_out[set_name]['bpb']:.6f} "
                f"bits={sets_out[set_name]['total_bits']:.1f}"
            )

    result = {
        "letter": letter,
        "arm": arm_name,
        "seed": seed,
        "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
        "joiner_vocab_ids": len(joiner_ids),
        "sets": sets_out,
        "complete": True,
    }
    if letter == "A":
        result["type_nll_nats"] = {
            str(tid): {
                "count_on_joiner_sets": len(vals),
                "mean_nll_nats": float(sum(vals) / len(vals)),
                "mean_bits": bits(sum(vals) / len(vals)),
            }
            for tid, vals in sorted(type_nll.items())
            if vals
        }
    progress[key] = result
    # Drop bulky type map from progress? keep it — needed for (d).
    save_progress(progress)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def stream_joiner_counts(joiner_ids: set[int]) -> dict[int, int]:
    """Count joiner-token occurrences in the M9 training stream.

    Prefer the uint16 tokens.bin (content tokens only). If it is missing or
    incomplete, fall back to encoding data/m9/text with ancoda-64k.
    """
    man = json.loads(TOKENS_MAN.read_text(encoding="utf-8")) if TOKENS_MAN.is_file() else {}
    expected = int(man.get("content_tokens", 0)) * 2  # uint16 content tokens
    # tokens.bin stores content tokens only (no EOS); size = content_tokens * 2.
    if man.get("content_tokens"):
        expected = int(man["content_tokens"]) * 2
    if TOKENS_BIN.is_file() and expected and TOKENS_BIN.stat().st_size == expected:
        _log(f"stream counts from tokens.bin ({expected} bytes)")
        counts = np.zeros(65536, dtype=np.int64)
        mm = np.memmap(TOKENS_BIN, dtype="<u2", mode="r")
        chunk = 8_000_000
        for i in range(0, mm.size, chunk):
            part = np.asarray(mm[i : i + chunk])
            counts += np.bincount(part, minlength=65536)
        return {int(tid): int(counts[tid]) for tid in sorted(joiner_ids)}

    _log("tokens.bin incomplete; counting by encoding data/m9/text")
    from ancoda_tok.m9.arms import load_arm

    arm = load_arm("ancoda-64k")
    wanted = np.zeros(65536, dtype=np.uint8)
    for tid in joiner_ids:
        if tid < 65536:
            wanted[tid] = 1
    counts = np.zeros(65536, dtype=np.int64)
    stream = REPO_ROOT / "data" / "m9" / "text"
    shards = sorted(stream.glob("shard-*.jsonl.zst"))
    docs = 0
    for shard in shards:
        for row in read_jsonl_zst(shard):
            ids = arm.encode(row["text"])
            if ids:
                arr = np.asarray(ids, dtype=np.int64)
                # Only accumulate ids that are joiner types.
                mask = wanted[np.clip(arr, 0, 65535)] == 1
                if mask.any():
                    hit = arr[mask]
                    counts += np.bincount(hit, minlength=65536)
            docs += 1
            if docs % 100_000 == 0:
                _log(f"  stream encode docs={docs}")
    return {int(tid): int(counts[tid]) for tid in sorted(joiner_ids)}


def spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: (vals[i], i))
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


VIRAMA = "\u094d"


def _esc(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def _joiner_sites(text: str, joiner: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        i = text.find(joiner, start)
        if i < 0:
            return positions
        if i > 0 and text[i - 1] == VIRAMA:
            positions.append(i)
        start = i + 1


def _site_kind(span: str, joiner: str) -> str:
    if span == joiner:
        return "atomic_joiner"
    idx = span.find(joiner)
    if idx < 0:
        return "split_or_uncovered"
    left = span[:idx]
    right = span[idx + len(joiner) :]
    if left.endswith(VIRAMA) and right:
        return "virama_joiner_and_following"
    if left.endswith(VIRAMA):
        return "ends_with_virama_joiner"
    if not left and right:
        return "starts_with_joiner"
    return "other_merge"


def virama_profile(rows_by_set: dict[str, list[dict]]) -> dict:
    """How arms A and F tokenize a joiner that sits immediately after a virama."""
    profiles: dict[str, dict] = {}
    loaded = {
        "A": load_arm("ancoda-64k"),
        "F": load_arm("ancoda-64k-nojoiner"),
    }
    for set_name, joiner in (("sealed_npi_zwj", ZWJ), ("sealed_npi_zwnj", ZWNJ)):
        rows = rows_by_set[set_name]
        profiles[set_name] = {"joiner": _esc(joiner)}
        for letter, arm in loaded.items():
            assert arm._hf is not None
            tok = arm._hf
            kinds: Counter[str] = Counter()
            pieces: Counter[str] = Counter()
            sites = 0
            following_inside = 0
            for row in rows:
                text = row["text"]
                positions = _joiner_sites(text, joiner)
                if not positions:
                    continue
                enc = tok.encode(text, add_special_tokens=False)
                offsets = enc.offsets
                ti = 0
                n_tok = len(offsets)
                for pos in positions:
                    sites += 1
                    while ti < n_tok and offsets[ti][1] <= pos:
                        ti += 1
                    if ti >= n_tok or not (offsets[ti][0] <= pos < offsets[ti][1]):
                        kinds["split_or_uncovered"] += 1
                        continue
                    a, b = offsets[ti]
                    span = text[a:b]
                    kinds[_site_kind(span, joiner)] += 1
                    pieces[span] += 1
                    if pos + 1 < len(text) and b > pos + 1:
                        following_inside += 1
            top = [
                {"count": count, "chars": len(piece), "piece": _esc(piece)}
                for piece, count in pieces.most_common(8)
            ]
            block = {
                "sites": sites,
                "kinds": dict(sorted(kinds.items())),
                "following_char_inside_same_token": following_inside,
                "distinct_pieces": len(pieces),
                "top_pieces": top,
            }
            profiles[set_name][letter] = block
            _log(
                f"virama {letter} {set_name} sites={sites} "
                f"kinds={block['kinds']} distinct={len(pieces)} "
                f"follow_inside={following_inside}"
            )
    return profiles


def write_md(payload: dict) -> None:
    letter_by_arm = {name: letter for letter, name in ARMS_CORE + ARMS_F}
    scored_arms = sorted(
        {
            key.rsplit("-s", 1)[0]
            for key in payload["runs"]
            if "-s" in key
        }
    )
    arm_list = [(letter_by_arm.get(a, "?"), a) for a in scored_arms]
    lines = [
        "# Joiner loss split",
        "",
        "Analysis only. No training run, and the server queue was not touched. "
        f"Final checkpoints (step 7,630), seeds 1–3, arms {', '.join(a for _, a in arm_list)}. "
        f"Device: `{payload['device']}`. Reproduction: `scripts/analysis_joiner_loss.py` "
        "(add `--include-f` or `--only-f` for arm F).",
        "",
        "BPB denominator: UTF-8 bytes of the original unmodified text "
        "(`original_bytes`). Bits = NLL nats / ln 2. Scoring matches "
        "`ancoda_tok.m9.bpb` (EOS context-only, non-overlapping 1,023-token windows).",
        "",
        "## (a)(b) Total bits, BPB, joiner vs other",
        "",
        "| Arm | Seed | Set | Docs | Bytes | Total bits | BPB | Bits on joiner tokens | Bits on other |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for letter, arm_name in arm_list:
        for seed in SEEDS:
            key = f"{arm_name}-s{seed}"
            if key not in payload["runs"]:
                continue
            block = payload["runs"][key]
            for set_name, _, _ in SETS:
                s = block["sets"][set_name]
                lines.append(
                    f"| {letter} ({arm_name}) | {seed} | `{set_name}` | {s['docs']} | "
                    f"{s['bytes']} | {s['total_bits']:.3f} | {s['bpb']:.6f} | "
                    f"{s['bits_joiner_tokens']:.3f} | {s['bits_other_tokens']:.3f} |"
                )
    lines += [
        "",
        "## (c) Extra bits over clean (`sealed_npi_2000`), by doc_id",
        "",
        "Extra = bits(perturbed) − bits(clean), paired by `doc_id`. "
        "Class columns are bits_class(pert) − bits_class(clean) and sum to Extra.",
        "",
        "| Arm | Seed | Set | Extra bits | Extra on joiner toks | On neighbors | On rest |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for letter, arm_name in arm_list:
        for seed in SEEDS:
            key = f"{arm_name}-s{seed}"
            if key not in payload["runs"]:
                continue
            block = payload["runs"][key]
            for set_name in ("sealed_npi_zwj", "sealed_npi_zwnj"):
                e = block["sets"][set_name]["extra"]
                lines.append(
                    f"| {letter} ({arm_name}) | {seed} | `{set_name}` | "
                    f"{e['extra_bits']:.3f} | "
                    f"{e['extra_bits_on_joiner_tokens']:.3f} | "
                    f"{e['extra_bits_on_neighbor_tokens']:.3f} | "
                    f"{e['extra_bits_on_rest_tokens']:.3f} |"
                )

    d = payload.get("arm_a_type_correlation")
    lines += ["", "## (d) Arm A: joiner token types — stream count vs mean loss", ""]
    if d is None:
        lines.append("Not computed (tokens.bin missing or incomplete).")
    else:
        lines += [
            f"Joiner-containing vocabulary entries with ≥1 occurrence on the joiner "
            f"eval sets: **{d['n_types_scored']}**. "
            f"Spearman rank correlation (stream count vs mean bits on joiner sets): "
            f"**{d['spearman_stream_count_vs_mean_bits']}** "
            f"(n={d['n_types_in_correlation']}).",
            "",
            "Top 15 by mean bits on the joiner sets:",
            "",
            "| Token id | Stream count | Mean bits (joiner sets) | Occurrences scored |",
            "|---:|---:|---:|---:|",
        ]
        for row in d["top15_by_mean_bits"]:
            lines.append(
                f"| {row['token_id']} | {row['stream_count']} | "
                f"{row['mean_bits']:.4f} | {row['count_on_joiner_sets']} |"
            )

    vir = payload.get("virama_after_virama")
    if vir:
        virama_note = (
            "A site is a ZWJ (U+200D) or ZWNJ (U+200C) whose previous character is "
            "virama U+094D. Counts are over `sealed_npi_zwj` and `sealed_npi_zwnj` "
            "(2,000 documents each). The class is the one token whose character span "
            "covers that joiner. `ends_with_virama_joiner` means the token includes "
            "the virama and the joiner and stops before the next character. "
            "`following_inside` is how many sites keep the next character inside "
            "that same token."
        )
        lines += [
            "",
            "## (e) Joiner after a virama, arm F versus arm A",
            "",
            virama_note,
            "",
            "| Set | Arm | Sites | ends with virama+joiner | virama+joiner+following | atomic joiner | other | distinct pieces | following inside |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for set_name in ("sealed_npi_zwj", "sealed_npi_zwnj"):
            block = vir[set_name]
            for letter in ("A", "F"):
                row = block[letter]
                kinds = row["kinds"]
                other = sum(
                    n
                    for name, n in kinds.items()
                    if name
                    not in (
                        "ends_with_virama_joiner",
                        "virama_joiner_and_following",
                        "atomic_joiner",
                    )
                )
                lines.append(
                    f"| `{set_name}` | {letter} | {row['sites']} | "
                    f"{kinds.get('ends_with_virama_joiner', 0)} | "
                    f"{kinds.get('virama_joiner_and_following', 0)} | "
                    f"{kinds.get('atomic_joiner', 0)} | {other} | "
                    f"{row['distinct_pieces']} | "
                    f"{row['following_char_inside_same_token']} |"
                )
        lines += [
            "",
            "Most common token texts at those sites (unicode escapes):",
            "",
        ]
        for set_name in ("sealed_npi_zwj", "sealed_npi_zwnj"):
            lines.append(f"### `{set_name}`")
            lines.append("")
            for letter in ("A", "F"):
                lines.append(f"{letter}:")
                lines.append("")
                lines.append("| Count | Chars | Piece |")
                lines.append("|---:|---:|---|")
                for piece in vir[set_name][letter]["top_pieces"]:
                    lines.append(
                        f"| {piece['count']} | {piece['chars']} | `{piece['piece']}` |"
                    )
                lines.append("")
    lines += ["", "JSON: `artifacts/analysis/joiner_loss.json`.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-f",
        action="store_true",
        help="Also score arm F (ancoda-64k-nojoiner) after A/B/C",
    )
    parser.add_argument(
        "--only-f",
        action="store_true",
        help="Score only arm F (requires F checkpoints under artifacts/m9/checkpoints)",
    )
    args = parser.parse_args()
    arms = list(ARMS_F) if args.only_f else list(ARMS_CORE)
    if args.include_f and not args.only_f:
        arms.extend(ARMS_F)

    started = now_utc()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _log(f"device={device}")

    rows_by_set: dict[str, list[dict]] = {}
    for set_name, path, _ in SETS:
        rows = read_jsonl_zst(path)
        rows_by_set[set_name] = rows
        _log(f"loaded {set_name} docs={len(rows)}")

    def repair_extra(block: dict) -> None:
        """Fill (c) class extras on resumed runs that only stored perturbed partitions."""
        clean = block["sets"].get("sealed_npi_2000")
        if not clean:
            return
        for set_name in ("sealed_npi_zwj", "sealed_npi_zwnj"):
            s = block["sets"].get(set_name)
            if not s or not s.get("extra"):
                continue
            e = s["extra"]
            if "extra_bits_on_joiner_tokens" in e:
                continue
            pj = s["bits_joiner_tokens"]
            pn = s["bits_neighbor_tokens"]
            pr = s["bits_rest_tokens"]
            e["perturbed_bits_on_joiner_tokens"] = pj
            e["perturbed_bits_on_neighbor_tokens"] = pn
            e["perturbed_bits_on_rest_tokens"] = pr
            e["extra_bits_on_joiner_tokens"] = pj - clean["bits_joiner_tokens"]
            e["extra_bits_on_neighbor_tokens"] = pn - clean["bits_neighbor_tokens"]
            e["extra_bits_on_rest_tokens"] = pr - clean["bits_rest_tokens"]
            e["note"] = (
                "extra_bits = total_bits(perturbed) − total_bits(clean), "
                "paired by doc_id. Class extras are bits_class(pert) − "
                "bits_class(clean) and sum to extra_bits."
            )

    virama = None
    if any(name == "ancoda-64k-nojoiner" for _, name in arms):
        _log("virama-joiner tokenization, arm A versus arm F")
        virama = virama_profile(rows_by_set)

    # Clean must be scored before joiner sets for pairing; SETS order ensures that.
    progress = load_progress()
    for block in progress.values():
        if isinstance(block, dict) and block.get("complete"):
            repair_extra(block)
    save_progress(progress)
    runs: dict[str, dict] = {}
    for letter, arm_name in arms:
        for seed in SEEDS:
            ckpt = CKPT / f"{arm_name}-s{seed}" / "checkpoint-007630.pt"
            # A finished checkpoint is 711,132,485 bytes. A shorter file is an
            # rsync still in progress and is not a loadable zip.
            if args.only_f and (
                not ckpt.is_file() or ckpt.stat().st_size != 711_132_485
            ):
                _log(f"skip {arm_name}-s{seed}: checkpoint not complete on the Mac yet")
                continue
            runs[f"{arm_name}-s{seed}"] = run_one(
                letter, arm_name, seed, device, rows_by_set, progress
            )
    if args.only_f and not runs:
        raise SystemExit("STOP: no arm F checkpoint under artifacts/m9/checkpoints")

    corr = None
    if "ancoda-64k-s1" in runs:
        try:
            arm = load_arm("ancoda-64k")
            jids = joiner_id_set(arm)
            _log(f"counting stream joiner ids n={len(jids)}")
            stream_counts = stream_joiner_counts(jids)
            # Merge mean loss across seeds 1–3 for arm A type table.
            merged: dict[int, list[float]] = defaultdict(list)
            occ: dict[int, int] = defaultdict(int)
            for seed in SEEDS:
                block = runs[f"ancoda-64k-s{seed}"]
                for tid_s, info in block.get("type_nll_nats", {}).items():
                    tid = int(tid_s)
                    occ[tid] += int(info["count_on_joiner_sets"])
                    merged[tid].append(
                        (
                            float(info["mean_nll_nats"]),
                            int(info["count_on_joiner_sets"]),
                        )
                    )
            types = []
            xs: list[float] = []
            ys: list[float] = []
            for tid in sorted(merged):
                weighted = 0.0
                w = 0
                for mean_nll, c in merged[tid]:
                    weighted += mean_nll * c
                    w += c
                mean_nll = weighted / w if w else 0.0
                sc = int(stream_counts.get(tid, 0))
                row = {
                    "token_id": tid,
                    "stream_count": sc,
                    "mean_nll_nats": mean_nll,
                    "mean_bits": bits(mean_nll),
                    "count_on_joiner_sets": w,
                }
                types.append(row)
                if w > 0:
                    xs.append(float(sc))
                    ys.append(bits(mean_nll))
            types_sorted = sorted(types, key=lambda r: (-r["mean_bits"], r["token_id"]))
            corr = {
                "n_types_scored": len(types),
                "n_types_in_correlation": len(xs),
                "spearman_stream_count_vs_mean_bits": spearman(xs, ys),
                "top15_by_mean_bits": types_sorted[:15],
                "all_types": types_sorted,
                "stream_tokens": int(
                    json.loads(TOKENS_MAN.read_text(encoding="utf-8"))["stream_tokens"]
                ),
                "content_tokens": int(
                    json.loads(TOKENS_MAN.read_text(encoding="utf-8"))["content_tokens"]
                ),
            }
        except SystemExit as exc:
            _log(str(exc))
            corr = None

    if args.only_f and OUT_JSON.is_file():
        old = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        merged = dict(old.get("runs") or {})
        merged.update(runs)
        runs = merged
        if corr is None:
            corr = old.get("arm_a_type_correlation")
        if virama is None:
            virama = old.get("virama_after_virama")

    payload = {
        "kind": "analysis_joiner_loss",
        "started_at": started,
        "finished_at": now_utc(),
        "device": str(device),
        "ln2": LN2,
        "runs": runs,
        "arm_a_type_correlation": corr,
        "virama_after_virama": virama,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(payload)
    _log(f"wrote {OUT_JSON}")
    _log(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()

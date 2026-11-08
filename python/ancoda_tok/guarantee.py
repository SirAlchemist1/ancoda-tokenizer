"""M6 guarantee checks for the 64k build. Prints a report and returns failures."""

from __future__ import annotations

import argparse
import bisect
import json
import random
import subprocess
from pathlib import Path

import ancoda_seg
import pyarrow.parquet as pq

from ancoda_tok.export_hf import bytes_to_unicode, load_hf
from ancoda_tok.export_tiktoken import load_tiktoken
from ancoda_tok.inventory import find_n50_snapshot, find_snapshot
from ancoda_tok.storage import REPO_ROOT

DEFAULT_OUT = REPO_ROOT / "artifacts" / "tokenizers" / "ancoda-64k"
DEFAULT_COVERED = REPO_ROOT / "artifacts" / "inventory" / "covered_64k.json"
UNICODE = REPO_ROOT / "third_party" / "unicode" / "16.0.0" / "UnicodeData.txt"
SEED = 20261104
MARKS = {"Mn", "Mc", "Me"}

# Mutable paths used by the M6/M7 runners. Defaults match the flagship n50 build.
OUT = DEFAULT_OUT
COVERED = DEFAULT_COVERED
SNAPSHOT: Path | None = None
REPORT_PATH = REPO_ROOT / "artifacts" / "validation" / "guarantee.json"
ORPHANS_PATH = REPO_ROOT / "artifacts" / "validation" / "orphans.txt"
SNAPSHOT_LINE_LIMIT = 10_000_000
SKIP_RANDOM_UNICODE = False
SKIP_BAN_TEST = False


def _marks() -> set[int]:
    found: set[int] = set()
    pending: tuple[int, str] | None = None
    for line in UNICODE.read_text(encoding="utf-8").splitlines():
        fields = line.split(";")
        cp = int(fields[0], 16)
        name = fields[1]
        category = fields[2]
        if name.endswith(", First>"):
            pending = (cp, category)
            continue
        if name.endswith(", Last>"):
            if pending is None:
                raise RuntimeError("UnicodeData Last without First")
            start, start_cat = pending
            pending = None
            if start_cat in MARKS:
                found.update(range(start, cp + 1))
            continue
        if category in MARKS:
            found.add(cp)
    return found


def _covered() -> tuple[set[str], list[str]]:
    data = json.loads(COVERED.read_text(encoding="utf-8"))
    units = [item["akshara"] for item in data["units_selected"]]
    multi = set(units)
    singles = [chr(cp) for cp in range(0x0900, 0x0980)]
    singles.extend(["\u200c", "\u200d"])
    return multi, units + singles


def _inverse() -> dict[str, int]:
    return {char: byte for byte, char in bytes_to_unicode().items()}


def _hf_piece(token: str, inverse: dict[str, int]) -> bytes:
    return bytes(inverse[char] for char in token)


def _bounds(pieces: list[bytes]) -> list[int]:
    total = 0
    bounds = []
    for piece in pieces:
        total += len(piece)
        bounds.append(total)
    return bounds


def _inside(start: int, end: int, bounds: list[int]) -> bool:
    # bounds are cumulative end offsets, so they are sorted.
    index = bisect.bisect_right(bounds, start)
    return index < len(bounds) and bounds[index] < end


def _aksharas(text: str) -> list[tuple[str, int, int]]:
    spans = []
    offset = 0
    for unit in ancoda_seg.aksharas(text, False):
        raw = unit.encode("utf-8")
        spans.append((unit, offset, offset + len(raw)))
        offset += len(raw)
    return spans


def _has_deva(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" or char in "\u200c\u200d" for char in text)


class Engines:
    def __init__(self) -> None:
        self.hf = load_hf(OUT / "tokenizer.json")
        self.tt = load_tiktoken(OUT)
        self.inverse = _inverse()
        self.marks = _marks()
        self.multi, self.units = _covered()

    def pieces(self, text: str) -> tuple[list[int], list[bytes], list[int], list[bytes]]:
        hf_ids = self.hf.encode(text, add_special_tokens=False).ids
        tt_ids = self.tt.encode_ordinary(text)
        hf_pieces = [_hf_piece(self.hf.id_to_token(ident), self.inverse) for ident in hf_ids]
        tt_pieces = [self.tt.decode_single_token_bytes(ident) for ident in tt_ids]
        return hf_ids, hf_pieces, tt_ids, tt_pieces


def _check(
    engines: Engines, text: str, failures: list[str], label: str, stats: dict
) -> tuple[list[int], list[bytes], list[int], list[bytes]]:
    raw = text.encode("utf-8")
    hf_ids, hf_pieces, tt_ids, tt_pieces = engines.pieces(text)
    stats["texts"] += 1
    if hf_ids != tt_ids:
        stats["equality"] += 1
        if len(failures) < 12:
            failures.append(f"equality {label}: {text[:60]!r}")
    if b"".join(hf_pieces) != raw or engines.hf.decode(hf_ids, skip_special_tokens=False) != text:
        stats["hf_round"] += 1
        if len(failures) < 12:
            failures.append(f"hf round-trip {label}: {text[:60]!r}")
    if b"".join(tt_pieces) != raw or engines.tt.decode(tt_ids) != text:
        stats["tt_round"] += 1
        if len(failures) < 12:
            failures.append(f"tt round-trip {label}: {text[:60]!r}")
    if not _has_deva(text):
        return hf_ids, hf_pieces, tt_ids, tt_pieces
    spans = _aksharas(text)
    for name, pieces in (("hf", hf_pieces), ("tt", tt_pieces)):
        bounds = _bounds(pieces)
        for unit, start, end in spans:
            if unit not in engines.multi and unit not in _SINGLE:
                continue
            if _inside(start, end, bounds):
                stats[f"{name}_violations"] += 1
                if len(failures) < 12:
                    failures.append(f"covered split {name} {label}: {unit!r}")
    return hf_ids, hf_pieces, tt_ids, tt_pieces


_SINGLE = {chr(cp) for cp in range(0x0900, 0x0980)} | {"\u200c", "\u200d"}


def _tail(
    engines: Engines,
    text: str,
    pieces: tuple[list[int], list[bytes], list[int], list[bytes]] | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if not _has_deva(text):
        return (0, 0), (0, 0)
    raw_pieces = engines.pieces(text) if pieces is None else pieces
    hf_pieces, tt_pieces = raw_pieces[1], raw_pieces[3]
    spans = _aksharas(text)
    totals = [0, 0]
    broken = [0, 0]
    for index, token_pieces in enumerate((hf_pieces, tt_pieces)):
        bounds = _bounds(token_pieces)
        for unit, start, end in spans:
            if unit in engines.multi or unit in _SINGLE:
                continue
            if not any("\u0900" <= char <= "\u097f" for char in unit):
                continue
            totals[index] += 1
            if _inside(start, end, bounds):
                broken[index] += 1
    return (broken[0], totals[0]), (broken[1], totals[1])


def _bootstrap(pairs: list[tuple[int, int]]) -> tuple[float, float, float]:
    total_b = sum(item[0] for item in pairs)
    total_n = sum(item[1] for item in pairs)
    point = 0.0 if total_n == 0 else total_b / total_n
    if not pairs or total_n == 0:
        return point, point, point
    rng = random.Random(SEED)
    rates = []
    size = len(pairs)
    for _ in range(1000):
        broken = 0
        total = 0
        for _draw in range(size):
            b, n = pairs[rng.randrange(size)]
            broken += b
            total += n
        if total:
            rates.append(broken / total)
    rates.sort()
    if not rates:
        return point, point, point
    lo = rates[int(0.025 * (len(rates) - 1))]
    hi = rates[min(len(rates) - 1, int(0.975 * (len(rates) - 1)))]
    return point, lo, hi


def _vocab(engines: Engines, failures: list[str]) -> int:
    ranks = json.loads((OUT / "ranks.json").read_text(encoding="utf-8"))
    tokens = [bytes.fromhex(item) for item in ranks["tokens_hex"]]
    layout = ranks["layout"]
    bad = 0
    for ident in range(layout["block0_end"], layout["reserved_start"]):
        raw = tokens[ident]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not text:
            continue
        cp = ord(text[0])
        if cp in engines.marks or cp in (0x200C, 0x200D):
            bad += 1
            if len(failures) < 12:
                failures.append(f"mark-initial id {ident} {text[:20]!r}")
    return bad


def _coverage(engines: Engines, failures: list[str]) -> int:
    bad = 0
    for cp in list(range(0x0900, 0x0980)) + [0x200C, 0x200D]:
        text = chr(cp)
        hf_ids, _hp, tt_ids, _tp = engines.pieces(text)
        if len(hf_ids) != 1 or len(tt_ids) != 1 or hf_ids != tt_ids:
            bad += 1
            if len(failures) < 12:
                failures.append(f"coverage U+{cp:04X} hf={hf_ids} tt={tt_ids}")
    return bad


def _units(engines: Engines, contexts: list[str], failures: list[str]) -> int:
    bad = 0
    for unit in engines.units:
        texts = [unit, *[f"{ctx}{unit}" for ctx in contexts]]
        for text in texts:
            raw = text.encode("utf-8")
            unit_raw = unit.encode("utf-8")
            start = raw.rfind(unit_raw)
            _h, hf_pieces, _t, tt_pieces = engines.pieces(text)
            for name, pieces in (("hf", hf_pieces), ("tt", tt_pieces)):
                if b"".join(pieces) != raw or _inside(
                    start, start + len(unit_raw), _bounds(pieces)
                ):
                    bad += 1
                    if len(failures) < 12:
                        failures.append(f"unit {name} {unit!r} in {text[:40]!r}")
                    break
    return bad


def _orphans(
    engines: Engines, texts: list[tuple[str, str]], failures: list[str]
) -> tuple[int, int, int]:
    path = ORPHANS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    listed = 0
    tokens = 0
    bad = 0
    lines = []
    for label, text in texts:
        if not text:
            continue
        _h, hf_pieces, _t, _tt = engines.pieces(text)
        tokens += len(hf_pieces)
        spans = _aksharas(text) if _has_deva(text) else []
        offset = 0
        for piece in hf_pieces:
            try:
                decoded = piece.decode("utf-8")
            except UnicodeDecodeError:
                offset += len(piece)
                continue
            if not decoded:
                offset += len(piece)
                continue
            cp = ord(decoded[0])
            if cp in engines.marks or cp in (0x200C, 0x200D):
                listed += 1
                unit = ""
                for candidate, start, end in spans:
                    if start <= offset < end:
                        unit = candidate
                        break
                # Selected multi-char units must stay whole, so an orphan token
                # inside one is a failure. A lone mark is a malformed sequence.
                if unit in engines.multi:
                    bad += 1
                    if len(failures) < 12:
                        failures.append(f"orphan inside covered {label}: {unit!r}")
                char_at = len(text.encode("utf-8")[:offset].decode("utf-8", errors="ignore"))
                context = text[max(0, char_at - 12) : char_at + 12]
                lines.append(f"{label}\tU+{cp:04X}\t{decoded[:20]!r}\t{context!r}\n")
            offset += len(piece)
    path.write_text("".join(lines), encoding="utf-8")
    return listed, bad, tokens


def _eval_texts() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    flores = REPO_ROOT / "third_party" / "yak-tokenizer" / "data" / "eval" / "flores_plus"
    for path in sorted(flores.rglob("*.txt")):
        found.append((str(path.relative_to(REPO_ROOT)), path.read_text(encoding="utf-8")))
    for source in (
        "aya_npi",
        "belebele_npi",
        "fw2_npi_test",
        "fw2_hin_test",
        "c4_val",
        "fw2_npi_latn_test",
    ):
        for shard in sorted((REPO_ROOT / "data" / "raw" / source).glob("*.parquet")):
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(columns=["text"], batch_size=256):
                for index, value in enumerate(batch.column(0).to_pylist()):
                    found.append((f"{source}/{shard.name}#{index}", "" if value is None else value))
    adversarial = REPO_ROOT / "tests" / "data" / "pretok_adversarial.tsv"
    for line in adversarial.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, payload = line.split("\t", 1)
        found.append((f"adversarial:{name}", json.loads(payload)))
    return found


def _iter_snapshot_lines(limit: int):
    produced = 0
    snapshot = SNAPSHOT if SNAPSHOT is not None else find_n50_snapshot()
    for shard in sorted(snapshot.glob("shard-*.parquet")):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(columns=["text"], batch_size=32):
            for value in batch.column(0).to_pylist():
                text = "" if value is None else value
                for line in text.split("\n"):
                    yield line
                    produced += 1
                    if produced >= limit:
                        return


def _examine_line(
    engines: Engines,
    text: str,
    failures: list[str],
    stats: dict,
    label: str,
    equality: bool,
    doc_hf: list[tuple[int, int]],
    doc_tt: list[tuple[int, int]],
) -> None:
    if not equality and not _has_deva(text):
        return
    raw = text.encode("utf-8")
    hf_ids, hf_pieces, tt_ids, tt_pieces = engines.pieces(text)
    if equality:
        stats["texts"] += 1
        if hf_ids != tt_ids:
            stats["equality"] += 1
            if len(failures) < 12:
                failures.append(f"equality {label}: {text[:60]!r}")
        if (
            b"".join(hf_pieces) != raw
            or engines.hf.decode(hf_ids, skip_special_tokens=False) != text
        ):
            stats["hf_round"] += 1
            if len(failures) < 12:
                failures.append(f"hf round-trip {label}: {text[:60]!r}")
        if b"".join(tt_pieces) != raw or engines.tt.decode(tt_ids) != text:
            stats["tt_round"] += 1
            if len(failures) < 12:
                failures.append(f"tt round-trip {label}: {text[:60]!r}")
    if not _has_deva(text):
        return
    spans = _aksharas(text)
    tail_rows = []
    for name, pieces, key in (
        ("hf", hf_pieces, "hf_violations"),
        ("tt", tt_pieces, "tt_violations"),
    ):
        bounds = _bounds(pieces)
        broken = total = 0
        if b"".join(pieces) == raw:
            for unit, start, end in spans:
                covered = unit in engines.multi or unit in _SINGLE
                deva = any("\u0900" <= char <= "\u097f" for char in unit)
                if covered and _inside(start, end, bounds):
                    stats[key] += 1
                    if len(failures) < 12:
                        failures.append(f"covered split {name} {label}: {unit!r}")
                if not covered and deva:
                    total += 1
                    if _inside(start, end, bounds):
                        broken += 1
        tail_rows.append((broken, total))
    if tail_rows[0][1] or tail_rows[1][1]:
        doc_hf.append(tail_rows[0])
        doc_tt.append(tail_rows[1])


def _random_unicode(count: int) -> list[str]:
    rng = random.Random(SEED)
    texts = []
    for _ in range(count):
        chars = []
        for _char in range(rng.randint(1, 24)):
            cp = rng.randrange(0x110000)
            if 0xD800 <= cp <= 0xDFFF:
                cp = 0x20
            chars.append(chr(cp))
        texts.append("".join(chars))
    return texts


def configure(
    *,
    out: Path | None = None,
    covered: Path | None = None,
    snapshot: Path | None = None,
    report: Path | None = None,
    orphans: Path | None = None,
    snapshot_line_limit: int | None = None,
    skip_random_unicode: bool = False,
    skip_ban_test: bool = False,
) -> None:
    """Point the guarantee runner at a tokenizer / covered set / snapshot."""
    global OUT, COVERED, SNAPSHOT, REPORT_PATH, ORPHANS_PATH
    global SNAPSHOT_LINE_LIMIT, SKIP_RANDOM_UNICODE, SKIP_BAN_TEST
    OUT = Path(out) if out is not None else DEFAULT_OUT
    COVERED = Path(covered) if covered is not None else DEFAULT_COVERED
    SNAPSHOT = Path(snapshot) if snapshot is not None else None
    REPORT_PATH = (
        Path(report)
        if report is not None
        else REPO_ROOT / "artifacts" / "validation" / "guarantee.json"
    )
    ORPHANS_PATH = (
        Path(orphans)
        if orphans is not None
        else REPO_ROOT / "artifacts" / "validation" / "orphans.txt"
    )
    if snapshot_line_limit is not None:
        SNAPSHOT_LINE_LIMIT = snapshot_line_limit
    SKIP_RANDOM_UNICODE = skip_random_unicode
    SKIP_BAN_TEST = skip_ban_test


def run() -> list[str]:
    if not (OUT / "tokenizer.json").exists():
        raise SystemExit(f"STOP: missing {OUT / 'tokenizer.json'}")
    failures: list[str] = []
    stats = {
        "texts": 0,
        "equality": 0,
        "hf_round": 0,
        "tt_round": 0,
        "hf_violations": 0,
        "tt_violations": 0,
    }
    print(f"loading exports from {OUT}", flush=True)
    engines = Engines()
    mark_initial = _vocab(engines, failures)
    coverage_bad = _coverage(engines, failures)
    print(f"mark-initial block1/2={mark_initial} coverage_bad={coverage_bad}", flush=True)

    specials = json.loads((OUT / "special_tokens_map.json").read_text(encoding="utf-8"))
    names = [specials["bos_token"], specials["eos_token"], specials["pad_token"]]
    names.extend(specials["additional_special_tokens"])
    extras = [
        ("digits", "12345678901234567890"),
        ("deva-digits", "१२३४५६७८९०"),
        ("joiners", "क्‍षर्‍य"),
        ("mixed", "Nepal नेपाल 123 ।"),
        ("punct-nl", "हो ।\n\n"),
        ("lone-mark", "ा"),
        ("all-specials", " ".join(names)),
    ]
    for name in names:
        extras.append((f"special:{name}", name))
        extras.append((f"special-in:{name}", f"नेपाली {name} 123"))
    for label, text in extras:
        if text:
            _check(engines, text, failures, label, stats)

    print("eval texts", flush=True)
    eval_texts = _eval_texts()
    print(f"loaded eval texts={len(eval_texts)}", flush=True)
    hf_tail: list[tuple[int, int]] = []
    tt_tail: list[tuple[int, int]] = []
    for index, (label, text) in enumerate(eval_texts):
        pieces = _check(engines, text, failures, label, stats)
        hf_one, tt_one = _tail(engines, text, pieces)
        hf_tail.append(hf_one)
        tt_tail.append(tt_one)
        if index % 2000 == 0:
            print(
                f"eval {index} equality={stats['equality']} hf_violations={stats['hf_violations']}",
                flush=True,
            )
    npi = [
        item
        for item in eval_texts
        if item[0].endswith("npi_Deva.txt") or item[0].startswith("aya_npi/")
    ]
    orphans, orphan_bad, orphan_tokens = _orphans(engines, npi, failures)
    orphan_share = 0.0 if orphan_tokens == 0 else orphans / orphan_tokens
    print(
        f"orphans listed={orphans} bad={orphan_bad} tokens={orphan_tokens} share={orphan_share}",
        flush=True,
    )

    print("snapshot lines", flush=True)
    contexts: list[str] = []
    doc_hf: list[tuple[int, int]] = []
    doc_tt: list[tuple[int, int]] = []
    snapshot_count = 0
    nonempty = 0
    context_rng = random.Random(SEED)
    for line in _iter_snapshot_lines(SNAPSHOT_LINE_LIMIT):
        _examine_line(
            engines,
            line,
            failures,
            stats,
            f"snap:{snapshot_count}",
            snapshot_count < 1_000_000,
            doc_hf,
            doc_tt,
        )
        if line:
            nonempty += 1
            if len(contexts) < 100:
                contexts.append(line)
            else:
                slot = context_rng.randrange(nonempty)
                if slot < 100:
                    contexts[slot] = line
        snapshot_count += 1
        if snapshot_count % 100000 == 0:
            print(
                f"snapshot {snapshot_count} equality={stats['equality']} "
                f"hf_violations={stats['hf_violations']}",
                flush=True,
            )
    print("per-unit contexts", flush=True)
    unit_bad = _units(engines, contexts, failures)
    print(f"unit_bad={unit_bad}", flush=True)

    if SKIP_RANDOM_UNICODE:
        print("random unicode skipped", flush=True)
    else:
        print("random unicode", flush=True)
        for index, text in enumerate(_random_unicode(100_000)):
            _check(engines, text, failures, f"rand:{index}", stats)

    if SKIP_BAN_TEST:
        ban_code = 0
        print("ban predicate cargo test skipped", flush=True)
    else:
        env = dict(**{k: v for k, v in __import__("os").environ.items()})
        cargo_bin = str(Path.home() / ".cargo" / "bin")
        path = env.get("PATH", "")
        if cargo_bin not in path.split(":"):
            env["PATH"] = f"{cargo_bin}:{path}"
        ban = subprocess.run(
            [
                "cargo",
                "test",
                "-p",
                "ancoda-bpe",
                "--offline",
                "--lib",
                "ban::",
                "--",
                "--quiet",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        ban_code = ban.returncode
        if ban_code != 0:
            failures.append("ban predicate cargo test failed")
            failures.append(ban.stderr[-500:])

    hf_ci = _bootstrap(doc_hf)
    tt_ci = _bootstrap(doc_tt)
    snapshot = SNAPSHOT if SNAPSHOT is not None else find_n50_snapshot()
    report = {
        "tokenizer_dir": str(OUT.relative_to(REPO_ROOT)),
        "covered": str(COVERED.relative_to(REPO_ROOT)),
        "snapshot": str(snapshot.relative_to(REPO_ROOT)),
        "texts_checked_in_equality_pass": stats["texts"],
        "equality_mismatches": stats["equality"],
        "hf_roundtrip_failures": stats["hf_round"],
        "tt_roundtrip_failures": stats["tt_round"],
        "hf_covered_violations": stats["hf_violations"],
        "tt_covered_violations": stats["tt_violations"],
        "mark_initial_block1_or_block2": mark_initial,
        "coverage_failures": coverage_bad,
        "unit_context_failures": unit_bad,
        "orphan_occurrences_npi_aya": orphans,
        "orphan_tokens_npi_aya": orphan_tokens,
        "orphan_mark_share_npi_aya": orphan_share,
        "orphan_inside_covered": orphan_bad,
        "tail_hf_rate_lo_hi": hf_ci,
        "tail_tt_rate_lo_hi": tt_ci,
        "snapshot_lines": snapshot_count,
        "eval_texts": len(eval_texts),
        "special_names": len(names),
        "ban_predicate_exit": ban_code,
        "failures": failures,
    }
    dest = REPORT_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--covered", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--snapshot-prefix", type=str, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--orphans", type=Path, default=None)
    parser.add_argument("--snapshot-line-limit", type=int, default=None)
    parser.add_argument("--skip-random-unicode", action="store_true")
    parser.add_argument("--skip-ban-test", action="store_true")
    args = parser.parse_args()
    snapshot = args.snapshot
    if snapshot is None and args.snapshot_prefix:
        snapshot = find_snapshot(args.snapshot_prefix)
    configure(
        out=args.tokenizer_dir,
        covered=args.covered,
        snapshot=snapshot,
        report=args.report,
        orphans=args.orphans,
        snapshot_line_limit=args.snapshot_line_limit,
        skip_random_unicode=args.skip_random_unicode,
        skip_ban_test=args.skip_ban_test,
    )
    failed = run()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

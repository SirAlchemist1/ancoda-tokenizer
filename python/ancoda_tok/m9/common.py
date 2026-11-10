"""Constants and pure helpers locked in docs/M9_PREREG.md."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

SHARES: dict[str, float] = {
    "nepali": 0.40,
    "hindi": 0.15,
    "other_deva": 0.05,
    "english": 0.35,
    "code": 0.05,
}
GROUPS = tuple(SHARES)
OTHER_BUCKETS = frozenset({"Marathi", "Sanskrit", "Smaller Devanagari", "Languages of Nepal"})
SAMPLE_BYTES = 50_000_000
CONTENT_TOKEN_FLOOR = 1_050_000_000
SAMPLE_PAD = 1.03
CONTEXT = 1024
SEQUENCES_PER_STEP = 128
TOKENS_PER_STEP = SEQUENCES_PER_STEP * CONTEXT
FULL_STEPS = 7630
FULL_STREAM_TOKENS = FULL_STEPS * TOKENS_PER_STEP
PEAK_LR = 1e-3
MIN_LR = 1e-4
WARMUP_STEPS = round(0.02 * FULL_STEPS)
SELECTION_PREFIX = b"m9-v1\n"
DEVA11 = (
    "Hindi",
    "Marathi",
    "Nepali",
    "Sanskrit",
    "anp_Deva",
    "awa_Deva",
    "bho_Deva",
    "gom_Deva",
    "hne_Deva",
    "mai_Deva",
    "new_Deva",
)
_DISPLAY = {
    "npi_Deva": "Nepali",
    "hin_Deva": "Hindi",
    "mar_Deva": "Marathi",
    "san_Deva": "Sanskrit",
    "eng_Latn": "English",
    "Nepali": "Nepali",
    "Hindi": "Hindi",
    "Marathi": "Marathi",
    "Sanskrit": "Sanskrit",
    "English": "English",
    "Code": "Code",
    "Math": "Math",
}
EOS_NAMES = ("<|eos|>", "<|endoftext|>", "</s>")


def mix_group(bucket: str) -> str | None:
    if bucket == "Nepali":
        return "nepali"
    if bucket == "Hindi":
        return "hindi"
    if bucket == "English":
        return "english"
    if bucket == "Code":
        return "code"
    if bucket in OTHER_BUCKETS:
        return "other_deva"
    return None


def canonical_lang(bucket: str, lang: str) -> str:
    if lang in _DISPLAY:
        return _DISPLAY[lang]
    if bucket in _DISPLAY:
        return _DISPLAY[bucket]
    return lang


def rank_u64(doc_id: str) -> int:
    digest = hashlib.sha256(SELECTION_PREFIX + doc_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def rank_hex(rank: int) -> str:
    return f"{rank:016x}"


def doc_id_sha256(doc_id: str) -> str:
    return hashlib.sha256(doc_id.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def update_list_hash(hasher: hashlib._Hash, doc_id: str) -> None:
    hasher.update(doc_id.encode("utf-8"))
    hasher.update(b"\n")


def group_targets(total: int) -> dict[str, int]:
    """Exact share of a multiple of 20. Nepali absorbs any remainder (there is none)."""
    if total % 20 != 0:
        raise ValueError(f"total bytes must be a multiple of 20, got {total}")
    out = {name: int(total * share) for name, share in SHARES.items()}
    gap = total - sum(out.values())
    out["nepali"] += gap
    return out


def multiple_of_20(n: int, *, up: bool) -> int:
    if up:
        return ((n + 19) // 20) * 20
    return (n // 20) * 20


def choose_total_bytes(min_tpb: float, available: dict[str, int]) -> int:
    """Smallest multiple of 20 that can reach the content-token floor, with a 3% pad."""
    if min_tpb <= 0:
        raise ValueError("min tokens per byte must be positive")
    need = math.ceil(CONTENT_TOKEN_FLOOR / min_tpb)
    need_pad = math.ceil(need * SAMPLE_PAD)
    caps = []
    for name, share in SHARES.items():
        caps.append(int(available.get(name, 0) / share))
    max_total = multiple_of_20(min(caps), up=False)
    padded = multiple_of_20(need_pad, up=True)
    exact = multiple_of_20(need, up=True)
    if padded <= max_total:
        return padded
    if exact <= max_total:
        return max_total if max_total >= need else exact
    raise SystemExit(
        f"STOP: lowest arm needs {need} bytes to reach {CONTENT_TOKEN_FLOOR} "
        f"content tokens at {min_tpb:.6f} tok/byte, but the mix caps the stream "
        f"at {max_total} bytes. available={available}"
    )


def bpb(nll_nats: float, original_bytes: int) -> float:
    if original_bytes <= 0:
        raise ValueError("original_bytes must be positive")
    return nll_nats / (math.log(2) * original_bytes)


def unweighted_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("no values")
    return sum(values) / len(values)


def paired_call(diffs: list[float]) -> str:
    """Label for BPB(A) - BPB(other) on three seeds. See docs/M9_PREREG.md."""
    if len(diffs) != 3:
        raise ValueError("the primary rule uses three seeds")
    if not all(d < 0 for d in diffs):
        return "no measurable difference"
    mean = sum(diffs) / 3.0
    var = sum((d - mean) ** 2 for d in diffs) / 2.0
    std = math.sqrt(var)
    if abs(mean) > std:
        return "A better"
    return "no measurable difference"


def text_windows(n_tokens: int, context: int = CONTEXT) -> list[tuple[int, int]]:
    """Half-open spans of at most context-1 text tokens. EOS occupies position 0."""
    if n_tokens < 0:
        raise ValueError(n_tokens)
    span = context - 1
    if span <= 0:
        raise ValueError(context)
    out: list[tuple[int, int]] = []
    i = 0
    while i < n_tokens:
        j = min(n_tokens, i + span)
        out.append((i, j))
        i = j
    return out


def eos_choice(specials: dict[str, int], vocab: int) -> tuple[int, str, int]:
    """Return (eos_id, rule, model_vocab). specials maps a special-token string to its id."""
    for name in EOS_NAMES:
        if name in specials:
            eos = specials[name]
            if eos < 0 or eos >= vocab:
                raise ValueError(f"{name} id {eos} is outside vocab {vocab}")
            return eos, name, vocab
    return vocab, "extended", vocab + 1


def storage_dtype(max_id: int) -> str:
    if max_id <= 65535:
        return "uint16"
    return "uint32"


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def attach_build(manifest: dict, build_path: Path) -> dict:
    """Copy the Mac build record into a server artifact manifest."""
    path = Path(build_path)
    if not path.is_file():
        raise SystemExit(f"STOP: missing build record {path}")
    build = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "git_commit",
        "config_sha256",
        "library_versions",
        "unicode",
        "unicode_segmentation",
    )
    missing = [key for key in required if key not in build]
    if missing:
        raise SystemExit(f"STOP: build record missing {missing}")
    manifest["git_commit"] = build["git_commit"]
    manifest["config_sha256"] = build["config_sha256"]
    manifest["library_versions"] = build["library_versions"]
    manifest["unicode"] = build["unicode"]
    manifest["unicode_segmentation"] = build["unicode_segmentation"]
    return manifest

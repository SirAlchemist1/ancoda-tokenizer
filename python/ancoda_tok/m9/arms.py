"""Frozen tokenizer arms. Loading does not modify the tokenizer files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ancoda_tok.m9.common import REPO_ROOT, eos_choice, storage_dtype

YAK_SNAPSHOT = (
    "data/hf_cache/hub/models--ampixa--yak-tokenizer/snapshots/"
    "12b6d3010ec542b484ec3b673354852fe8bb85fb"
)
SID_MODEL = (
    "data/hf_cache/hub/models--sidskarki--nepali-bpe-tokenizer/snapshots/"
    "fc5773dcf7b9dc2272e8f595f99828969621a855/nepali_bpe_64k.model"
)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    letter: str
    kind: str
    core: bool
    mac_rel: str


ARMS: dict[str, ArmSpec] = {
    "ancoda-64k": ArmSpec(
        "ancoda-64k",
        "A",
        "hf",
        True,
        "artifacts/tokenizers/m8/ancoda-64k/tokenizer.json",
    ),
    "yak-64k-v1": ArmSpec(
        "yak-64k-v1",
        "B",
        "hf",
        True,
        f"{YAK_SNAPSHOT}/v1.0/64k/tokenizer.json",
    ),
    "sidskarki-64k": ArmSpec("sidskarki-64k", "C", "sp", True, SID_MODEL),
    "yak-64k-v2.0": ArmSpec(
        "yak-64k-v2.0",
        "E",
        "hf",
        False,
        f"{YAK_SNAPSHOT}/v2.0/64k/tokenizer.json",
    ),
}
CORE_ARMS = tuple(name for name, spec in ARMS.items() if spec.core)
PILOT_ARMS = ("ancoda-64k", "sidskarki-64k", "yak-64k-v1", "yak-64k-v2.0")


class LoadedArm:
    def __init__(self, spec: ArmSpec, path: Path) -> None:
        self.spec = spec
        self.path = path
        self.kind = spec.kind
        if spec.kind == "hf":
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(path))
            tok.no_truncation()
            tok.no_padding()
            added = tok.get_added_tokens_decoder()
            specials: dict[str, int] = {}
            for tid, info in added.items():
                if info.special:
                    specials[info.content] = int(tid)
            vocab = int(tok.get_vocab_size(with_added_tokens=True))
            self._hf = tok
            self._sp = None
        elif spec.kind == "sp":
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor(model_file=str(path))
            vocab = int(sp.vocab_size())
            specials = {}
            if int(sp.eos_id()) >= 0:
                specials["</s>"] = int(sp.eos_id())
            self._hf = None
            self._sp = sp
        else:
            raise ValueError(spec.kind)
        eos, rule, model_vocab = eos_choice(specials, vocab)
        self.vocab = vocab
        self.eos_id = eos
        self.eos_rule = rule
        self.model_vocab = model_vocab
        self.dtype = storage_dtype(max(eos, vocab - 1))

    def encode(self, text: str) -> list[int]:
        if self._hf is not None:
            return list(self._hf.encode(text, add_special_tokens=False).ids)
        assert self._sp is not None
        return list(self._sp.encode(text, out_type=int, add_bos=False, add_eos=False))

    def decode(self, ids: list[int]) -> str:
        if self._hf is not None:
            return str(self._hf.decode(ids, skip_special_tokens=False))
        assert self._sp is not None
        return str(self._sp.decode(ids))


def mac_path(spec: ArmSpec) -> Path:
    return REPO_ROOT / spec.mac_rel


def load_arm(name: str, path: Path | None = None) -> LoadedArm:
    spec = ARMS[name]
    return LoadedArm(spec, path if path is not None else mac_path(spec))

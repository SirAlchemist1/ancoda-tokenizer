"""Independent Python implementation of PLAN.md 3.3.

Grapheme clusters follow UAX #29 for Unicode 16.0.0, using
``GraphemeBreakProperty.txt``, ``DerivedCoreProperties.txt`` (InCB), and
``emoji-data.txt`` (Extended_Pictographic). The default mode then starts a unit
at every Devanagari letter (General_Category Lo and Script Devanagari, from
``UnicodeData.txt`` and ``Scripts.txt``) that is not the cluster's first code
point. ``gb9c=True`` returns the raw clusters.

This module does not import the Rust segmenter.
"""

from __future__ import annotations

import array
from pathlib import Path

OTHER = 0
CR = 1
LF = 2
CONTROL = 3
EXTEND = 4
ZWJ = 5
RI = 6
PREPEND = 7
SPACING = 8
L = 9
V = 10
T = 11
LV = 12
LVT = 13

GCB_VALUE = {
    "CR": CR,
    "LF": LF,
    "Control": CONTROL,
    "Extend": EXTEND,
    "ZWJ": ZWJ,
    "Regional_Indicator": RI,
    "Prepend": PREPEND,
    "SpacingMark": SPACING,
    "L": L,
    "V": V,
    "T": T,
    "LV": LV,
    "LVT": LVT,
}

INCB_NONE = 0
INCB_EXTEND = 1
INCB_LINKER = 2
INCB_CONSONANT = 3

INCB_VALUE = {
    "None": INCB_NONE,
    "Extend": INCB_EXTEND,
    "Linker": INCB_LINKER,
    "Consonant": INCB_CONSONANT,
}

_TABLES: Tables | None = None


def repo_root() -> Path:
    for cand in Path(__file__).resolve().parents:
        if (cand / "PLAN.md").is_file():
            return cand
    raise RuntimeError("PLAN.md not found")


def unicode_dir() -> Path:
    return repo_root() / "third_party" / "unicode" / "16.0.0"


def _blank() -> array.array:
    return array.array("B", b"\x00" * 0x110000)


def _fill(table: array.array, start: int, end: int, value: int) -> None:
    span = end - start + 1
    table[start : end + 1] = array.array("B", bytes((value,)) * span)


def _range(field: str) -> tuple[int, int]:
    field = field.strip()
    if ".." in field:
        left, right = field.split("..", 1)
        return int(left, 16), int(right, 16)
    value = int(field, 16)
    return value, value


def _body(line: str) -> str:
    return line.split("#", 1)[0].strip()


class Tables:
    def __init__(self) -> None:
        self.gcb = _blank()
        self.incb = _blank()
        self.extpict = _blank()
        self.dev = _blank()
        self.block: dict[int, str] = {}
        self.letters: list[int] = []

    @classmethod
    def load(cls, root: Path) -> Tables:
        tables = cls()
        tables._load_gcb(root / "auxiliary" / "GraphemeBreakProperty.txt")
        tables._load_incb(root / "DerivedCoreProperties.txt")
        tables._load_extpict(root / "emoji" / "emoji-data.txt")
        tables._load_letters(root / "Scripts.txt", root / "UnicodeData.txt")
        if tables.incb[0x200D] != INCB_EXTEND or tables.incb[0x200C] != INCB_NONE:
            raise RuntimeError("ZWJ/ZWNJ Indic_Conjunct_Break values are not Unicode 16.0.0")
        if tables.incb[0x094D] != INCB_LINKER or tables.incb[0x0915] != INCB_CONSONANT:
            raise RuntimeError("Devanagari InCB values are not Unicode 16.0.0")
        if tables.gcb[0x200D] != ZWJ or tables.gcb[0x200C] != EXTEND:
            raise RuntimeError("ZWJ/ZWNJ grapheme classes are not Unicode 16.0.0")
        return tables

    def _load_gcb(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            body = _body(line)
            if not body or body.startswith("@"):
                continue
            field, value = (part.strip() for part in body.split(";", 1))
            code = GCB_VALUE.get(value)
            if code is None:
                raise RuntimeError(f"unknown Grapheme_Cluster_Break value {value}")
            start, end = _range(field)
            _fill(self.gcb, start, end, code)

    def _load_incb(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            body = _body(line)
            if not body or body.startswith("@"):
                continue
            parts = [part.strip() for part in body.split(";")]
            if len(parts) < 3 or parts[1] != "InCB":
                continue
            code = INCB_VALUE.get(parts[2])
            if code is None:
                raise RuntimeError(f"unknown InCB value {parts[2]}")
            start, end = _range(parts[0])
            _fill(self.incb, start, end, code)

    def _load_extpict(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            body = _body(line)
            if not body:
                continue
            field, value = (part.strip() for part in body.split(";", 1))
            if value != "Extended_Pictographic":
                continue
            start, end = _range(field)
            _fill(self.extpict, start, end, 1)

    def _load_letters(self, scripts: Path, data: Path) -> None:
        in_script = _blank()
        for line in scripts.read_text(encoding="utf-8").splitlines():
            body = _body(line)
            if not body:
                continue
            field, name = (part.strip() for part in body.split(";", 1))
            if name != "Devanagari":
                continue
            start, end = _range(field)
            _fill(in_script, start, end, 1)
        range_start: tuple[int, str] | None = None
        letters: list[int] = []
        for line in data.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            fields = line.split(";")
            cp = int(fields[0], 16)
            name = fields[1]
            category = fields[2]
            if name.endswith(", First>"):
                range_start = (cp, category)
                continue
            if name.endswith(", Last>"):
                if range_start is None:
                    raise RuntimeError(f"UnicodeData Last without First at {fields[0]}")
                start, start_cat = range_start
                range_start = None
                if start_cat == "Lo":
                    for item in range(start, cp + 1):
                        if in_script[item]:
                            self.dev[item] = 1
                            letters.append(item)
                continue
            if 0x0900 <= cp <= 0x097F:
                self.block[cp] = category
            if category == "Lo" and in_script[cp]:
                self.dev[cp] = 1
                letters.append(cp)
        letters.sort()
        self.letters = letters
        if not letters:
            raise RuntimeError("Devanagari letter set is empty")

    def segment(self, text: str, gb9c: bool) -> list[str]:
        if not text:
            return []
        if text.isascii() and "\r" not in text:
            # ASCII other than CR/LF breaks between every code point, and none
            # of those code points is a Devanagari letter.
            return list(text)
        gcb = self.gcb
        incb = self.incb
        ext = self.extpict
        cps = [ord(ch) for ch in text]
        n = len(cps)
        starts = [0]
        for index in range(n - 1):
            if _break_at(cps, index, gcb, incb, ext):
                starts.append(index + 1)
        starts.append(n)
        parts = [text[starts[i] : starts[i + 1]] for i in range(len(starts) - 1)]
        if gb9c:
            return parts
        return _override(parts, self.dev)


def _break_at(
    cps: list[int],
    index: int,
    gcb: array.array,
    incb: array.array,
    ext: array.array,
) -> bool:
    left = cps[index]
    right = cps[index + 1]
    gl = gcb[left]
    gr = gcb[right]
    if gl == CR and gr == LF:
        return False
    if gl in (CONTROL, CR, LF) or gr in (CONTROL, CR, LF):
        return True
    if gl == L and gr in (L, V, LV, LVT):
        return False
    if gl in (LV, V) and gr in (V, T):
        return False
    if gl in (LVT, T) and gr == T:
        return False
    if gr in (EXTEND, ZWJ) or gr == SPACING or gl == PREPEND:
        return False
    if _gb9c(cps, index, incb):
        return False
    if gl == ZWJ and _gb11(cps, index, gcb, ext):
        return False
    return not _gb12(cps, index, gcb)


def _gb9c(cps: list[int], index: int, incb: array.array) -> bool:
    if incb[cps[index + 1]] != INCB_CONSONANT:
        return False
    seen_linker = False
    cursor = index
    while cursor >= 0:
        value = incb[cps[cursor]]
        if value == INCB_LINKER:
            seen_linker = True
        elif value == INCB_EXTEND:
            pass
        elif value == INCB_CONSONANT:
            return seen_linker
        else:
            return False
        cursor -= 1
    return False


def _gb11(cps: list[int], index: int, gcb: array.array, ext: array.array) -> bool:
    if gcb[cps[index]] != ZWJ or not ext[cps[index + 1]]:
        return False
    cursor = index - 1
    while cursor >= 0 and gcb[cps[cursor]] == EXTEND:
        cursor -= 1
    return cursor >= 0 and bool(ext[cps[cursor]])


def _gb12(cps: list[int], index: int, gcb: array.array) -> bool:
    if gcb[cps[index]] != RI or gcb[cps[index + 1]] != RI:
        return False
    count = 0
    cursor = index
    while cursor >= 0 and gcb[cps[cursor]] == RI:
        count += 1
        cursor -= 1
    return count % 2 == 1


def _override(parts: list[str], dev: array.array) -> list[str]:
    out: list[str] = []
    for part in parts:
        if len(part) < 2:
            out.append(part)
            continue
        start = 0
        for index, ch in enumerate(part):
            if index and dev[ord(ch)]:
                out.append(part[start:index])
                start = index
        out.append(part[start:])
    return out


def tables() -> Tables:
    global _TABLES
    if _TABLES is None:
        _TABLES = Tables.load(unicode_dir())
    return _TABLES


def aksharas(text: str, gb9c: bool = False) -> list[str]:
    return tables().segment(text, gb9c)


def devanagari_letters() -> list[int]:
    return list(tables().letters)


def block_category(cp: int) -> str | None:
    return tables().block.get(cp)


def iter_grapheme_break_cases(path: Path | None = None):
    """Yield ``(text, clusters)`` from Unicode's GraphemeBreakTest.txt."""
    source = path or (unicode_dir() / "auxiliary" / "GraphemeBreakTest.txt")
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        body = _body(line)
        if not body:
            continue
        tokens = body.split()
        if tokens[0] != "÷":
            raise RuntimeError(f"GraphemeBreakTest line {line_no} does not start with a break")
        chars: list[str] = []
        break_after: list[bool] = []
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in {"÷", "×"}:
                raise RuntimeError(f"GraphemeBreakTest line {line_no} has an unexpected operator")
            cp = int(token, 16)
            if cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF:
                raise RuntimeError(f"GraphemeBreakTest line {line_no} code point {token}")
            chars.append(chr(cp))
            index += 1
            operator = tokens[index]
            if operator not in {"÷", "×"}:
                raise RuntimeError(f"GraphemeBreakTest line {line_no} missing break operator")
            break_after.append(operator == "÷")
            index += 1
        clusters: list[str] = []
        buf: list[str] = []
        for ch, is_break in zip(chars, break_after, strict=True):
            buf.append(ch)
            if is_break:
                clusters.append("".join(buf))
                buf = []
        if buf:
            raise RuntimeError(f"GraphemeBreakTest line {line_no} has no final break")
        yield "".join(chars), clusters

"""cleaning step 2: GlotLID v3 document-level language ID."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ancoda_tok.clean.common import REPO_ROOT, _script_name


@dataclass(frozen=True)
class LidDecision:
    action: str  # keep | reroute | drop | skip
    label: str
    prob: float
    target_bucket: str | None = None


_NUMPY2_PATCHED = False


def _patch_numpy_array_for_fasttext() -> None:
    """fasttext-wheel calls ``np.array(..., copy=False)``, rejected by NumPy 2."""
    global _NUMPY2_PATCHED
    if _NUMPY2_PATCHED:
        return
    import numpy as np

    original = np.array

    def _array(*args, **kwargs):  # noqa: ANN001
        if kwargs.get("copy") is False:
            kwargs = dict(kwargs)
            kwargs.pop("copy")
            return np.asarray(*args, **kwargs)
        return original(*args, **kwargs)

    np.array = _array  # type: ignore[method-assign]
    _NUMPY2_PATCHED = True


class GlotLID:
    """Thin wrapper around the local GlotLID v3 ``model.bin``."""

    def __init__(self, model_path: Path | None = None) -> None:
        import fasttext

        _patch_numpy_array_for_fasttext()
        path = model_path or (REPO_ROOT / "data" / "raw" / "glotlid" / "model.bin")
        self._model = fasttext.load_model(str(path))

    def predict(self, text: str) -> tuple[str, float]:
        # Newlines break fasttext's single-line predict; truncate long docs.
        sample = text.replace("\n", " ").replace("\r", " ").strip()
        if not sample:
            return "und_Latn", 0.0
        if len(sample) > 4000:
            sample = sample[:4000]
        labels, probs = self._model.predict(sample, k=1)
        label = labels[0].removeprefix("__label__")
        return label, float(probs[0])


def decide_lid(
    label: str,
    prob: float,
    *,
    source_bucket: str,
    keep_labels: set[str],
    label_buckets: dict[str, str],
    skip_lid: bool,
) -> LidDecision:
    if skip_lid:
        return LidDecision(action="skip", label=label, prob=prob, target_bucket=source_bucket)
    if label in keep_labels:
        return LidDecision(action="keep", label=label, prob=prob, target_bucket=source_bucket)
    if label in label_buckets:
        target = label_buckets[label]
        if target == source_bucket:
            return LidDecision(action="keep", label=label, prob=prob, target_bucket=source_bucket)
        return LidDecision(action="reroute", label=label, prob=prob, target_bucket=target)
    return LidDecision(action="drop", label=label, prob=prob, target_bucket=None)


def audit_script_share(texts: list[str], expected_scripts: list[str]) -> float:
    """Share of non-space characters outside the expected scripts."""
    if not texts or not expected_scripts:
        return 0.0
    expected = set(expected_scripts)
    outside = 0
    total = 0
    for text in texts:
        for ch in text:
            if ch.isspace():
                continue
            total += 1
            name = _script_name(ch)
            if name not in expected and name not in {"Common", "Inherited", "Unknown"}:
                outside += 1
    if total == 0:
        return 0.0
    return outside / total


def empty_stats() -> dict[str, Any]:
    return {
        "docs_in": 0,
        "bytes_in": 0,
        "docs_kept": 0,
        "bytes_kept": 0,
        "docs_rerouted": 0,
        "bytes_rerouted": 0,
        "docs_dropped": 0,
        "bytes_dropped": 0,
        "docs_utf8_dropped": 0,
        "docs_legacy_dropped": 0,
        "confusion": {},
        "prob_hist": {},
        "script_outside_share": None,
    }

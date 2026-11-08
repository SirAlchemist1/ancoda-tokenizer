"""M4 step 1: UTF-8 validity and control stripping. No normalization."""

from __future__ import annotations

from ancoda_tok.clean.common import strip_controls


def clean_utf8_text(text: str | None) -> str | None:
    """Return cleaned text, or None if the document must be dropped.

    Documents that are not valid Unicode strings never reach here (parquet
    stores Unicode). Empty-after-strip documents are dropped.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        return None
    # Surrogate-heavy or non-str payloads are rejected.
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return None
    cleaned = strip_controls(text)
    if cleaned.strip() == "":
        return None
    return cleaned

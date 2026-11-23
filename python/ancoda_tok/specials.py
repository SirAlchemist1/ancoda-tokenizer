"""The 1,024 reserved special tokens (docs/SPEC.md 3.7).

The same list is produced by ``ancoda-bpe specials``. Ordinary-text encoding must
not treat these names as specials; see ``docs/MODEL_CARD_NOTES.md``.
"""

from __future__ import annotations

NAMED_SPECIALS: tuple[str, ...] = (
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|start_turn|>",
    "<|end_turn|>",
    "<|think|>",
    "<|/think|>",
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_result|>",
    "<|/tool_result|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|mask|>",
    "<|sep|>",
)


def special_token_names() -> list[str]:
    names = list(NAMED_SPECIALS)
    names.extend(f"<|reserved_{index}|>" for index in range(len(NAMED_SPECIALS), 1024))
    return names

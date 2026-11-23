# Model card notes (trainer)

These notes are for the release-candidate model card. They record behavior the card must state.

## Ban 1 (human correction, 8 Nov 2026)

A Block 2 merge `M = L + R` is allowed under the character-integrity ban when either:

- both `L` and `R` are complete, valid UTF-8 strings, or
- `M` lies inside one character's UTF-8 encoding: it starts with a valid lead byte, every later byte is a valid continuation for that position, and `len(M)` is at most that lead's sequence length.

Proper prefixes are allowed. `E6 + 97` may merge to `E6 97`, and `E6 97 + A5` may complete the character. A merge is banned when it starts at a continuation byte, is structurally invalid (overlong, surrogate, out of range), or joins part of one character to bytes outside it.

## Special tokens in ordinary text (human correction, 8 Nov 2026)

The 1,024 special tokens are added tokens. They are inserted by the post-processor (`<|bos|>` before a sequence) when `add_special_tokens=True`.

A literal special-token string in ordinary text, such as the characters `<|bos|>`, must be encoded as plain text. That matches tiktoken `encode_ordinary`.

tokenizers 0.23.3 implements this with `Tokenizer.encode_special_tokens = True`. The installed library's default is `False`, and `tokenizer.json` does not store the flag. After `Tokenizer.from_file`, `False` extracts added special tokens from the text, so `<|bos|>` becomes one special id. Callers that want ordinary-text behavior, including every Ancoda equality and guarantee check, set:

```python
tokenizer.encode_special_tokens = True
tokenizer.encode(text, add_special_tokens=False)
```

`add_special_tokens=False` keeps the post-processor from inserting a leading `<|bos|>`. `encode_special_tokens=True` keeps a literal `<|bos|>` already in the text from being swallowed. With the Hugging Face `transformers` wrapper, set `tokenizer.backend_tokenizer.encode_special_tokens = True` before encoding.

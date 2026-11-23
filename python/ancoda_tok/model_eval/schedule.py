"""Learning-rate schedule from docs/MODEL_EVAL_PREREG.md."""

from __future__ import annotations

import math

from ancoda_tok.model_eval.common import FULL_STEPS, MIN_LR, PEAK_LR, WARMUP_STEPS


def learning_rate(
    step: int,
    *,
    total_steps: int = FULL_STEPS,
    warmup: int = WARMUP_STEPS,
    peak: float = PEAK_LR,
    min_lr: float = MIN_LR,
) -> float:
    """0-indexed step. Warmup uses (step + 1) / warmup. Cosine starts at `warmup`."""
    if step < 0:
        raise ValueError(step)
    if warmup < 1:
        raise ValueError(warmup)
    if step < warmup:
        return peak * (step + 1) / warmup
    if total_steps <= warmup:
        return min_lr
    progress = (step - warmup) / (total_steps - warmup)
    if progress > 1:
        progress = 1.0
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak - min_lr) * cosine

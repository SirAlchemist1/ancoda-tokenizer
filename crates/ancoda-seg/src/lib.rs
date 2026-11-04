//! Ancoda pre-tokenizer, akshara segmenter, and counting.
//! Implementation lands in M3+. M0 only pins deps and asserts Unicode version.

use unicode_segmentation::UNICODE_VERSION;

/// Unicode version shipped by the pinned `unicode-segmentation` crate.
pub fn unicode_version() -> (u64, u64, u64) {
    UNICODE_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_version_is_16_0_0() {
        assert_eq!(
            unicode_version(),
            (16, 0, 0),
            "PLAN.md 3.3 requires unicode-segmentation 1.12.0 with Unicode 16.0.0"
        );
    }
}

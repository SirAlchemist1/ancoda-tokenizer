//! Pre-tokenizer and akshara segmenter.
//!
//! `aksharas(text, gb9c=false)` is docs/SPEC.md 3.3: Unicode 16.0.0 extended grapheme
//! clusters, then a boundary before every Devanagari letter that is not the
//! cluster's first code point. `gb9c=true` returns the raw clusters and is the
//! ablation mode. The pre-tokenizer is the tokenizers crate Split (onig),
//! behavior isolated, over `configs/pretokenizer_pattern.txt`.
//!
//! `pretokenize_fancy` compiles that same pattern with fancy-regex 0.19.0, the
//! version in tiktoken 0.14.0's Cargo.toml, and keeps each `find_iter` match.
//! It exists so the cross-engine check can see tiktoken's regex engine.

mod letters;
mod pretok;
mod segment;

pub use letters::{devanagari_block_category, devanagari_letter_codepoints, is_devanagari_letter};
pub use pretok::{pretokenize, pretokenize_batch, pretokenize_fancy, pretokenize_fancy_batch};
pub use segment::{aksharas, aksharas_batch};

use unicode_segmentation::UNICODE_VERSION;

/// Unicode version shipped by the pinned `unicode-segmentation` crate.
pub fn unicode_version() -> (u64, u64, u64) {
    UNICODE_VERSION
}

pub fn unicode_dir() -> &'static std::path::Path {
    std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../third_party/unicode/16.0.0"
    ))
}

pub fn pattern_path() -> &'static std::path::Path {
    std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/pretokenizer_pattern.txt"
    ))
}

#[cfg(feature = "python")]
mod py {
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;

    use crate::{pretok, segment};

    fn err(message: String) -> PyErr {
        PyRuntimeError::new_err(message)
    }

    #[pyfunction]
    fn unicode_version() -> (u64, u64, u64) {
        crate::unicode_version()
    }

    #[pyfunction]
    fn devanagari_letters() -> Vec<u32> {
        crate::devanagari_letter_codepoints()
    }

    #[pyfunction]
    #[pyo3(signature = (text, gb9c=false))]
    fn aksharas(text: &str, gb9c: bool) -> Vec<String> {
        segment::aksharas(text, gb9c)
    }

    #[pyfunction]
    #[pyo3(signature = (texts, gb9c=false))]
    fn aksharas_batch(py: Python<'_>, texts: Vec<String>, gb9c: bool) -> Vec<Vec<String>> {
        py.allow_threads(|| segment::aksharas_batch(&texts, gb9c))
    }

    #[pyfunction]
    fn pretokenize(text: &str) -> PyResult<Vec<String>> {
        pretok::pretokenize(text).map_err(err)
    }

    #[pyfunction]
    fn pretokenize_batch(py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<String>>> {
        py.allow_threads(|| pretok::pretokenize_batch(&texts))
            .map_err(err)
    }

    #[pyfunction]
    fn pretokenize_fancy(text: &str) -> PyResult<Vec<String>> {
        pretok::pretokenize_fancy(text).map_err(err)
    }

    #[pyfunction]
    fn pretokenize_fancy_batch(py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<String>>> {
        py.allow_threads(|| pretok::pretokenize_fancy_batch(&texts))
            .map_err(err)
    }

    #[pymodule]
    fn ancoda_seg(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(unicode_version, m)?)?;
        m.add_function(wrap_pyfunction!(devanagari_letters, m)?)?;
        m.add_function(wrap_pyfunction!(aksharas, m)?)?;
        m.add_function(wrap_pyfunction!(aksharas_batch, m)?)?;
        m.add_function(wrap_pyfunction!(pretokenize, m)?)?;
        m.add_function(wrap_pyfunction!(pretokenize_batch, m)?)?;
        m.add_function(wrap_pyfunction!(pretokenize_fancy, m)?)?;
        m.add_function(wrap_pyfunction!(pretokenize_fancy_batch, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_version_is_16_0_0() {
        assert_eq!(
            unicode_version(),
            (16, 0, 0),
            "docs/SPEC.md 3.3 requires unicode-segmentation 1.12.0 with Unicode 16.0.0"
        );
    }
}

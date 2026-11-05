//! Pre-tokenizer. The production path is tokenizers' onig Split.
//! The fancy-regex path is only the tiktoken engine in the equality check.

use std::fs;
use std::sync::OnceLock;

use fancy_regex::Regex as FancyRegex;
use rayon::prelude::*;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::tokenizer::PreTokenizedString;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizer, SplitDelimiterBehavior};

use crate::pattern_path;

fn pattern() -> Result<&'static str, String> {
    static CELL: OnceLock<String> = OnceLock::new();
    if CELL.get().is_none() {
        let text = fs::read_to_string(pattern_path()).map_err(|err| {
            format!(
                "read {}: {err}. Run python -m ancoda_tok.pattern first.",
                pattern_path().display()
            )
        })?;
        if text.contains('\n') || text.contains('\r') {
            return Err("pretokenizer pattern file must be a single line with no newline".into());
        }
        let _ = CELL.set(text);
    }
    Ok(CELL.get().expect("pattern set").as_str())
}

fn onig_split() -> Result<&'static Split, String> {
    static CELL: OnceLock<Split> = OnceLock::new();
    if let Some(split) = CELL.get() {
        return Ok(split);
    }
    let compiled = Split::new(
        SplitPattern::Regex(pattern()?.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|err| err.to_string())?;
    let _ = CELL.set(compiled);
    Ok(CELL.get().expect("onig split set"))
}

fn fancy() -> Result<&'static FancyRegex, String> {
    static CELL: OnceLock<FancyRegex> = OnceLock::new();
    if let Some(regex) = CELL.get() {
        return Ok(regex);
    }
    let compiled = FancyRegex::new(pattern()?).map_err(|err| err.to_string())?;
    let _ = CELL.set(compiled);
    Ok(CELL.get().expect("fancy regex set"))
}

pub fn pretokenize(text: &str) -> Result<Vec<String>, String> {
    let mut pretokenized = PreTokenizedString::from(text);
    onig_split()?
        .pre_tokenize(&mut pretokenized)
        .map_err(|err| err.to_string())?;
    Ok(pretokenized
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(piece, _, _)| piece.to_string())
        .collect())
}

pub fn pretokenize_batch(texts: &[String]) -> Result<Vec<Vec<String>>, String> {
    texts.par_iter().map(|text| pretokenize(text)).collect()
}

pub fn pretokenize_fancy(text: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for found in fancy()?.find_iter(text) {
        let found = found.map_err(|err| err.to_string())?;
        if found.as_str().is_empty() {
            return Err("fancy-regex produced an empty pre-token".into());
        }
        out.push(found.as_str().to_string());
    }
    Ok(out)
}

pub fn pretokenize_fancy_batch(texts: &[String]) -> Result<Vec<Vec<String>>, String> {
    texts
        .par_iter()
        .map(|text| pretokenize_fancy(text))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expect_both(text: &str, expected: &[&str]) {
        let onig = pretokenize(text).unwrap();
        let fancy = pretokenize_fancy(text).unwrap();
        let owned: Vec<String> = expected.iter().map(|part| (*part).to_string()).collect();
        assert_eq!(onig, owned, "onig {text:?}");
        assert_eq!(fancy, owned, "fancy {text:?}");
        assert_eq!(onig.concat(), text);
        assert_eq!(fancy.concat(), text);
    }

    #[test]
    fn pattern_file_uses_the_required_joiner_escapes() {
        let pat = pattern().unwrap();
        assert_eq!(pat.matches(r"\p{M}\u200C\u200D]").count(), 4);
        assert!(!pat.contains(r"\p{M}]"));
        assert!(!pat.contains(r"\p{N}{"));
        assert!(pat.contains(r"\u200C"));
        assert!(pat.contains(r"\u200D"));
        assert!(!pat.contains('\u{200C}'));
        assert!(!pat.contains('\u{200D}'));
        assert!(!pat.contains(r"\x{"));
    }

    #[test]
    fn plan_31_examples() {
        expect_both(
            "नेपालको राजधानी काठमाडौं हो ।\n",
            &["नेपालको", " राजधानी", " काठमाडौं", " हो", " ।\n"],
        );
        expect_both(
            "र्\u{200D}य क्\u{200C}ष १२३ abc 123",
            &[
                "र्\u{200D}य",
                " क्\u{200C}ष",
                " ",
                "१",
                "२",
                "३",
                " abc",
                " ",
                "1",
                "2",
                "3",
            ],
        );
    }

    #[test]
    fn adversarial_file_engines_agree() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/data/pretok_adversarial.tsv"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let mut count = 0;
        for line in text.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (id, payload) = line.split_once('\t').unwrap();
            let input: String = serde_json::from_str(payload).unwrap();
            let onig = pretokenize(&input).unwrap();
            let fancy = pretokenize_fancy(&input).unwrap();
            assert_eq!(onig, fancy, "{id}");
            assert_eq!(onig.concat(), input, "{id} did not cover the input");
            count += 1;
        }
        assert!(count >= 20, "pretok_adversarial.tsv has {count} cases");
    }
}

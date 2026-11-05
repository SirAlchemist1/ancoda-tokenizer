//! Akshara segmentation (PLAN.md 3.3).

use rayon::prelude::*;
use unicode_segmentation::UnicodeSegmentation;

use crate::letters::is_devanagari_letter;

/// Extended grapheme clusters when `gb9c` is true. Otherwise the 3.3 override:
/// inside each cluster, start a new unit at every Devanagari letter that is not
/// the cluster's first code point.
pub fn aksharas(text: &str, gb9c: bool) -> Vec<String> {
    let mut out = Vec::new();
    for cluster in text.graphemes(true) {
        if gb9c {
            out.push(cluster.to_string());
            continue;
        }
        let mut start = 0;
        for (index, ch) in cluster.char_indices() {
            if index > 0 && is_devanagari_letter(ch) {
                out.push(cluster[start..index].to_string());
                start = index;
            }
        }
        out.push(cluster[start..].to_string());
    }
    out
}

pub fn aksharas_batch(texts: &[String], gb9c: bool) -> Vec<Vec<String>> {
    texts.par_iter().map(|text| aksharas(text, gb9c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::letters::{devanagari_block_category, is_devanagari_letter};
    use crate::unicode_dir;
    use std::fs;

    fn raw(text: &str) -> Vec<String> {
        aksharas(text, true)
    }

    fn units(text: &str) -> Vec<String> {
        aksharas(text, false)
    }

    fn eq(got: Vec<String>, expected: &[&str]) {
        let owned: Vec<String> = expected.iter().map(|part| (*part).to_string()).collect();
        assert_eq!(got, owned);
    }

    #[test]
    fn plan_required_examples() {
        eq(units("स्कूल"), &["स्", "कू", "ल"]);
        eq(units("गर्नु"), &["ग", "र्", "नु"]);
        eq(units("क्ष"), &["क्", "ष"]);
        eq(units("र्\u{200D}य"), &["र्\u{200D}", "य"]);
        eq(units("\u{0958}"), &["\u{0958}"]);
        eq(units("\u{0915}\u{093C}"), &["\u{0915}\u{093C}"]);
        assert_ne!("\u{0958}", "\u{0915}\u{093C}");
    }

    #[test]
    fn plan_boundary_table_and_checked_raw_clusters() {
        eq(units("क्ष"), &["क्", "ष"]);
        eq(raw("क्ष"), &["क्ष"]);
        eq(units("क्\u{200D}ष"), &["क्\u{200D}", "ष"]);
        eq(raw("क्\u{200D}ष"), &["क्\u{200D}ष"]);
        eq(units("क्\u{200C}ष"), &["क्\u{200C}", "ष"]);
        eq(raw("क्\u{200C}ष"), &["क्\u{200C}", "ष"]);
        eq(units("र्\u{200D}य"), &["र्\u{200D}", "य"]);
        eq(units("र\u{200D}\u{094D}"), &["र\u{200D}\u{094D}"]);
        eq(raw("र\u{200D}\u{094D}य"), &["र\u{200D}\u{094D}य"]);
        eq(units("र\u{200D}\u{094D}य"), &["र\u{200D}\u{094D}", "य"]);
        eq(units("क्\u{093C}ष"), &["क्\u{093C}", "ष"]);
        eq(raw("क्\u{093C}ष"), &["क्\u{093C}ष"]);
        eq(units("क्\u{094D}ष"), &["क्\u{094D}", "ष"]);
        eq(raw("क्\u{094D}ष"), &["क्\u{094D}ष"]);
        eq(units("\u{0600}क"), &["\u{0600}", "क"]);
        eq(raw("\u{0600}क"), &["\u{0600}क"]);
        eq(units("क़ा"), &["क़ा"]);
        eq(units("कां"), &["कां"]);
        eq(units("काँ"), &["काँ"]);
        eq(units("कः"), &["कः"]);
        eq(units("अं"), &["अं"]);
        eq(units("\u{093E}"), &["\u{093E}"]);
        eq(units(" \u{093E}"), &[" \u{093E}"]);
        eq(units("क\u{0301}"), &["क\u{0301}"]);
        eq(units("१२३"), &["१", "२", "३"]);
        eq(units("।"), &["।"]);
        eq(units("॥"), &["॥"]);
    }

    #[test]
    fn devanagari_block_joiners_are_marks_and_none_prepend() {
        for cp in 0x0900..=0x097F {
            let ch = char::from_u32(cp).unwrap();
            let joined = format!("क{ch}");
            let clusters = raw(&joined);
            let joins = clusters.len() == 1;
            let category = devanagari_block_category(cp);
            let mark = matches!(category.as_deref(), Some("Mn" | "Mc" | "Me"));
            if joins {
                assert!(
                    mark,
                    "U+{cp:04X} category {category:?} joined क's raw cluster"
                );
            }
            let forward = format!("{ch}a");
            assert_eq!(
                raw(&forward).len(),
                2,
                "U+{cp:04X} behaved as Prepend before 'a'"
            );
        }
        eq(raw("क\u{200D}"), &["क\u{200D}"]);
        eq(raw("क\u{200C}"), &["क\u{200C}"]);
    }

    #[test]
    fn default_mode_only_adds_devanagari_letter_boundaries() {
        let samples = [
            "स्कूल गर्नु क्ष",
            "क्\u{200D}ष क्\u{200C}ष",
            "aकbক\u{09CD}\u{0996}",
            "\u{0600}क१।\n\u{093E}",
            "👨\u{200D}👩",
        ];
        for text in samples {
            let raw_parts = raw(text);
            let default_parts = units(text);
            assert_eq!(raw_parts.concat(), text);
            assert_eq!(default_parts.concat(), text);
            assert_eq!(default_parts, override_of(&raw_parts));
        }
    }

    fn override_of(raw_parts: &[String]) -> Vec<String> {
        let mut out = Vec::new();
        for cluster in raw_parts {
            let mut start = 0;
            for (index, ch) in cluster.char_indices() {
                if index > 0 && is_devanagari_letter(ch) {
                    out.push(cluster[start..index].to_string());
                    start = index;
                }
            }
            out.push(cluster[start..].to_string());
        }
        out
    }

    #[test]
    fn grapheme_break_test_passes() {
        let path = unicode_dir().join("auxiliary/GraphemeBreakTest.txt");
        let text = fs::read_to_string(&path).unwrap();
        let mut cases = 0;
        for (line_no, line) in text.lines().enumerate() {
            let body = line.split('#').next().unwrap_or("").trim();
            if body.is_empty() {
                continue;
            }
            let (input, expected) = parse_case(body, line_no + 1);
            let got = raw(&input);
            assert_eq!(got, expected, "GraphemeBreakTest line {}", line_no + 1);
            assert_eq!(units(&input), override_of(&got));
            cases += 1;
        }
        assert!(cases >= 500, "parsed only {cases} grapheme cases");
    }

    fn parse_case(body: &str, line_no: usize) -> (String, Vec<String>) {
        let tokens: Vec<&str> = body.split_whitespace().collect();
        assert_eq!(tokens.first().copied(), Some("÷"), "line {line_no}");
        let mut chars = Vec::new();
        let mut break_after = Vec::new();
        let mut index = 1;
        while index < tokens.len() {
            let hex = tokens[index];
            assert!(hex != "÷" && hex != "×", "line {line_no}");
            let cp = u32::from_str_radix(hex, 16).unwrap();
            let ch = char::from_u32(cp).unwrap_or_else(|| panic!("line {line_no} cp {hex}"));
            chars.push(ch);
            index += 1;
            let op = tokens[index];
            assert!(op == "÷" || op == "×", "line {line_no}");
            break_after.push(op == "÷");
            index += 1;
        }
        let mut expected = Vec::new();
        let mut buf = String::new();
        for (ch, is_break) in chars.iter().zip(break_after.iter()) {
            buf.push(*ch);
            if *is_break {
                expected.push(std::mem::take(&mut buf));
            }
        }
        assert!(buf.is_empty(), "line {line_no} missing final break");
        (chars.into_iter().collect(), expected)
    }

    #[test]
    fn akshara_cases_file() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/data/akshara_cases.tsv"
        );
        let text = fs::read_to_string(path).unwrap();
        let mut count = 0;
        for line in text.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut fields = line.split('\t');
            let id = fields.next().unwrap();
            let input: String = serde_json::from_str(fields.next().unwrap()).unwrap();
            let default_expected: Vec<String> =
                serde_json::from_str(fields.next().unwrap()).unwrap();
            let gb9c_expected: Vec<String> = serde_json::from_str(fields.next().unwrap()).unwrap();
            assert_eq!(units(&input), default_expected, "{id} default");
            assert_eq!(raw(&input), gb9c_expected, "{id} gb9c");
            count += 1;
        }
        assert!(count >= 200, "akshara_cases.tsv has {count} cases");
    }
}

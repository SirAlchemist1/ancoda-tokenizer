//! General_Category Mn / Mc / Me from Unicode 16.0.0 UnicodeData.txt.
//! Ban 2 also rejects a merge whose decoded form starts with U+200C or U+200D.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Clone)]
pub struct MarkSet {
    marks: HashSet<u32>,
}

impl MarkSet {
    pub fn load(unicode_data: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(unicode_data)
            .map_err(|err| format!("read {}: {err}", unicode_data.display()))?;
        let mut marks = HashSet::new();
        let mut pending: Option<(u32, String)> = None;
        for raw in text.lines() {
            if raw.is_empty() {
                continue;
            }
            let mut fields = raw.split(';');
            let cp_hex = fields.next().ok_or("UnicodeData code point")?;
            let name = fields.next().ok_or("UnicodeData name")?;
            let category = fields.next().ok_or("UnicodeData category")?;
            let cp = u32::from_str_radix(cp_hex, 16).map_err(|err| err.to_string())?;
            if name.ends_with(", First>") {
                pending = Some((cp, category.to_string()));
                continue;
            }
            if name.ends_with(", Last>") {
                let (start, start_cat) = pending.take().ok_or("UnicodeData Last without First")?;
                if is_mark(&start_cat) {
                    for code in start..=cp {
                        marks.insert(code);
                    }
                }
                continue;
            }
            if is_mark(category) {
                marks.insert(cp);
            }
        }
        if marks.is_empty() {
            return Err("mark set is empty".into());
        }
        Ok(Self { marks })
    }

    pub fn contains(&self, cp: u32) -> bool {
        self.marks.contains(&cp)
    }

    pub fn empty() -> Self {
        Self {
            marks: HashSet::new(),
        }
    }
}

fn is_mark(category: &str) -> bool {
    matches!(category, "Mn" | "Mc" | "Me")
}

/// Ban 2. Incomplete UTF-8 does not "start with" a code point, so it is not mark-initial.
pub fn starts_with_forbidden_mark(bytes: &[u8], marks: &MarkSet) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    let Some(ch) = text.chars().next() else {
        return false;
    };
    let cp = ch as u32;
    cp == 0x200C || cp == 0x200D || marks.contains(cp)
}

/// Both Ban 1 and Ban 2. Pre-token boundaries are enforced by training on pre-tokens.
pub fn block2_merge_allowed(left: &[u8], right: &[u8], marks: &MarkSet) -> bool {
    if !crate::ban::merge_allowed(left, right) {
        return false;
    }
    let mut merged = Vec::with_capacity(left.len() + right.len());
    merged.extend_from_slice(left);
    merged.extend_from_slice(right);
    !starts_with_forbidden_mark(&merged, marks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marks() -> MarkSet {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../third_party/unicode/16.0.0/UnicodeData.txt");
        MarkSet::load(&path).expect("UnicodeData")
    }

    #[test]
    fn vowel_sign_is_mark_initial_and_consonant_is_not() {
        let marks = marks();
        // U+093E DEVANAGARI VOWEL SIGN AA is Mc. U+0915 DEVANAGARI LETTER KA is Lo.
        let aa = [0xE0, 0xA4, 0xBE];
        let ka = [0xE0, 0xA4, 0x95];
        assert!(starts_with_forbidden_mark(&aa, &marks));
        assert!(!starts_with_forbidden_mark(&ka, &marks));
        assert!(!block2_merge_allowed(&[0xE0, 0xA4], &[0xBE], &marks));
        assert!(block2_merge_allowed(&[0xE0, 0xA4], &[0x95], &marks));
        // A proper prefix does not decode, so Ban 2 does not fire; Ban 1 allows it.
        assert!(block2_merge_allowed(&[0xE0], &[0xA4], &marks));
        assert!(starts_with_forbidden_mark(&[0xE2, 0x80, 0x8C], &marks));
        assert!(starts_with_forbidden_mark(&[0xE2, 0x80, 0x8D], &marks));
    }
}

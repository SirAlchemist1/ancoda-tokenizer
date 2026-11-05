//! Devanagari letters = General_Category Lo ∩ Script Devanagari, from Unicode 16.0.0.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::OnceLock;

use crate::unicode_dir;

struct Tables {
    letters: HashSet<char>,
    block_category: HashMap<u32, String>,
}

fn tables() -> &'static Tables {
    static CELL: OnceLock<Tables> = OnceLock::new();
    CELL.get_or_init(|| load().expect("Unicode 16.0.0 Devanagari tables"))
}

fn load() -> Result<Tables, String> {
    let root = unicode_dir();
    let scripts = fs::read_to_string(root.join("Scripts.txt")).map_err(|err| err.to_string())?;
    let data = fs::read_to_string(root.join("UnicodeData.txt")).map_err(|err| err.to_string())?;
    let devanagari = parse_script(&scripts, "Devanagari")?;
    let mut letters = HashSet::new();
    let mut block_category = HashMap::new();
    let mut range_start: Option<(u32, String)> = None;

    for raw in data.lines() {
        if raw.is_empty() {
            continue;
        }
        let mut fields = raw.split(';');
        let cp_hex = fields.next().ok_or("UnicodeData code point")?;
        let name = fields.next().ok_or("UnicodeData name")?;
        let category = fields.next().ok_or("UnicodeData category")?;
        let cp = u32::from_str_radix(cp_hex, 16).map_err(|err| err.to_string())?;

        if name.ends_with(", First>") {
            range_start = Some((cp, category.to_string()));
            continue;
        }
        if name.ends_with(", Last>") {
            let (start, start_cat) = range_start.take().ok_or("UnicodeData Last without First")?;
            if start_cat == "Lo" {
                insert_lo_range(&mut letters, &devanagari, start, cp);
            }
            continue;
        }
        if (0x0900..=0x097F).contains(&cp) {
            block_category.insert(cp, category.to_string());
        }
        if category == "Lo" && devanagari.contains(&cp) {
            if let Some(ch) = char::from_u32(cp) {
                letters.insert(ch);
            }
        }
    }
    if letters.is_empty() {
        return Err("Devanagari letter set is empty".to_string());
    }
    Ok(Tables {
        letters,
        block_category,
    })
}

fn insert_lo_range(letters: &mut HashSet<char>, devanagari: &HashSet<u32>, start: u32, end: u32) {
    for cp in start..=end {
        if devanagari.contains(&cp) {
            if let Some(ch) = char::from_u32(cp) {
                letters.insert(ch);
            }
        }
    }
}

fn parse_script(text: &str, script: &str) -> Result<HashSet<u32>, String> {
    let mut out = HashSet::new();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(';');
        let range = parts
            .next()
            .ok_or_else(|| format!("script range: {line}"))?
            .trim();
        let name = parts
            .next()
            .ok_or_else(|| format!("script name: {line}"))?
            .trim();
        if name != script {
            continue;
        }
        let (start, end) = parse_range(range)?;
        for cp in start..=end {
            out.insert(cp);
        }
    }
    Ok(out)
}

fn parse_range(field: &str) -> Result<(u32, u32), String> {
    if let Some((left, right)) = field.split_once("..") {
        let start = u32::from_str_radix(left, 16).map_err(|err| err.to_string())?;
        let end = u32::from_str_radix(right, 16).map_err(|err| err.to_string())?;
        return Ok((start, end));
    }
    let cp = u32::from_str_radix(field, 16).map_err(|err| err.to_string())?;
    Ok((cp, cp))
}

pub fn is_devanagari_letter(ch: char) -> bool {
    tables().letters.contains(&ch)
}

pub fn devanagari_letter_codepoints() -> Vec<u32> {
    let mut cps: Vec<u32> = tables().letters.iter().copied().map(u32::from).collect();
    cps.sort_unstable();
    cps
}

pub fn devanagari_block_category(cp: u32) -> Option<String> {
    tables().block_category.get(&cp).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letter_set_matches_the_spec_examples() {
        assert!(is_devanagari_letter('क'));
        assert!(is_devanagari_letter('अ'));
        assert!(is_devanagari_letter('\u{0950}'));
        assert!(is_devanagari_letter('\u{093D}'));
        assert!(is_devanagari_letter('\u{0958}'));
        assert!(!is_devanagari_letter('\u{094D}'));
        assert!(!is_devanagari_letter('\u{093E}'));
        assert!(!is_devanagari_letter('\u{093C}'));
        assert!(!is_devanagari_letter('\u{0966}'));
        assert!(!is_devanagari_letter('\u{0964}'));
        assert!(!is_devanagari_letter('\u{200C}'));
        assert!(!is_devanagari_letter('\u{200D}'));
        assert_eq!(devanagari_block_category(0x0915).as_deref(), Some("Lo"));
        assert_eq!(devanagari_block_category(0x094D).as_deref(), Some("Mn"));
        assert_eq!(devanagari_block_category(0x093E).as_deref(), Some("Mc"));
    }
}

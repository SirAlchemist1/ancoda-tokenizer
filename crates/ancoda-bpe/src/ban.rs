//! PLAN.md 3.4 Ban 1, with the 8 Nov 2026 human reading.
//!
//! `merge_allowed` is the character-integrity predicate only. The mark-initial
//! ban is separate. The 256×256 table is this predicate on every single-byte pair.

/// Character integrity (Ban 1).
///
/// Allowed when both sides are complete valid UTF-8, or when `M = L + R` lies
/// inside one character's UTF-8 encoding: valid lead, valid continuations for
/// each later position, and `len(M)` no greater than that lead's width.
/// Proper prefixes such as `E6 97` are allowed.
pub fn merge_allowed(left: &[u8], right: &[u8]) -> bool {
    if left.is_empty() || right.is_empty() {
        return false;
    }
    if is_complete_utf8(left) && is_complete_utf8(right) {
        return true;
    }
    let mut merged = Vec::with_capacity(left.len() + right.len());
    merged.extend_from_slice(left);
    merged.extend_from_slice(right);
    within_one_character(&merged)
}

pub fn is_complete_utf8(bytes: &[u8]) -> bool {
    std::str::from_utf8(bytes).is_ok()
}

/// `M` is a proper or complete prefix of one UTF-8 character, starting at its lead.
pub fn within_one_character(bytes: &[u8]) -> bool {
    if bytes.is_empty() || bytes.len() > 4 {
        return false;
    }
    let Some(width) = lead_width(bytes[0]) else {
        return false;
    };
    if bytes.len() > width {
        return false;
    }
    for (index, &byte) in bytes.iter().enumerate().skip(1) {
        if !valid_continuation(bytes[0], index, byte) {
            return false;
        }
    }
    true
}

fn lead_width(lead: u8) -> Option<usize> {
    match lead {
        0x00..=0x7F => Some(1),
        0xC2..=0xDF => Some(2),
        0xE0..=0xEF => Some(3),
        0xF0..=0xF4 => Some(4),
        _ => None,
    }
}

/// `index` is the 0-based position of `byte` inside the sequence (`1` is the second byte).
fn valid_continuation(lead: u8, index: usize, byte: u8) -> bool {
    let (lo, hi) = match (lead, index) {
        (0xE0, 1) => (0xA0, 0xBF),
        (0xED, 1) => (0x80, 0x9F),
        (0xF0, 1) => (0x90, 0xBF),
        (0xF4, 1) => (0x80, 0x8F),
        (_, 1..=3) => (0x80, 0xBF),
        _ => return false,
    };
    (lo..=hi).contains(&byte)
}

/// Ban 1 for every ordered pair of single bytes. Index is `(left << 8) | right`.
pub struct BytePairTable {
    allowed: [bool; 65536],
}

impl BytePairTable {
    pub fn build() -> Self {
        let mut allowed = [false; 65536];
        for left in 0..256u16 {
            for right in 0..256u16 {
                let ok = merge_allowed(&[left as u8], &[right as u8]);
                allowed[((left << 8) | right) as usize] = ok;
            }
        }
        Self { allowed }
    }

    pub fn allows(&self, left: u8, right: u8) -> bool {
        self.allowed[((left as usize) << 8) | right as usize]
    }
}

pub fn byte_pair_allowed(table: &BytePairTable, left: u8, right: u8) -> bool {
    table.allows(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force reference. Validity of a finished sequence is `from_utf8`,
    /// not the production continuation table. A proper prefix is allowed when
    /// at least one completion is one scalar — not when exactly one scalar
    /// starts with `M`.
    fn reference_merge_allowed(left: &[u8], right: &[u8]) -> bool {
        if left.is_empty() || right.is_empty() {
            return false;
        }
        if std::str::from_utf8(left).is_ok() && std::str::from_utf8(right).is_ok() {
            return true;
        }
        let mut merged = Vec::with_capacity(left.len() + right.len());
        merged.extend_from_slice(left);
        merged.extend_from_slice(right);
        reference_within_one(&merged)
    }

    fn reference_within_one(bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }
        let width = match bytes[0] {
            0x00..=0x7F => 1usize,
            0xC2..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF4 => 4,
            _ => return false,
        };
        if bytes.len() > width {
            return false;
        }
        let mut buf = bytes.to_vec();
        reference_search(&mut buf, width - bytes.len())
    }

    fn reference_search(buf: &mut Vec<u8>, missing: usize) -> bool {
        if missing == 0 {
            return std::str::from_utf8(buf)
                .map(|text| text.chars().count() == 1)
                .unwrap_or(false);
        }
        for byte in 0x80u8..=0xBF {
            buf.push(byte);
            if reference_search(buf, missing - 1) {
                buf.pop();
                return true;
            }
            buf.pop();
        }
        false
    }

    #[test]
    fn human_cases() {
        assert!(merge_allowed(&[0xE6], &[0x97]), "E6+97");
        assert!(merge_allowed(&[0xE6, 0x97], &[0xA5]), "E697+A5");
        assert!(!merge_allowed(&[0xA5], &[0x20]), "A5+20");
        assert!(!merge_allowed(&[0xE6, 0x97], &[0x41]), "E697+41");
        assert!(merge_allowed(&[0xF0], &[0x9F]), "F0+9F");
        assert!(merge_allowed(b"ab", b"cd"), "complete+complete ascii");
        assert!(
            merge_allowed(&[0xE6, 0x97, 0xA5], b"A"),
            "complete+complete multibyte"
        );
        for (left, right) in [
            (&[0xE6][..], &[0x97][..]),
            (&[0xE6, 0x97][..], &[0xA5][..]),
            (&[0xA5][..], &[0x20][..]),
            (&[0xE6, 0x97][..], &[0x41][..]),
            (&[0xF0][..], &[0x9F][..]),
            (b"ab".as_slice(), b"cd".as_slice()),
            (&[0xE6, 0x97, 0xA5][..], b"A".as_slice()),
        ] {
            assert_eq!(
                merge_allowed(left, right),
                reference_merge_allowed(left, right),
                "reference mismatch {left:02x?} {right:02x?}"
            );
        }
    }

    #[test]
    fn block0_prefixes_and_overlongs() {
        assert!(merge_allowed(&[0xE0], &[0xA4]));
        assert!(merge_allowed(&[0xE0], &[0xA5]));
        assert!(merge_allowed(&[0xE2], &[0x80]));
        assert!(merge_allowed(&[0xE0, 0xA4], &[0x80]));
        assert!(!merge_allowed(&[0xE0], &[0x80]), "overlong");
        assert!(!merge_allowed(&[0xED], &[0xA0]), "surrogate");
        assert!(!merge_allowed(&[0xF0], &[0x80]), "overlong 4-byte");
        assert!(!merge_allowed(&[0xF4], &[0x90]), "above U+10FFFF");
        assert!(!merge_allowed(&[0xC0], &[0x80]), "invalid lead");
        assert!(!merge_allowed(&[0x41], &[0x80]), "ascii plus continuation");
    }

    #[test]
    fn byte_pair_table_matches_reference_and_predicate() {
        let table = BytePairTable::build();
        for left in 0..256u16 {
            for right in 0..256u16 {
                let l = left as u8;
                let r = right as u8;
                let direct = merge_allowed(&[l], &[r]);
                let reference = reference_merge_allowed(&[l], &[r]);
                assert_eq!(direct, reference, "reference {l:02x} {r:02x}");
                assert_eq!(
                    byte_pair_allowed(&table, l, r),
                    direct,
                    "table {l:02x} {r:02x}"
                );
            }
        }
        assert!(table.allows(0xE6, 0x97));
        assert!(!table.allows(0xA5, 0x20));
        assert!(table.allows(0xF0, 0x9F));
    }

    #[test]
    fn block0_alphabet_strings_match_reference() {
        let mut alphabet = Vec::new();
        for byte in 0x80u8..=0xBF {
            alphabet.push(byte);
        }
        alphabet.push(0xE0);
        alphabet.push(0xE2);
        let mut buf = [0u8; 4];
        for length in 2..=4 {
            fill_and_check(&alphabet, &mut buf, 0, length);
        }
    }

    fn fill_and_check(alphabet: &[u8], buf: &mut [u8; 4], filled: usize, length: usize) {
        if filled == length {
            let bytes = &buf[..length];
            for cut in 1..length {
                let left = &bytes[..cut];
                let right = &bytes[cut..];
                assert_eq!(
                    merge_allowed(left, right),
                    reference_merge_allowed(left, right),
                    "block0 bytes {bytes:02x?} cut {cut}"
                );
            }
            return;
        }
        for &byte in alphabet {
            buf[filled] = byte;
            fill_and_check(alphabet, buf, filled + 1, length);
        }
    }
}

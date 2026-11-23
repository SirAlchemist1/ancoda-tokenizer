//! Lowest-rank concatenation (docs/SPEC.md 3.5). Rank equals token id.

use ahash::AHashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct Encoder {
    index: AHashMap<Vec<u8>, u32>,
    pair_map: AHashMap<(u32, u32), u32>,
    prefix: AHashMap<[u8; 2], u32>,
    char3: AHashMap<[u8; 3], u32>,
    block0: bool,
}

impl Encoder {
    pub fn new(tokens: &[Arc<Vec<u8>>], block0_end: u32) -> Self {
        let mut index = AHashMap::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            index.insert(token.as_ref().clone(), id as u32);
        }
        let mut pair_map = AHashMap::new();
        for (id, token) in tokens.iter().enumerate() {
            if token.len() < 2 {
                continue;
            }
            for cut in 1..token.len() {
                let left = &token[..cut];
                let right = &token[cut..];
                if let (Some(&left_id), Some(&right_id)) = (index.get(left), index.get(right)) {
                    pair_map.insert((left_id, right_id), id as u32);
                }
            }
        }
        let mut prefix = AHashMap::new();
        let mut char3 = AHashMap::new();
        if block0_end > 256 {
            for (id, token) in tokens
                .iter()
                .enumerate()
                .take(block0_end as usize)
                .skip(256)
            {
                if token.len() == 2 {
                    prefix.insert([token[0], token[1]], id as u32);
                } else if token.len() == 3 {
                    char3.insert([token[0], token[1], token[2]], id as u32);
                }
            }
        }
        Self {
            index,
            pair_map,
            prefix,
            char3,
            block0: block0_end > 256,
        }
    }

    pub fn encode(&self, bytes: &[u8]) -> Vec<u32> {
        if let Some(&id) = self.index.get(bytes) {
            return vec![id];
        }
        let mut ids = if self.block0 {
            self.scan_block0(bytes)
        } else {
            self.byte_ids(bytes)
        };
        self.merge_lowest(&mut ids);
        ids
    }

    /// Reference encoder: start from byte tokens and use only the pair map.
    pub fn encode_generic(&self, bytes: &[u8]) -> Vec<u32> {
        if let Some(&id) = self.index.get(bytes) {
            return vec![id];
        }
        let mut ids = self.byte_ids(bytes);
        self.merge_lowest(&mut ids);
        ids
    }

    fn byte_id(&self, byte: u8) -> u32 {
        self.index
            .get(&[byte][..])
            .copied()
            .unwrap_or(u32::from(byte))
    }

    fn byte_ids(&self, bytes: &[u8]) -> Vec<u32> {
        bytes
            .iter()
            .copied()
            .map(|byte| self.byte_id(byte))
            .collect()
    }

    fn scan_block0(&self, bytes: &[u8]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if i + 1 < bytes.len() {
                let key = [bytes[i], bytes[i + 1]];
                if let Some(&prefix_id) = self.prefix.get(&key) {
                    if i + 2 < bytes.len() {
                        let three = [bytes[i], bytes[i + 1], bytes[i + 2]];
                        if let Some(&char_id) = self.char3.get(&three) {
                            ids.push(char_id);
                            i += 3;
                            continue;
                        }
                    }
                    ids.push(prefix_id);
                    i += 2;
                    continue;
                }
            }
            ids.push(self.byte_id(bytes[i]));
            i += 1;
        }
        ids
    }

    fn merge_lowest(&self, ids: &mut Vec<u32>) {
        while ids.len() >= 2 {
            let mut best_index = None;
            let mut best_rank = u32::MAX;
            for index in 0..ids.len() - 1 {
                if let Some(&rank) = self.pair_map.get(&(ids[index], ids[index + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_index = Some(index);
                    }
                }
            }
            let Some(index) = best_index else {
                break;
            };
            ids[index] = best_rank;
            ids.remove(index + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arc_tokens(raw: &[&[u8]]) -> Vec<Arc<Vec<u8>>> {
        raw.iter().map(|token| Arc::new(token.to_vec())).collect()
    }

    #[test]
    fn toy_abcd_merges_lowest_rank_then_leftmost() {
        // ids: a b c d bc ab abc, but the generic map uses these ids, not byte values.
        let tokens = arc_tokens(&[b"a", b"b", b"c", b"d", b"bc", b"ab", b"abc"]);
        let encoder = Encoder::new(&tokens, 0);
        assert_eq!(encoder.encode_generic(b"abcd"), vec![6, 3]);
        assert_eq!(encoder.encode_generic(b"bcbc"), vec![4, 4]);
    }

    #[test]
    fn fast_block0_matches_generic() {
        let mut raw: Vec<Vec<u8>> = (0..=255u8).map(|byte| vec![byte]).collect();
        raw.push(vec![0xE0, 0xA4]);
        raw.push(vec![0xE0, 0xA5]);
        raw.push(vec![0xE2, 0x80]);
        for cp in 0x0900u32..=0x097F {
            let ch = char::from_u32(cp).unwrap();
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            raw.push(encoded.as_bytes().to_vec());
        }
        for cp in [0x200Cu32, 0x200D] {
            let ch = char::from_u32(cp).unwrap();
            let mut buf = [0u8; 4];
            raw.push(ch.encode_utf8(&mut buf).as_bytes().to_vec());
        }
        // का as a Block 1 token, after the Block 0 characters.
        raw.push("का".as_bytes().to_vec());
        let tokens: Vec<Arc<Vec<u8>>> = raw.into_iter().map(Arc::new).collect();
        let block0_end = 256 + 133;
        let encoder = Encoder::new(&tokens, block0_end as u32);
        let samples: Vec<&[u8]> = vec![
            "क".as_bytes(),
            "का".as_bytes(),
            "काठ".as_bytes(),
            b"abc",
            &[0xE0, 0xA4],
            &[0xE0, 0x80],
            &[0xE2, 0x80, 0x8C],
            &[0xE2, 0x80, 0x8E],
            "hello का".as_bytes(),
        ];
        for sample in samples {
            assert_eq!(
                encoder.encode(sample),
                encoder.encode_generic(sample),
                "mismatch {sample:02x?}"
            );
        }
        assert_eq!(encoder.encode("क".as_bytes()).len(), 1);
        assert_eq!(encoder.encode("का".as_bytes()).len(), 1);
    }
}

//! Block 2 trainer. Tie-break: count descending, then left bytes, then right bytes ascending.
//! A concatenation that is already a token is resegmented and does not take a new id.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::Path;
use std::sync::Arc;

use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;
use serde::Serialize;

use crate::encode::Encoder;
use crate::marks::{block2_merge_allowed, starts_with_forbidden_mark, MarkSet};

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub vocab_size: usize,
    pub plain: bool,
    pub block0: bool,
    pub block1: bool,
    pub bans: bool,
    pub block1_tokens: Vec<String>,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Layout {
    pub byte_end: u32,
    pub block0_end: u32,
    pub block1_end: u32,
    pub reserved_start: u32,
    pub vocab_size: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct MergeRow {
    pub left: u32,
    pub right: u32,
    pub id: u32,
    pub count: u64,
}

#[derive(Clone, Debug)]
pub struct TrainedModel {
    pub tokens: Vec<Vec<u8>>,
    pub specials: Vec<(String, u32)>,
    pub merges: Vec<MergeRow>,
    pub layout: Layout,
    pub learned: u64,
    pub duplicates: u64,
    pub banned: u64,
    pub unique_pretokens: u64,
    pub kept_words: u64,
    pub initial_symbols: u64,
    pub initial_pair_entries: u64,
    pub peak_rss_bytes: u64,
    pub stopped: &'static str,
    pub seed: u64,
}

pub fn special_token_names() -> Vec<String> {
    let named = [
        "<|bos|>",
        "<|eos|>",
        "<|pad|>",
        "<|start_turn|>",
        "<|end_turn|>",
        "<|think|>",
        "<|/think|>",
        "<|tool_call|>",
        "<|/tool_call|>",
        "<|tool_result|>",
        "<|/tool_result|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|mask|>",
        "<|sep|>",
    ];
    let mut names: Vec<String> = named.iter().map(|name| (*name).to_string()).collect();
    for index in 16..1024 {
        names.push(format!("<|reserved_{index}|>"));
    }
    names
}

pub fn train_from_counts(
    mut entries: Vec<(Vec<u8>, u64)>,
    config: &TrainConfig,
    unicode_data: &Path,
) -> Result<TrainedModel, String> {
    if config.vocab_size < 256 {
        return Err("vocab_size must be at least 256".into());
    }
    let (block0, block1, bans) = if config.plain {
        (false, false, false)
    } else {
        (config.block0, config.block1, config.bans)
    };
    if block1 && !block0 {
        return Err("block1 requires block0".into());
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    for pair in entries.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err("duplicate pre-token in count table".into());
        }
    }
    let unique_pretokens = entries.len() as u64;

    let mut tokens: Vec<Arc<Vec<u8>>> = Vec::new();
    let mut index: AHashMap<Vec<u8>, u32> = AHashMap::new();
    for byte in 0..256u16 {
        push_token(&mut tokens, &mut index, vec![byte as u8])?;
    }
    if block0 {
        push_token(&mut tokens, &mut index, vec![0xE0, 0xA4])?;
        push_token(&mut tokens, &mut index, vec![0xE0, 0xA5])?;
        push_token(&mut tokens, &mut index, vec![0xE2, 0x80])?;
        for cp in 0x0900u32..=0x097F {
            let ch = char::from_u32(cp).ok_or("block0 code point")?;
            let mut buf = [0u8; 4];
            push_token(
                &mut tokens,
                &mut index,
                ch.encode_utf8(&mut buf).as_bytes().to_vec(),
            )?;
        }
        for cp in [0x200Cu32, 0x200D] {
            let ch = char::from_u32(cp).ok_or("joiner")?;
            let mut buf = [0u8; 4];
            push_token(
                &mut tokens,
                &mut index,
                ch.encode_utf8(&mut buf).as_bytes().to_vec(),
            )?;
        }
        if tokens.len() != 256 + 133 {
            return Err(format!("Block 0 size {} is not 133", tokens.len() - 256));
        }
    }
    let block0_end = tokens.len() as u32;
    if block1 {
        for token in &config.block1_tokens {
            if token.is_empty() {
                return Err("empty Block 1 token".into());
            }
            push_token(&mut tokens, &mut index, token.as_bytes().to_vec())?;
        }
    }
    let block1_end = tokens.len() as u32;
    let reserved = if config.plain { 0 } else { 1024 };
    if config.vocab_size < tokens.len() + reserved {
        return Err(format!(
            "vocab_size {} is smaller than base {} plus reserved {}",
            config.vocab_size,
            tokens.len(),
            reserved
        ));
    }
    let reserved_start = config.vocab_size - reserved;
    let target_len = reserved_start;
    let marks = if bans {
        MarkSet::load(unicode_data)?
    } else {
        MarkSet::empty()
    };

    let encoder = Encoder::new(&tokens, block0_end);
    let mut words: Vec<Vec<u32>> = Vec::new();
    let mut counts: Vec<u64> = Vec::new();
    let mut initial_symbols = 0u64;
    for (bytes, count) in &entries {
        let ids = encoder.encode(bytes);
        initial_symbols = initial_symbols.saturating_add(ids.len() as u64);
        if ids.len() >= 2 {
            words.push(ids);
            counts.push(*count);
        }
    }
    drop(entries);
    let kept_words = words.len() as u64;

    let mut pair_count: AHashMap<(u32, u32), u64> = AHashMap::new();
    let mut pair_words: AHashMap<(u32, u32), AHashSet<u32>> = AHashMap::new();
    for (wi, word) in words.iter().enumerate() {
        let count = counts[wi];
        let mut seen = AHashSet::new();
        for window in word.windows(2) {
            let pair = (window[0], window[1]);
            let slot = pair_count.entry(pair).or_insert(0);
            *slot = slot.checked_add(count).ok_or("pair count overflow")?;
            if seen.insert(pair) {
                pair_words.entry(pair).or_default().insert(wi as u32);
            }
        }
    }
    let initial_pair_entries = pair_count.len() as u64;

    let mut heap = BinaryHeap::new();
    for (&pair, &count) in &pair_count {
        if count > 0 {
            heap.push(make_item(&tokens, pair, count));
        }
    }

    let mut banned: AHashSet<(u32, u32)> = AHashSet::new();
    let mut merges = Vec::new();
    let mut learned = 0u64;
    let mut duplicates = 0u64;
    let mut banned_n = 0u64;
    let mut steps = 0u64;
    let step_cap = (target_len as u64)
        .saturating_mul(50)
        .saturating_add(1_000_000);
    let mut stopped = "exhausted";

    while tokens.len() < target_len {
        steps += 1;
        if steps > step_cap {
            return Err("trainer exceeded its step cap".into());
        }
        if heap.len() > pair_count.len().saturating_mul(4).saturating_add(1024) {
            heap = rebuild_heap(&tokens, &pair_count, &banned);
        }
        let Some((left, right, count)) = pop_best(&mut heap, &pair_count, &banned) else {
            break;
        };
        let left_bytes = tokens[left as usize].as_ref().clone();
        let right_bytes = tokens[right as usize].as_ref().clone();
        if bans && !block2_merge_allowed(&left_bytes, &right_bytes, &marks) {
            if banned.insert((left, right)) {
                banned_n += 1;
            }
            continue;
        }
        let mut merged = Vec::with_capacity(left_bytes.len() + right_bytes.len());
        merged.extend_from_slice(&left_bytes);
        merged.extend_from_slice(&right_bytes);
        let new_id = if let Some(&id) = index.get(&merged) {
            duplicates += 1;
            id
        } else {
            let id = tokens.len() as u32;
            index.insert(merged.clone(), id);
            tokens.push(Arc::new(merged));
            learned += 1;
            merges.push(MergeRow {
                left,
                right,
                id,
                count,
            });
            if learned.is_multiple_of(1000) {
                eprintln!(
                    "merge {learned} id={id} count={count} words={kept_words} pairs={} rss={}",
                    pair_count.len(),
                    peak_rss_bytes()
                );
            }
            id
        };
        let word_ids = word_ids_for(&pair_words, (left, right));
        apply_merge(
            &mut words,
            &counts,
            &mut pair_count,
            &mut pair_words,
            &mut heap,
            &tokens,
            &banned,
            &word_ids,
            left,
            right,
            new_id,
        )?;
        if pair_count.get(&(left, right)).copied().unwrap_or(0) != 0 {
            return Err(format!(
                "merged pair ({left}, {right}) still has count {}",
                pair_count[&(left, right)]
            ));
        }
        if tokens.len() >= target_len {
            stopped = "filled";
            break;
        }
    }
    if tokens.len() >= target_len {
        stopped = "filled";
    }

    if bans {
        for token in tokens.iter().skip(block0_end as usize) {
            if starts_with_forbidden_mark(token, &marks) {
                return Err(format!("mark-initial token survived bans: {}", hex(token)));
            }
        }
    }

    let specials = if config.plain {
        Vec::new()
    } else {
        special_token_names()
            .into_iter()
            .enumerate()
            .map(|(offset, name)| (name, reserved_start as u32 + offset as u32))
            .collect()
    };
    if !config.plain && specials.len() != 1024 {
        return Err(format!("special token count {}", specials.len()));
    }

    let owned: Vec<Vec<u8>> = tokens.iter().map(|token| token.as_ref().clone()).collect();
    Ok(TrainedModel {
        tokens: owned,
        specials,
        merges,
        layout: Layout {
            byte_end: 256,
            block0_end,
            block1_end,
            reserved_start: reserved_start as u32,
            vocab_size: config.vocab_size as u32,
        },
        learned,
        duplicates,
        banned: banned_n,
        unique_pretokens,
        kept_words,
        initial_symbols,
        initial_pair_entries,
        peak_rss_bytes: peak_rss_bytes(),
        stopped,
        seed: config.seed,
    })
}

fn word_ids_for(pair_words: &AHashMap<(u32, u32), AHashSet<u32>>, pair: (u32, u32)) -> Vec<u32> {
    let mut ids: Vec<u32> = pair_words
        .get(&pair)
        .map(|set| set.iter().copied().collect())
        .unwrap_or_default();
    ids.sort_unstable();
    ids
}

fn push_token(
    tokens: &mut Vec<Arc<Vec<u8>>>,
    index: &mut AHashMap<Vec<u8>, u32>,
    bytes: Vec<u8>,
) -> Result<(), String> {
    if index.contains_key(&bytes) {
        return Err(format!("duplicate vocabulary bytes {}", hex(&bytes)));
    }
    let id = tokens.len() as u32;
    index.insert(bytes.clone(), id);
    tokens.push(Arc::new(bytes));
    Ok(())
}

fn make_item(tokens: &[Arc<Vec<u8>>], pair: (u32, u32), count: u64) -> Item {
    Item {
        count,
        left_id: pair.0,
        right_id: pair.1,
        left: Arc::clone(&tokens[pair.0 as usize]),
        right: Arc::clone(&tokens[pair.1 as usize]),
    }
}

fn rebuild_heap(
    tokens: &[Arc<Vec<u8>>],
    pair_count: &AHashMap<(u32, u32), u64>,
    banned: &AHashSet<(u32, u32)>,
) -> BinaryHeap<Item> {
    let mut heap = BinaryHeap::new();
    for (&pair, &count) in pair_count {
        if count > 0 && !banned.contains(&pair) {
            heap.push(make_item(tokens, pair, count));
        }
    }
    heap
}

fn pop_best(
    heap: &mut BinaryHeap<Item>,
    pair_count: &AHashMap<(u32, u32), u64>,
    banned: &AHashSet<(u32, u32)>,
) -> Option<(u32, u32, u64)> {
    while let Some(item) = heap.pop() {
        let pair = (item.left_id, item.right_id);
        if banned.contains(&pair) {
            continue;
        }
        let live = pair_count.get(&pair).copied().unwrap_or(0);
        if live == 0 || live != item.count {
            continue;
        }
        return Some((pair.0, pair.1, live));
    }
    None
}

type PairDelta = ((u32, u32), i64);
type WordRewrite = (u32, Vec<u32>, Vec<PairDelta>);

#[allow(clippy::too_many_arguments)]
fn apply_merge(
    words: &mut [Vec<u32>],
    counts: &[u64],
    pair_count: &mut AHashMap<(u32, u32), u64>,
    pair_words: &mut AHashMap<(u32, u32), AHashSet<u32>>,
    heap: &mut BinaryHeap<Item>,
    tokens: &[Arc<Vec<u8>>],
    banned: &AHashSet<(u32, u32)>,
    word_ids: &[u32],
    left: u32,
    right: u32,
    new_id: u32,
) -> Result<(), String> {
    pair_words.remove(&(left, right));
    let computed: Vec<WordRewrite> = if word_ids.len() >= 4096 {
        word_ids
            .par_iter()
            .map(|&wi| {
                let (new_word, delta) = rewrite(&words[wi as usize], left, right, new_id);
                (wi, new_word, delta)
            })
            .collect()
    } else {
        word_ids
            .iter()
            .map(|&wi| {
                let (new_word, delta) = rewrite(&words[wi as usize], left, right, new_id);
                (wi, new_word, delta)
            })
            .collect()
    };
    for (wi, new_word, delta) in computed {
        let wcount = counts[wi as usize];
        words[wi as usize] = new_word;
        for (pair, occ_delta) in delta {
            if occ_delta == 0 {
                continue;
            }
            let count = add_count(pair_count, pair, occ_delta, wcount)?;
            if count == 0 {
                pair_words.remove(&pair);
                continue;
            }
            if occ_delta > 0 {
                pair_words.entry(pair).or_default().insert(wi);
            } else if !word_contains(&words[wi as usize], pair) {
                if let Some(set) = pair_words.get_mut(&pair) {
                    set.remove(&wi);
                    if set.is_empty() {
                        pair_words.remove(&pair);
                    }
                }
            }
            if !banned.contains(&pair) {
                heap.push(make_item(tokens, pair, count));
            }
        }
    }
    Ok(())
}

fn add_count(
    pair_count: &mut AHashMap<(u32, u32), u64>,
    pair: (u32, u32),
    occ_delta: i64,
    word_count: u64,
) -> Result<u64, String> {
    let scaled = (occ_delta as i128)
        .checked_mul(word_count as i128)
        .ok_or("pair count scale overflow")?;
    let slot = pair_count.entry(pair).or_insert(0);
    let next = (*slot as i128)
        .checked_add(scaled)
        .ok_or("pair count overflow")?;
    if next < 0 {
        return Err(format!("pair count underflow for {pair:?}"));
    }
    let count = next as u64;
    if count == 0 {
        pair_count.remove(&pair);
    } else {
        *slot = count;
    }
    Ok(count)
}

fn rewrite(word: &[u32], left: u32, right: u32, new_id: u32) -> (Vec<u32>, Vec<PairDelta>) {
    let mut out = Vec::with_capacity(word.len());
    let mut index = 0;
    while index < word.len() {
        if index + 1 < word.len() && word[index] == left && word[index + 1] == right {
            out.push(new_id);
            index += 2;
        } else {
            out.push(word[index]);
            index += 1;
        }
    }
    let delta = pair_delta(word, &out);
    (out, delta)
}

fn pair_delta(old: &[u32], new: &[u32]) -> Vec<PairDelta> {
    let mut map: AHashMap<(u32, u32), i64> = AHashMap::new();
    for window in old.windows(2) {
        *map.entry((window[0], window[1])).or_default() -= 1;
    }
    for window in new.windows(2) {
        *map.entry((window[0], window[1])).or_default() += 1;
    }
    map.into_iter().filter(|(_, delta)| *delta != 0).collect()
}

fn word_contains(word: &[u32], pair: (u32, u32)) -> bool {
    word.windows(2)
        .any(|window| window[0] == pair.0 && window[1] == pair.1)
}

pub fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return 0;
    }
    let usage = unsafe { usage.assume_init() };
    usage.ru_maxrss.max(0) as u64
}

pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

struct Item {
    count: u64,
    left_id: u32,
    right_id: u32,
    left: Arc<Vec<u8>>,
    right: Arc<Vec<u8>>,
}

impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.left.as_slice().cmp(self.left.as_slice()))
            .then_with(|| other.right.as_slice().cmp(self.right.as_slice()))
            .then_with(|| other.left_id.cmp(&self.left_id))
            .then_with(|| other.right_id.cmp(&self.right_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unicode_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../third_party/unicode/16.0.0/UnicodeData.txt")
    }

    fn plain(vocab_size: usize) -> TrainConfig {
        TrainConfig {
            vocab_size,
            plain: true,
            block0: false,
            block1: false,
            bans: false,
            block1_tokens: Vec::new(),
            seed: 20261104,
        }
    }

    #[test]
    fn higher_count_wins_and_equal_count_picks_smaller_bytes() {
        let model = train_from_counts(
            vec![(b"ab".to_vec(), 4), (b"cd".to_vec(), 4)],
            &plain(258),
            &unicode_path(),
        )
        .unwrap();
        assert_eq!(model.merges.len(), 2);
        assert_eq!(model.tokens[256], b"ab");
        assert_eq!(model.tokens[257], b"cd");
        assert_eq!(model.merges[0].count, 4);
        assert_eq!(model.stopped, "filled");
    }

    #[test]
    fn count_order_beats_byte_order() {
        let model = train_from_counts(
            vec![(b"ab".to_vec(), 5), (b"cd".to_vec(), 3)],
            &plain(258),
            &unicode_path(),
        )
        .unwrap();
        assert_eq!(model.tokens[256], b"ab");
        assert_eq!(model.tokens[257], b"cd");
        assert_eq!(model.merges[0].count, 5);
        assert_eq!(model.merges[1].count, 3);
    }

    #[test]
    fn bans_allow_character_prefix_and_reject_mark_and_continuation() {
        let config = TrainConfig {
            vocab_size: 256 + 1024 + 3,
            plain: false,
            block0: false,
            block1: false,
            bans: true,
            block1_tokens: Vec::new(),
            seed: 20261104,
        };
        // क (Lo) may be assembled. ा (Mc) may not. A5 20 is continuation-initial.
        let model = train_from_counts(
            vec![
                ("क".as_bytes().to_vec(), 10),
                ("ा".as_bytes().to_vec(), 30),
                (vec![0xA5, 0x20], 50),
            ],
            &config,
            &unicode_path(),
        )
        .unwrap();
        assert!(
            model.tokens.iter().any(|token| token == "क".as_bytes()),
            "क should be learned"
        );
        assert!(
            model
                .tokens
                .iter()
                .all(|token| token.as_slice() != "ा".as_bytes()),
            "vowel sign must not be a learned token"
        );
        assert!(model
            .tokens
            .iter()
            .all(|token| token.as_slice() != [0xA5, 0x20]));
        assert!(model.banned > 0);
    }

    #[test]
    fn special_names_are_1024_and_start_with_bos() {
        let names = special_token_names();
        assert_eq!(names.len(), 1024);
        assert_eq!(names[0], "<|bos|>");
        assert_eq!(names[15], "<|sep|>");
        assert_eq!(names[16], "<|reserved_16|>");
        assert_eq!(names[1023], "<|reserved_1023|>");
    }
}

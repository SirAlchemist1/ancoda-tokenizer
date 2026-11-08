//! Ancoda BPE trainer library. The canonical encoder, Ban 1, and Block 2 live here.

mod ban;
mod countsio;
mod encode;
mod marks;
mod train;

pub use ban::{byte_pair_allowed, merge_allowed, BytePairTable};
pub use countsio::{read_counts, write_counts, CountTable};
pub use encode::Encoder;
pub use marks::{block2_merge_allowed, starts_with_forbidden_mark, MarkSet};
pub use train::{
    hex, peak_rss_bytes, special_token_names, train_from_counts, Layout, MergeRow, TrainConfig,
    TrainedModel,
};

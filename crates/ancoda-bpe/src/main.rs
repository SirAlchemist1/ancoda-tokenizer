//! Ancoda BPE trainer.
//!
//! `ancoda-bpe train --config <yaml>` reads a sorted count table and writes
//! `ranks.json` plus `manifest.json`. `ancoda-bpe specials` prints the 1,024
//! special-token names, one per line.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use ancoda_bpe::{read_counts, special_token_names, train_from_counts, TrainConfig, TrainedModel};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Parser, Debug)]
#[command(name = "ancoda-bpe", about = "Ancoda BPE trainer")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Learn a vocabulary from a pre-token count table.
    Train {
        #[arg(long)]
        config: PathBuf,
    },
    /// Print the 1,024 special-token names.
    Specials,
    /// Read a count table and print its size.
    CheckCounts { path: PathBuf },
}

fn main() {
    let cli = Cli::parse();
    let code = match cli.command {
        Cmd::Specials => {
            for name in special_token_names() {
                println!("{name}");
            }
            0
        }
        Cmd::CheckCounts { path } => match check_counts(&path) {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("error: {err}");
                1
            }
        },
        Cmd::Train { config } => match train_main(&config) {
            Ok(code) => code,
            Err(err) => {
                eprintln!("error: {err}");
                1
            }
        },
    };
    std::process::exit(code);
}

fn check_counts(path: &Path) -> Result<(), String> {
    let table = read_counts(path)?;
    println!(
        "entries={} occurrences={} key_bytes={}",
        table.entries.len(),
        table.occurrences,
        table.key_bytes
    );
    Ok(())
}

fn train_main(config_path: &Path) -> Result<i32, String> {
    let started = now_utc();
    let started_unix = unix_now();
    let config_text = fs::read_to_string(config_path)
        .map_err(|err| format!("read {}: {err}", config_path.display()))?;
    let config_sha = sha256_bytes(config_text.as_bytes());
    let raw = parse_yaml(&config_text)?;
    let cfg = FileConfig::from_map(&raw)?;
    let block1_tokens = if cfg.block1 {
        let file = read_covered(&cfg.block1_path)?;
        if file.gb9c != cfg.gb9c {
            return Err(format!(
                "covered file gb9c={} but config gb9c={}",
                file.gb9c, cfg.gb9c
            ));
        }
        file.block1_merges
    } else {
        Vec::new()
    };
    eprintln!(
        "reading counts {} (block0={} block1={} bans={} plain={} vocab={})",
        cfg.counts.display(),
        cfg.block0,
        cfg.block1,
        cfg.bans,
        cfg.plain,
        cfg.vocab_size
    );
    let table = read_counts(&cfg.counts)?;
    eprintln!(
        "counts entries={} occurrences={} key_bytes={}",
        table.entries.len(),
        table.occurrences,
        table.key_bytes
    );
    let train_cfg = TrainConfig {
        vocab_size: cfg.vocab_size,
        plain: cfg.plain,
        block0: cfg.block0,
        block1: cfg.block1,
        bans: cfg.bans,
        block1_tokens,
        seed: cfg.seed,
    };
    let model = train_from_counts(table.entries, &train_cfg, &cfg.unicode_data)?;
    fs::create_dir_all(&cfg.output_dir).map_err(|err| err.to_string())?;
    let ranks_path = cfg.output_dir.join("ranks.json");
    write_ranks(&ranks_path, &model, cfg.seed)?;
    let finished = now_utc();
    let manifest = json!({
        "started_at": started,
        "finished_at": finished,
        "elapsed_secs": unix_now().saturating_sub(started_unix),
        "seed": cfg.seed,
        "git_commit": git_output(&["rev-parse", "HEAD"]),
        "git_dirty": !git_output(&["status", "--porcelain"]).is_empty(),
        "rustc": command_output("rustc", &["--version"]),
        "unicode_segmentation": "1.12.0",
        "unicode": {
            "version": "16.0.0",
            "UnicodeData.txt": sha256_file(&cfg.unicode_data)?,
        },
        "config_sha256": config_sha,
        "input_sha256": {
            "counts": sha256_file(&cfg.counts)?,
            "config": config_sha,
            "block1": if cfg.block1 { Value::String(sha256_file(&cfg.block1_path)?) } else { Value::Null },
            "unicode_data": sha256_file(&cfg.unicode_data)?,
        },
        "counts": {
            "entries": model.unique_pretokens,
            "unique": model.unique_pretokens,
            "occurrences": table.occurrences,
            "key_bytes": table.key_bytes,
        },
        "stats": stats_json(&model),
        "output": {
            "ranks": "ranks.json",
            "ranks_sha256": sha256_file(&ranks_path)?,
        },
    });
    let manifest_path = cfg.output_dir.join("manifest.json");
    write_json(&manifest_path, &manifest)?;
    println!("peak_rss_bytes={}", model.peak_rss_bytes);
    println!("unique_pretokens={}", model.unique_pretokens);
    println!("initial_symbols={}", model.initial_symbols);
    println!("pair_entries={}", model.initial_pair_entries);
    println!("learned={}", model.learned);
    println!("duplicates={}", model.duplicates);
    println!("banned={}", model.banned);
    println!("stopped={}", model.stopped);
    println!("tokens={}", model.tokens.len());
    println!("elapsed_secs={}", unix_now().saturating_sub(started_unix));
    println!("wrote {}", ranks_path.display());
    if cfg.require_full && model.stopped != "filled" {
        eprintln!(
            "STOP: trainer stopped at {} tokens (learned {}), target vocab {}",
            model.tokens.len(),
            model.learned,
            cfg.vocab_size
        );
        return Ok(3);
    }
    Ok(0)
}

fn stats_json(model: &TrainedModel) -> Value {
    json!({
        "learned": model.learned,
        "duplicates": model.duplicates,
        "banned": model.banned,
        "unique_pretokens": model.unique_pretokens,
        "kept_words": model.kept_words,
        "initial_symbols": model.initial_symbols,
        "initial_pair_entries": model.initial_pair_entries,
        "peak_rss_bytes": model.peak_rss_bytes,
        "stopped": model.stopped,
        "layout": model.layout,
        "mergeable_tokens": model.tokens.len(),
        "specials": model.specials.len(),
    })
}

fn write_ranks(path: &Path, model: &TrainedModel, seed: u64) -> Result<(), String> {
    let tokens_hex: Vec<String> = model
        .tokens
        .iter()
        .map(|token| ancoda_bpe::hex(token))
        .collect();
    let specials: Vec<Value> = model
        .specials
        .iter()
        .map(|(name, id)| json!({"name": name, "id": id}))
        .collect();
    let value = json!({
        "version": 1,
        "seed": seed,
        "layout": model.layout,
        "tokens_hex": tokens_hex,
        "special_tokens": specials,
        "block2_merges": model.merges,
        "stats": stats_json(model),
    });
    write_json(path, &value)
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let file = fs::File::create(path).map_err(|err| format!("create {}: {err}", path.display()))?;
    serde_json::to_writer_pretty(file, value).map_err(|err| err.to_string())?;
    Ok(())
}

#[derive(Deserialize)]
struct CoveredFile {
    block1_merges: Vec<String>,
    gb9c: bool,
}

fn read_covered(path: &Path) -> Result<CoveredFile, String> {
    let file = fs::File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    serde_json::from_reader(file).map_err(|err| format!("parse {}: {err}", path.display()))
}

struct FileConfig {
    vocab_size: usize,
    plain: bool,
    block0: bool,
    block1: bool,
    bans: bool,
    gb9c: bool,
    seed: u64,
    counts: PathBuf,
    block1_path: PathBuf,
    unicode_data: PathBuf,
    output_dir: PathBuf,
    require_full: bool,
}

impl FileConfig {
    fn from_map(map: &BTreeMap<String, String>) -> Result<Self, String> {
        let require = |key: &str| -> Result<String, String> {
            map.get(key)
                .cloned()
                .ok_or_else(|| format!("config missing {key}"))
        };
        let flag = |key: &str| -> Result<bool, String> {
            match require(key)?.as_str() {
                "true" => Ok(true),
                "false" => Ok(false),
                other => Err(format!("{key} must be true or false, got {other}")),
            }
        };
        Ok(Self {
            vocab_size: require("vocab_size")?
                .parse()
                .map_err(|err| format!("vocab_size: {err}"))?,
            plain: flag("plain")?,
            block0: flag("block0")?,
            block1: flag("block1")?,
            bans: flag("bans")?,
            gb9c: flag("gb9c")?,
            seed: require("seed")?
                .parse()
                .map_err(|err| format!("seed: {err}"))?,
            counts: PathBuf::from(require("counts")?),
            block1_path: PathBuf::from(require("block1_path")?),
            unicode_data: PathBuf::from(require("unicode_data")?),
            output_dir: PathBuf::from(require("output_dir")?),
            require_full: flag("require_full")?,
        })
    }
}

fn parse_yaml(text: &str) -> Result<BTreeMap<String, String>, String> {
    let mut map = BTreeMap::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            return Err(format!("config line {}: {raw}", lineno + 1));
        };
        let value = value.trim().trim_matches('"').trim_matches('\'');
        map.insert(key.trim().to_string(), value.to_string());
    }
    Ok(map)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|err| format!("hash {}: {err}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_bytes(&digest)
}

fn hex_bytes(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

fn command_output(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn git_output(args: &[&str]) -> String {
    command_output("git", args)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn now_utc() -> String {
    command_output("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])
}

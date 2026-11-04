//! Ancoda BPE trainer CLI. Implementation lands in M6+.

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "ancoda-bpe", about = "Ancoda BPE trainer (stub until M6)")]
struct Cli {
    #[arg(long, default_value = "help")]
    cmd: String,
}

fn main() {
    let cli = Cli::parse();
    eprintln!("ancoda-bpe stub: cmd={} (trainer lands in M6)", cli.cmd);
}

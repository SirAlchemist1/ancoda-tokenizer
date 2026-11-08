"""CLI: ``uv run python -m ancoda_tok.clean <command>``."""

from __future__ import annotations

import argparse
import sys

from ancoda_tok.clean.common import Stop, free_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ancoda_tok.clean")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_lex = sub.add_parser("build-lexicon", help="Build Preeti dictionary lexicon")
    p_lex.add_argument("--force", action="store_true")

    p_clean = sub.add_parser("run", help="UTF-8 + LID + legacy over training sources")
    p_clean.add_argument("--force", action="store_true")
    p_clean.add_argument("--no-delete-raw", action="store_true")

    sub.add_parser("dedup", help="Exact then MinHash dedup on Nepali sources")
    sub.add_parser("decontam", help="Decontaminate clean training sources")
    sub.add_parser("snapshot", help="Build four snapshots and audit")
    sub.add_parser("spotcheck", help="Dictionary detector spot-check v2")
    sub.add_parser(
        "all",
        help="Lexicon → spotcheck → clean → dedup → decontam → snapshot",
    )

    args = parser.parse_args(argv)
    try:
        if args.cmd == "build-lexicon":
            from ancoda_tok.clean.legacy_font import build_lexicon, lexicon_path

            path = lexicon_path()
            if path.exists() and not args.force:
                print(f"exists {path}", flush=True)
            else:
                build_lexicon(out_path=path)
        elif args.cmd == "run":
            from ancoda_tok.clean.pipeline import run_clean_phase

            run_clean_phase(force=args.force, delete_raw=not args.no_delete_raw)
        elif args.cmd == "dedup":
            from ancoda_tok.clean.dedup import run_dedup

            run_dedup()
        elif args.cmd == "decontam":
            from ancoda_tok.clean.decontam import run_decontam

            run_decontam()
        elif args.cmd == "snapshot":
            from ancoda_tok.clean.snapshot import build_snapshots

            build_snapshots()
        elif args.cmd == "spotcheck":
            from ancoda_tok.clean.spotcheck import run_spotcheck_v2

            run_spotcheck_v2()
        elif args.cmd == "all":
            from ancoda_tok.clean.common import artifacts_clean, save_json
            from ancoda_tok.clean.decontam import run_decontam
            from ancoda_tok.clean.dedup import run_dedup
            from ancoda_tok.clean.legacy_font import build_lexicon, lexicon_path
            from ancoda_tok.clean.pipeline import run_clean_phase
            from ancoda_tok.clean.snapshot import build_snapshots
            from ancoda_tok.clean.spotcheck import run_spotcheck_v2
            from ancoda_tok.storage import REPO_ROOT

            path = lexicon_path()
            if not path.exists():
                build_lexicon(out_path=path)
            summary = run_spotcheck_v2()
            mix_path = REPO_ROOT / "configs" / "mix.yaml"
            decisions = REPO_ROOT / "docs" / "DECISIONS.md"
            text = mix_path.read_text(encoding="utf-8")
            if summary["precision"] < 0.99:
                text = text.replace("enabled: true", "enabled: false", 1)
                note = (
                    "\n\n### Legacy-font filter status (after dictionary spot-check)\n\n"
                    "legacy-font filter disabled: no detector reached 99% precision "
                    f"(dictionary detector precision={summary['precision']:.4f}; "
                    f"known true Preeti caught={summary['known_true_preeti_caught']}/5).\n"
                    "Would-have-flagged line counts are recorded per source in "
                    "`artifacts/clean/report.json` under `legacy_would_flag_lines`.\n"
                )
                print(
                    f"legacy filter DISABLED (precision={summary['precision']:.4f})",
                    flush=True,
                )
            else:
                if "enabled: false" in text:
                    text = text.replace("enabled: false", "enabled: true", 1)
                note = (
                    "\n\n### Legacy-font filter status (after dictionary spot-check)\n\n"
                    f"legacy-font dictionary detector ENABLED "
                    f"(precision={summary['precision']:.4f}; "
                    f"known true Preeti caught={summary['known_true_preeti_caught']}/5).\n"
                )
                print(
                    f"legacy filter ENABLED (precision={summary['precision']:.4f})",
                    flush=True,
                )
            mix_path.write_text(text, encoding="utf-8")
            decisions.write_text(
                decisions.read_text(encoding="utf-8") + note, encoding="utf-8"
            )
            save_json(artifacts_clean() / "preeti_spotcheck_v2_decision.json", summary)
            run_clean_phase(force=False, delete_raw=True)
            run_dedup()
            run_decontam()
            build_snapshots()
        else:
            parser.error(f"unknown command {args.cmd}")
    except Stop as exc:
        print(str(exc), file=sys.stderr, flush=True)
        print(f"free_bytes={free_bytes()}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

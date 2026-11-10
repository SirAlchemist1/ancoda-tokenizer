"""M9 command line. Heavy imports stay inside the subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="ancoda_tok.m9")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare")
    sub.add_parser("evalprep")

    counts = sub.add_parser("counts")
    counts.add_argument("--text", type=Path, required=True)
    counts.add_argument("--tokenizer", type=Path, required=True)
    counts.add_argument("--arm", required=True)
    counts.add_argument("--n", type=int, default=10000)
    counts.add_argument("--roundtrip", type=int, default=1000)
    counts.add_argument("--out", type=Path, required=True)
    counts.add_argument("--tokens", type=Path, default=None)

    tok = sub.add_parser("tokenize")
    tok.add_argument("--text", type=Path, required=True)
    tok.add_argument("--tokenizer", type=Path, required=True)
    tok.add_argument("--arm", required=True)
    tok.add_argument("--out", type=Path, required=True)
    tok.add_argument("--workers", type=int, default=6)

    train = sub.add_parser("train")
    train.add_argument("--tokens", type=Path, required=True)
    train.add_argument("--arm", required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--vocab", type=int, required=True)
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--micro-batch", type=int, default=4)
    train.add_argument("--checkpoint-every", type=int, default=500)
    train.add_argument("--bstar", type=int, default=None)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--compile", action="store_true")

    bpb = sub.add_parser("bpb")
    bpb.add_argument("--arm", required=True)
    bpb.add_argument("--tokenizer", type=Path, required=True)
    bpb.add_argument("--checkpoint", type=Path, required=True)
    bpb.add_argument("--eval-dir", type=Path, required=True)
    bpb.add_argument("--out", type=Path, required=True)

    params = sub.add_parser("params")
    params.add_argument("--vocab", type=int, required=True)
    params.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    if args.cmd == "prepare":
        from ancoda_tok.m9.stream import prepare

        prepare()
    elif args.cmd == "evalprep":
        from ancoda_tok.m9.evalprep import prepare_eval

        prepare_eval()
    elif args.cmd == "counts":
        from ancoda_tok.m9.tokenize import compare_bins, prefix_counts

        report = prefix_counts(args.text, args.tokenizer, args.arm, args.n, args.roundtrip)
        if args.tokens is not None:
            compare_bins(args.tokens, report)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.arm != "sidskarki-64k" and report["roundtrip_rate"] != 1.0:
            raise SystemExit(
                f"STOP: {args.arm} round-trip {report['roundtrip_ok']}/{report['roundtrip_n']}"
            )
    elif args.cmd == "tokenize":
        from ancoda_tok.m9.tokenize import tokenize

        tokenize(args)
    elif args.cmd == "train":
        from ancoda_tok.m9.train import train

        train(args)
    elif args.cmd == "bpb":
        from ancoda_tok.m9.bpb import evaluate

        evaluate(args)
    elif args.cmd == "params":
        from ancoda_tok.m9.model import (
            assert_formula,
            build_model,
            count_parameters,
            parameter_count,
        )

        print(f"formula {parameter_count(args.vocab)}")
        model = build_model(args.vocab, args.seed)
        assert_formula(model)
        print(f"constructed {count_parameters(model)}")
    else:
        raise SystemExit(f"unknown {args.cmd}")


if __name__ == "__main__":
    main()

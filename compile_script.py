#!/usr/bin/env python3
"""Compile the same main.cu with distinct, recorded 64-bit seeds."""

import argparse
import csv
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile


CARDS = {"rtx3090": "sm_86", "a40": "sm_86", "a100": "sm_80",
         "1g.10gb": "sm_80", "h200": "sm_90"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="?", type=int, default=20)
    parser.add_argument("offset", nargs="?", type=int, default=0)
    parser.add_argument("--card", type=str.lower, choices=CARDS, default="a100")
    args = parser.parse_args(argv)
    if args.runs <= 0 or args.offset < 0:
        parser.error("runs must be positive and offset must be non-negative")

    root = Path(__file__).resolve().parent
    arch = CARDS[args.card]
    manifest = root / f"build_seeds_{args.offset}_{args.offset + args.runs - 1}.csv"
    used_seeds = set()
    # Record only binaries successfully compiled in this invocation.
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["executable", "seed", "architecture"])
        stream.flush()
        for number in range(args.offset, args.offset + args.runs):
            seed = secrets.randbits(64)
            while seed in used_seeds:
                seed = secrets.randbits(64)
            used_seeds.add(seed)
            target = f"bosegascl{number}"
            print(f"[{number - args.offset + 1}/{args.runs}] {target}: seed={seed}",
                  flush=True)
            # A fresh output forces compilation, even if an old binary exists.
            # Replace that binary only after a successful build.
            with tempfile.TemporaryDirectory(prefix=".build-", dir=root) as temp:
                output = Path(temp) / target
                subprocess.run(
                    ["make", "--no-print-directory", "-f", "Makefile",
                     f"TARGET={output}", f"SEED={seed}", f"ARCH={arch}", "all"],
                    cwd=root, check=True,
                )
                output.replace(root / target)
            writer.writerow([target, seed, arch])
            stream.flush()
    print(f"Completed {args.runs} builds. Seeds: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        sys.exit(1)

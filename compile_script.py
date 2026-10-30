#!/usr/bin/env python3
"""Build seeded binaries for each value in an inclusive MU sweep."""

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile


CARDS = {"rtx3090": "sm_86", "a40": "sm_86", "a100": "sm_80",
         "1g.10gb": "sm_80", "h200": "sm_90"}
MU_PATTERN = re.compile(r"(?m)^(\s*#define\s+MU\s+)([^\s/]+)(.*)$")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu-min", type=Decimal, default=Decimal("1.5"))
    parser.add_argument("--mu-max", type=Decimal, default=Decimal("2.7"))
    parser.add_argument("--mu-steps", type=int, default=4,
                        help="number of evenly spaced MU values, including both endpoints")
    parser.add_argument("--runs-per-mu", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0,
                        help="first executable number (for SLURM array task IDs)")
    parser.add_argument("--card", type=str.lower, choices=CARDS, default="a100")
    args = parser.parse_args(argv)
    if args.mu_steps < 2 or args.runs_per_mu <= 0 or args.offset < 0:
        parser.error("mu-steps must be at least 2, runs-per-mu positive, and offset non-negative")
    if args.mu_max <= args.mu_min:
        parser.error("mu-max must be greater than mu-min")

    root = Path(__file__).resolve().parent
    header = root / "Parameters.h"
    original = header.read_text()
    matches = list(MU_PATTERN.finditer(original))
    if len(matches) != 1:
        parser.error(f"expected exactly one #define MU in {header}")

    arch = CARDS[args.card]
    total = args.mu_steps * args.runs_per_mu
    manifest = root / f"build_mu_{args.offset}_{args.offset + total - 1}.csv"
    used_seeds = set()
    step = (args.mu_max - args.mu_min) / (args.mu_steps - 1)
    try:
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["executable", "mu", "run", "seed", "architecture"])
            stream.flush()
            for mu_index in range(args.mu_steps):
                mu = args.mu_min + step * mu_index
                mu_text = format(mu.normalize(), "f")
                updated = MU_PATTERN.sub(lambda m: f"{m.group(1)}{mu_text}{m.group(3)}",
                                         original, count=1)
                header.write_text(updated)
                for run_index in range(args.runs_per_mu):
                    number = args.offset + mu_index * args.runs_per_mu + run_index
                    seed = secrets.randbits(64)
                    while seed in used_seeds:
                        seed = secrets.randbits(64)
                    used_seeds.add(seed)
                    target = f"bosegascl{number}"
                    print(f"[{number - args.offset + 1}/{total}] {target}: MU={mu_text}, "
                          f"run={run_index + 1}, seed={seed}", flush=True)
                    with tempfile.TemporaryDirectory(prefix=".build-", dir=root) as temp:
                        output = Path(temp) / target
                        subprocess.run(
                            ["make", "--no-print-directory", "-f", "Makefile",
                             f"TARGET={output}", f"SEED={seed}", f"ARCH={arch}", "all"],
                            cwd=root, check=True,
                        )
                        output.replace(root / target)
                    writer.writerow([target, mu_text, run_index + 1, seed, arch])
                    stream.flush()
    finally:
        header.write_text(original)
    print(f"Completed {total} builds. Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError, InvalidOperation) as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        sys.exit(1)

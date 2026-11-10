#!/usr/bin/env python3
"""Plot a benchmark JSON or CSV file. See ``python -m mi300x_bench plot``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mi300x_bench.plot import plot_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot mi300x-bench JSON or CSV results")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        plot_results(args.input, args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

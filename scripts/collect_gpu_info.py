#!/usr/bin/env python3
"""Print rocm-smi product name and VRAM as JSON.

Exits 0 when rocm-smi is absent. The human-readable lines go to stderr and
the machine-readable object goes to stdout.
"""

from __future__ import annotations

import json
import sys

from mi300x_bench.gpu_info import collect_gpu_info, format_gpu_info


def main() -> int:
    info = collect_gpu_info()
    print(format_gpu_info(info), file=sys.stderr)
    json.dump(info, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

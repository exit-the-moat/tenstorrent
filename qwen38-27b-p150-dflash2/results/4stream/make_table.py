# SPDX-License-Identifier: Apache-2.0
"""Print the counted 4-chat tables of the README from the steady.py result files in this folder.

  python3 results/4stream/make_table.py            # the development runs (steady/)
  python3 results/4stream/make_table.py selftest   # the package self-test runs (selftest/)

Each cell is the median over the 3 counted rounds of one start. STEADY = tokens produced while all n streams decode,
divided by that window; AGGREGATE = all completion tokens divided by the burst wall time (incl. TTFT and the
serialized prefills). Both are tokens per second over all streams together.
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {
    "steady": [
        ("baseline: Episode 2 multi-stream schedule (DF_AUTO, before the 4-chat work)", "s0"),
        ("+ batching (DF_MS_ROWS_AUTO, DF_MS_BDPREFIX=2,4,6, DF_MS_BSDPA)", "s1"),
        ("+ WY block GDN verify (DF_GDN_WY), gate start 1", "g5"),
        ("+ WY block GDN verify (DF_GDN_WY), gate start 2", "g6"),
    ],
    "selftest": [
        ("`--ep2` (the Episode 2 release)", "ep2"),
        ("`--no-wy` (Episode 2 + batching)", "nowy"),
        ("default `--cards 2` (+ WY)", "wy"),
    ],
}


def cell(path, key):
    if not os.path.exists(path):
        return "-"
    rows = json.load(open(path))
    v = [r[key] for r in rows if r.get(key) is not None]
    return f"{statistics.median(v):.1f}" if v else "-"


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "steady"
    d = os.path.join(HERE, which)
    for key, title in (("steady_total_tps", "STEADY"), ("aggregate_tps", "AGGREGATE")):
        print(f"| {title} tok/s, total | 1 live | 2 live | 3 live | 4 live |")
        print("|---|---|---|---|---|")
        for label, tag in RUNS[which]:
            cells = [cell(os.path.join(d, f"steady-{tag}-{n}.json"), key) for n in (1, 2, 3, 4)]
            print(f"| {label} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()

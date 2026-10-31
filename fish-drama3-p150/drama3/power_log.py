#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Episode 0: card power while generating (host side, outside the container).

    power_log.py log <out.jsonl> <stop file>      sample `tt-smi -s` every 2 s until <stop file> exists
    power_log.py merge <out.jsonl> <timing.json>  add power_w (mean chip power) + board_power_w per line,
                                                  from each line's start_epoch .. end_epoch
"""
import json
import os
import subprocess
import sys
import time

TT_SMI = "/opt/tenstorrent/venv/bin/tt-smi"


def sample():
    d = json.loads(subprocess.run([TT_SMI, "-s"], capture_output=True, text=True, timeout=30).stdout)
    t = d["device_info"][0]["telemetry"]
    return dict(t=time.time(), power=float(t["power"]), board_power=float(t["board_power"]), aiclk=float(t["aiclk"]))


def log(out, stop):
    with open(out, "a") as f:
        while not os.path.exists(stop):
            try:
                f.write(json.dumps(sample()) + "\n")
                f.flush()
            except Exception as e:  # keep sampling through a transient tt-smi failure
                f.write(json.dumps(dict(t=time.time(), error=str(e))) + "\n")
            time.sleep(2)


def merge(samples_path, timing_path):
    s = [json.loads(l) for l in open(samples_path) if '"power"' in l]
    timing = json.load(open(timing_path))
    for lid, t in timing.items():
        if "start_epoch" not in t:
            continue
        w = [x for x in s if t["start_epoch"] <= x["t"] <= t["end_epoch"]]
        if w:
            t["power_w"] = round(sum(x["power"] for x in w) / len(w), 1)
            t["board_power_w"] = round(sum(x["board_power"] for x in w) / len(w), 1)
            t["power_samples"] = len(w)
    json.dump(timing, open(timing_path, "w"), indent=1)
    print({k: (v.get("power_w"), v.get("board_power_w")) for k, v in timing.items()})


if __name__ == "__main__":
    {"log": log, "merge": merge}[sys.argv[1]](*sys.argv[2:4])

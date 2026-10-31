# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 7: one scoreboard Day 3 -> Day 6 from the files the runs wrote (timing.json, sanity.json, gen.log,
batch_summary.json). Numbers that only exist in HANDOFF.md are taken from there and marked as such; numbers that
were never measured are null ("missing"). No Whisper or card runs here.

    python3 models/demos/blackhole/drama3/tests/day7_scoreboard.py /opt/ep0/day7

Definitions: overall RTF = sum(compute_s) / sum(audio_s) of the dir; TTFA first = the first line in run order (it
pays the voice prefill from Day 4b on), later = min..max of the others; backbone / tower / codec ms per frame =
sum(stage seconds) / sum(frames) (codec: Day 3 whole-clip CPU decode codec_s, Day 4 streamed CPU codec_decode_s,
Day 4b+ card codec card_codec_s, card time only, the windows overlap the generation); power = min..max of the per-line means of the 2 s tt-smi samples (batch rows:
min..max of the samples, from batch_summary.json); Whisper mean = mean word match of the stored faster-whisper
small transcript vs the written form (same words() as day6_pick, recomputed from sanity.json so every day uses one
definition).
"""
import difflib
import json
import re
import sys
from pathlib import Path

import yaml

E = Path("/opt/ep0")
HANDOFF = "models/demos/blackhole/drama3/HANDOFF.md"
WRITTEN = {l["id"]: l["written"] for l in yaml.safe_load(open(E / "voice/lines.yaml"))["lines"]}
WRITTEN.update({l["id"]: l["written"] for l in json.load(open(E / "voice/card_lines.json"))})
MISSING = [
    "Day 3 card DRAM beyond the backbone (the tower ran on the host)",
    "Day 4 card DRAM from a run file (gen.log logs the backbone only; 27.72 GB incl. the tower is in HANDOFF.md)",
    "a Day 3 ref_v2 8-line set (only ref_long was rendered end to end)",
    "Windows scorer (WER + speaker similarity) for the Day 6 batch-8 set",
]


def words(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("-", " ")).split()


def rng(v):
    v = [x for x in v if x is not None]
    return [min(v), max(v)] if v else None


def dram(log):
    """card DRAM after warm-up / captures (Day 4b+), else the resident backbone (Day 3/4)."""
    txt = Path(log).read_text(errors="replace") if Path(log).exists() else ""
    m = re.findall(r"card DRAM ([0-9.]+) GB", txt)
    if m:
        return float(m[0]), f"{log} ('card DRAM')"
    m = re.findall(r"backbone resident: DRAM ([0-9.]+) GB", txt)
    return (float(m[0]), f"{log} (backbone resident only)") if m else (None, None)


def run_dir(d, codec_key, log=None):
    d = E / d
    t = {k: v for k, v in json.load(open(d / "timing.json")).items() if not k.startswith("_")}  # _batch*: batch meta
    order = sorted(t, key=lambda k: t[k]["start_epoch"])
    fr = sum(t[k]["frames"] for k in order)
    san = json.load(open(d / "sanity.json"))
    match = [difflib.SequenceMatcher(None, words(WRITTEN[r["id"]]), words(r["transcript"])).ratio() for r in san]
    r = dict(
        source=f"{d}/timing.json, {d}/sanity.json",
        lines=len(order),
        audio_s=round(sum(t[k]["audio_s"] for k in order), 1),
        compute_s=round(sum(t[k]["compute_s"] for k in order), 1),
        frames=fr,
        overall_rtf=round(sum(t[k]["compute_s"] for k in order) / sum(t[k]["audio_s"] for k in order), 3),
        rtf_lines=rng([t[k]["rtf"] for k in order]),
        ttfa_first_s=t[order[0]]["ttfa_s"],
        ttfa_later_s=rng([t[k]["ttfa_s"] for k in order[1:]]),
        backbone_ms_per_frame=round(1e3 * sum(t[k]["decode_backbone_s"] for k in order) / fr, 1),
        tower_ms_per_frame=round(1e3 * sum(t[k]["tower_steps_s"] for k in order) / fr, 1),
        codec_ms_per_frame=round(1e3 * sum(t[k][codec_key] for k in order) / fr, 1),
        codec_kind=codec_key,
        chip_power_w=rng([t[k].get("power_w") for k in order]),
        board_power_w=rng([t[k].get("board_power_w") for k in order]),
        whisper_mean=round(sum(match) / len(match), 3),
        whisper_min=round(min(match), 3),
    )
    r["card_dram_gb"], r["card_dram_source"] = dram(E / log if log else d / "gen.log")
    r["card_dram_scope"] = "backbone only" if r["card_dram_source"] and "backbone" in r["card_dram_source"] else "card"
    return r


def batch_dir(d, streams):
    b = json.load(open(E / d / "batch_summary.json"))
    r = run_dir(d, "card_codec_s")
    for k in ("overall_rtf", "rtf_lines", "backbone_ms_per_frame", "tower_ms_per_frame", "codec_ms_per_frame", "codec_kind"):
        r[k] = None
    r.update(source=f"{E / d}/batch_summary.json, timing.json, sanity.json", streams=streams,
             audio_per_wall_s=b["audio_per_wall"], wall_s=b["wall_s"], rtf_per_stream=b["rtf_per_stream"],
             ttfa_first_s=None, ttfa_later_s=None, ttfa_streams_s=b["ttfa_s"],
             backbone_ms_per_step=b["backbone_ms_per_step"], tower_ms_per_step=b["tower_ms_per_step"],
             host_ms_per_step=b["host_ms_per_step"], chip_power_w=b["chip_power_w"], board_power_w=b["board_power_w"],
             card_dram_gb=b["card_dram_gb"], card_dram_source=f"{E / d}/batch_summary.json", card_dram_scope="card")
    return r


def f(v):
    if v is None:
        return "missing"
    if isinstance(v, list):
        return f"{v[0]:g}-{v[1]:g}" if v[0] != v[1] else f"{v[0]:g}"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def main():
    out = Path(sys.argv[1])
    rows = []

    def add(stage, name, voice, r, note):
        rows.append(dict(stage=stage, run=name, voice=voice, note=note, **r))

    add("Day 3", "8 test lines (first end to end)", "ref_long", run_dir("day3/out", "codec_s", "day3/gen_all.log"),
        "host tower + host codec (whole clip after generation); card DRAM = backbone only")
    add("Day 4", "8 test lines", "ref_long", run_dir("day4/out", "codec_decode_s"), "CPU streaming codec")
    add("Day 4", "8 test lines", "ref_v2", run_dir("day4/out_v2", "codec_decode_s"), "CPU streaming codec")
    add("Day 4b", "8 test lines (final)", "ref_v2", run_dir("day4b/out_v2", "card_codec_s"),
        "prefix cache, device sampling, card codec; the default pipeline since")
    add("Day 5", "8 test lines, BFP8 gate/up on 8 layers (rejected)", "ref_v2", run_dir("day5/out_bfp8_8", "card_codec_s"),
        "decision keep BFP4: the default stays bit-identical to Day 4b (/opt/ep0/day5/gate_base)")
    for b in (2, 4, 8):
        add("Day 5b", f"8 test lines, batch {b}", "ref_v2", batch_dir(f"day5b/m{b}", b),
            "opt-in --batch; ms per step (B frames), TTFA / RTF per stream from the batch start")
    add("Day 6", "38 narration lines, sequential (canonical)", "ref_v2", run_dir("day6/narration", "card_codec_s"),
        "seed 0 except s03-01 = seed 2 (picks.json)")
    add("Day 6", "38 narration lines, batch 8", "ref_v2", batch_dir("day6/narration_b8", 8), "opt-in, 5 static batches")
    solo = next(r for r in rows if r["stage"] == "Day 4b")
    throughput = [dict(streams=1, audio_per_wall_s=round(solo["audio_s"] / solo["compute_s"], 2),
                       source=f"{E}/day4b/out_v2/timing.json (sum audio / sum compute)")]
    throughput += [dict(streams=r["streams"], audio_per_wall_s=r["audio_per_wall_s"], source=r["source"].split(",")[0])
                   for r in rows if r["stage"] == "Day 5b"]
    handoff = dict(
        source=HANDOFF,
        note="numbers that exist only in HANDOFF.md (profiles / runs without a run dir)",
        day4_card_dram_gb_incl_tower=27.72,
        day4b_device_only_ms_per_frame=dict(backbone_trace=22.5, tower_trace=9.2, host=1.8),
        day5b_device_ms_per_step=dict(B1=[22.2, 9.3], B2=[26.5, 9.8], B4=[35.3, 10.5], B8=[53.5, 11.8],
                                      source="/opt/ep0/day5b/profile_batch2.json, profile_batch3.json"),
        idle_power_w=dict(chip=19, board=81),
        missing=MISSING,
    )
    json.dump(dict(rows=rows, batch_throughput=throughput, handoff_only=handoff), open(out / "scoreboard.json", "w"), indent=1)

    L = ["# Drama 3 on one p150: scoreboard Day 3 -> Day 6", "",
         "Generated by `models/demos/blackhole/drama3/tests/day7_scoreboard.py` from the run files (Source column,",
         "`scoreboard.json`); definitions in the script docstring. \"missing\" = not measured.", "",
         "## Sequential runs (one line after the other)", "",
         "| stage | run | voice | lines | audio s | overall RTF | RTF per line | TTFA first s | TTFA later s | ms/frame backbone | tower | codec | card DRAM GB | chip W | board W | Whisper mean (min) | source |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "streams" in r:
            continue
        L.append(f"| {r['stage']} | {r['run']} | {r['voice']} | {r['lines']} | {r['audio_s']} | {f(r['overall_rtf'])} | "
                 f"{f(r['rtf_lines'])} | {f(r['ttfa_first_s'])} | {f(r['ttfa_later_s'])} | {f(r['backbone_ms_per_frame'])} | "
                 f"{f(r['tower_ms_per_frame'])} | {f(r['codec_ms_per_frame'])} ({r['codec_kind']}) | {f(r['card_dram_gb'])}{' (backbone only)' if r['card_dram_scope'] != 'card' else ''} | "
                 f"{f(r['chip_power_w'])} | {f(r['board_power_w'])} | {r['whisper_mean']:.3f} ({r['whisper_min']:.3f}) | "
                 f"`{r['source'].split(',')[0]}` |")
    L += ["", "## Batch runs (opt-in `run_gen.sh --batch --batch-size B`)", "",
          "| stage | run | lines | audio s | audio s per wall s | RTF per stream | TTFA per stream s | ms/step backbone | tower | host | card DRAM GB | chip W (samples) | board W (samples) | Whisper mean (min) | source |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "streams" not in r:
            continue
        L.append(f"| {r['stage']} | {r['run']} | {r['lines']} | {r['audio_s']} | {f(r['audio_per_wall_s'])} | {f(r['rtf_per_stream'])} | "
                 f"{f(r['ttfa_streams_s'])} | {f(r['backbone_ms_per_step'])} | {f(r['tower_ms_per_step'])} | {f(r['host_ms_per_step'])} | "
                 f"{f(r['card_dram_gb'])} | {f(r['chip_power_w'])} | {f(r['board_power_w'])} | "
                 f"{r['whisper_mean']:.3f} ({r['whisper_min']:.3f}) | `{r['source'].split(',')[0]}` |")
    L += ["", "## Batch throughput, 8 test lines, ref_v2", "", "| streams | audio s per wall s | source |", "|---|---|---|"]
    L += [f"| {x['streams']} | {x['audio_per_wall_s']} | `{x['source']}` |" for x in throughput]
    L += ["", "## Notes per row", ""] + [f"- {r['stage']}, {r['run']} ({r['voice']}): {r['note']}. DRAM: {r['card_dram_source']}." for r in rows]
    L += ["", f"## From HANDOFF.md only (`{HANDOFF}`)", "",
          "- Day 4 card DRAM incl. the tower: 27.72 GB (its gen.log only logs the backbone, 26.05 GB).",
          "- Day 4b device-only time per frame (`tests/profile_frame.py`): backbone trace 22.5 + tower trace 9.2 ms, host 1.8 ms.",
          "- Day 5b device time per step B=1/2/4/8 (`/opt/ep0/day5b/profile_batch2.json`, `profile_batch3.json`): backbone "
          "22.2/26.5/35.3/53.5 ms, tower 9.3/9.8/10.5/11.8 ms.",
          "- Idle power: chip 19 W, board 81 W (Day 4).",
          "", "Missing: " + "; ".join(MISSING) + ".", "",
          "Day 7 narration retries (`/opt/ep0/day6/narration_retry`, not a scoreboard row): the power log has a gap from "
          "04:37:38 CET of ~20 s (a duplicate run stopped the sampler; restarted, appending to the same power.jsonl)."]
    (out / "SCOREBOARD.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

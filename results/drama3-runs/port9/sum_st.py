import json, statistics as S, sys


def mmm(v):
    v = sorted(v)
    return f"{v[0]:.2f} / {S.median(v):.2f} / {v[-1]:.2f}"


for d in sys.argv[1:]:
    t = json.load(open(f"{d}/timing.json"))
    s = t["_summary"]
    j = [v for k, v in t.items() if not k.startswith("_")]
    try:
        bp = json.load(open(f"{d}/power.txt")).get("board_power", {})
    except Exception:
        bp = {}
    print(json.dumps({
        "run": d.rstrip("/").split("/")[-1], "B": s["B"], "jobs": len(j), "audio_s": s["audio_s"], "wall_s": s["wall_s"],
        "audio_per_wall": s["audio_per_wall"], "mean_active": s["mean_active"],
        "ms_step": f'{s["backbone_ms_per_step"]} + {s["tower_ms_per_step"]} + {s["host_ms_per_step"]}',
        "rtf_decode": mmm([x["rtf_decode"] for x in j]), "rtf": mmm([x["rtf"] for x in j]),
        "ttfa_admit": mmm([x["ttfa_from_admit_s"] for x in j]), "ttfa_t0": mmm([x["ttfa_s"] for x in j]),
        "worst_late": mmm([x["worst_late_s"] for x in j]), "eos": sum(x["stopped_by_eos"] for x in j),
        "dram_gb": s.get("dram_gb"), "board_W": f'{bp.get("mean")} ({bp.get("max")})'}))

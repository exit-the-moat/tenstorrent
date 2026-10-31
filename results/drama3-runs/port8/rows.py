import json
import os

os.chdir("/opt/ep0/port8")
cfg = {
    "s8_1": "port7 flags + ADDNORM_FUSED + MOE_TOPK_SOFTMAX + EXPERT_MM v2 D 8 (S8)",
    "s8_2": "S8, warm repeat 2",
    "s8_3": "S8, warm repeat 3",
    "s8b8": "S8 + --tower-mlp bfp8 (precision call)",
    "s8b4": "S8 + --tower-mlp bfp4, default programs (precision candidate)",
    "s8b4c": "S8 + --tower-mlp bfp4, bfp8 program configs (precision candidate)",
    "s8b4a": "S8 + bfp4 MLP + TOWER_ATTN_BFP4 (rejected)",
}
wh = {"s8_1": "0.908 (0.744)", "s8b8": "0.908 (0.744)", "s8b4": "0.932 (0.837)", "s8b4c": "0.921 (0.837)",
      "s8b4a": "0.913 (0.727)", "s8_2": "= s8_1 (identical wavs)", "s8_3": "= s8_1 (identical wavs)"}
for d in cfg:
    t = json.load(open(d + "/timing.json"))
    v = [x for x in t.values() if isinstance(x, dict) and "ttfa_s" in x]
    v.sort(key=lambda x: x.get("start_epoch", 0))
    au = sum(x["audio_s"] for x in v)
    cs = sum(x["compute_s"] for x in v)
    gs = sum(x["gen_s"] for x in v)
    fr = sum(x["frames"] for x in v)
    r = [x["rtf"] for x in v]
    later = [x["ttfa_s"] for x in v[1:]]

    def m(k):
        return sum((x[k]["mean"] if isinstance(x[k], dict) else x[k]) for x in v) / len(v)

    print(f"| {d} | {cfg[d]} | {len(v)} | {au:.1f} | {cs / au:.3f} | {min(r):.3f}-{max(r):.3f} | {v[0]['ttfa_s']:.3f} | "
          f"{min(later):.3f}-{max(later):.3f} | {gs / fr * 1e3:.2f} | {m('power_w'):.1f} | {m('board_power_w'):.1f} | {wh[d]} | "
          f"`/opt/ep0/port8/{d}/timing.json` |")

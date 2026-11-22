#!/usr/bin/env python3
"""img2: summarise a tracy ops_perf_results CSV for one denoising step (rows between two signposts), per op family.
Device kernel time per family on one chip (device 0 by default), calls, share of the step's device-busy time and of the
measured (unprofiled) step time. Gaps = op-to-op latency (idle between ops on the device).
  python prof_sum.py <report dir or csv> --from denoising_step_2 --to denoising_step_3 --step-s 2.252 --config <name> [--out json]"""
import argparse, collections, csv, glob, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("--from", dest="frm", default="denoising_step_2"); ap.add_argument("--to", default="denoising_step_3")
ap.add_argument("--device", default="0"); ap.add_argument("--step-s", type=float, default=0.0); ap.add_argument("--config", default="")
ap.add_argument("--out", default=""); ap.add_argument("--detail", action="store_true")
a = ap.parse_args()
p = a.path
if os.path.isdir(p):
    p = sorted(glob.glob(os.path.join(p, "**", "ops_perf_results_*.csv"), recursive=True))[-1]
rows = list(csv.DictReader(open(p)))
# locate signposts
idx = {r["OP CODE"]: i for i, r in enumerate(rows) if r["OP TYPE"] == "signpost"}
i0, i1 = idx[a.frm], idx[a.to]
seg = [r for r in rows[i0 + 1:i1] if r["OP TYPE"] != "signpost" and r["DEVICE ID"] == a.device]


def family(r):
    oc = r["OP CODE"]
    y = r.get("INPUT_0_Y_PAD[LOGICAL]", "").split("[")[0]
    z = r.get("INPUT_0_Z_PAD[LOGICAL]", "").split("[")[0]
    if "AllGatherMinimalMatmul" in oc:
        return "matmul + fused all-gather"
    if "StridedReduceScatter" in oc:
        return "matmul + fused reduce-scatter"
    if "Addcmul" in oc and "Matmul" in oc:
        return "matmul + fused gated add"
    if "Matmul" in oc:
        return "matmul (linear layers)"
    if "SDPA" in oc.upper():
        return "attention (joint SDPA)"
    if "AllGather" in oc:
        return "all-gather (card to card)"
    if "ReduceScatter" in oc:
        return "reduce-scatter (card to card)"
    if "DitFusedDistributed" in oc:
        return "adaLN norm (fused, distributed)"
    if oc.startswith("LayerNorm") or "RmsNorm" in oc:
        return "q/k RMS norm"
    if ("Binary" in oc or "Unary" in oc) and z not in ("", "1"):
        return "rope (eltwise on q/k)"
    if "Binary" in oc or "Ternary" in oc:
        return "small eltwise (timestep path)" if y == "32" else "modulation glue (gate multiply, residual add)"
    if "Heads" in oc:
        return "head split / concat"
    if "Unary" in oc:
        return "small eltwise (timestep path)"
    return "other (" + oc.replace("DeviceOperation", "") + ")"


acc = collections.defaultdict(float); cnt = collections.Counter(); gaps = 0.0
det = collections.defaultdict(float); dcnt = collections.Counter()
for r in seg:
    d = float(r["DEVICE KERNEL DURATION [ns]"] or 0) / 1e9
    g = float(r["OP TO OP LATENCY [ns]"] or 0) / 1e9
    f = family(r)
    acc[f] += d; cnt[f] += 1; gaps += g
    if a.detail:
        key = f + " | " + "x".join(r.get(f"INPUT_0_{c}_PAD[LOGICAL]", "?").split("[")[0] for c in "WZYX") + " | " + "x".join(r.get(f"INPUT_1_{c}_PAD[LOGICAL]", "?").split("[")[0] for c in "WZYX")
        det[key] += d; dcnt[key] += 1
busy = sum(acc.values())
print(f"csv {p}\nsegment {a.frm} -> {a.to}, device {a.device}: {len(seg)} ops, device busy {busy:.3f} s, op-to-op gaps {gaps:.3f} s, busy+gaps {busy+gaps:.3f} s, measured step (unprofiled) {a.step_s} s")
out = []
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    sh = v / a.step_s if a.step_s else None
    print(f"  {k:<36} {v:7.3f} s {cnt[k]:6d} calls  {100*v/busy:5.1f} % of busy" + (f"  {100*sh:5.1f} % of step" if sh else ""))
    out.append(dict(family=k, seconds=round(v, 4), calls=cnt[k], share_busy=round(v / busy, 4), share_step=round(sh, 4) if sh else None))
if a.step_s:
    rest = a.step_s - busy
    print(f"  {'(not on device: gaps, host, sync)':<36} {rest:7.3f} s          {100*rest/a.step_s:5.1f} % of step")
if a.detail:
    print("detail (top 40):")
    for k, v in sorted(det.items(), key=lambda kv: -kv[1])[:40]:
        print(f"  {v:7.3f} s {dcnt[k]:5d}  {k}")
if a.out:
    json.dump(dict(config=a.config, source_csv=p, segment=[a.frm, a.to], device=int(a.device), ops=len(seg), device_busy_s=round(busy, 4), op_to_op_gaps_s=round(gaps, 4),
                   measured_step_s=a.step_s, not_on_device_s=round(a.step_s - busy, 4) if a.step_s else None, families=out,
                   method="tracy device profiler (DEVICE KERNEL DURATION per op), one denoising step between signposts; step time from the unprofiled queue runs"), open(a.out, "w"), indent=1)

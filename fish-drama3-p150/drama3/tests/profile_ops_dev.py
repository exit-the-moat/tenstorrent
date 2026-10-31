# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port5 Track B: per-op DEVICE time of one traced decode piece (GDN mixer, attention mixer, MoE block) with the
device profiler (TT_METAL_DEVICE_PROFILER=1): each piece is captured as a trace, replayed once with the profiler
buffer freshly read, and every program's kernel span (first start .. last end over all cores and RISCs) is listed
in execution order.

    (fast env) TT_METAL_DEVICE_PROFILER=1 python -m models.demos.blackhole.drama3.tests.profile_ops_dev [--out f.json]
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)
MODEL = "/models/drama-3"
LOG = Path(os.environ.get("TT_METAL_HOME", "/opt/ep0/tt-metal")) / "generated/profiler/.logs/profile_log_device.csv"


def parse_log(n_pieces):
    """The traced replays in the device log (written at close_device): programs that ran exactly twice (the two
    replays of a trace; eager runs get new run ids), second run, ordered by start, split into pieces at gaps > 2 ms."""
    with open(LOG) as f:
        freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        occ, spans, risc = {}, {}, {}
        for r in csv.DictReader(f, skipinitialspace=True):
            if not r["zone name"].endswith("-KERNEL"):
                continue
            rid, t = int(r["run host ID"]), int(r["time[cycles since reset]"])
            core = (r["core_x"], r["core_y"], r["RISC processor type"])
            if r["type"] == "ZONE_START":
                occ[(core, rid)] = occ.get((core, rid), -1) + 1
            n = occ.get((core, rid), 0)
            lo, hi = spans.get((rid, n), (t, t))
            spans[(rid, n)] = (min(lo, t), max(hi, t))
            key = (rid, n, core)
            lo, hi = risc.get(key, (t, t))
            risc[key] = (min(lo, t), max(hi, t))
    per_risc = {}  # (rid, n) -> {RISC type: longest single-core kernel span in us}
    for (rid, n, core), (lo, hi) in risc.items():
        d = per_risc.setdefault((rid, n), {})
        d[core[2]] = max(d.get(core[2], 0.0), (hi - lo) / freq)
    nmax = {}
    for rid, n in spans:
        nmax[rid] = max(nmax.get(rid, 0), n)
    prog = sorted(((rid,) + spans[(rid, 1)] for rid, n in nmax.items() if n == 1), key=lambda x: x[1])
    pieces, cur = [], []
    for p in prog:
        if cur and (p[1] - cur[-1][2]) / freq > 2000:
            pieces.append(cur)
            cur = []
        cur.append(p)
    pieces.append(cur)
    return [[(rid, (lo - pc[0][1]) / freq, (hi - lo) / freq, {k: round(v, 2) for k, v in per_risc[(rid, 1)].items()}) for rid, lo, hi in pc]
            for pc in pieces[-n_pieces:]]


def device_op_names(fn):
    """Device operations of one eager call of fn, in order (ttnn graph capture; best effort)."""
    try:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        fn()
        g = ttnn.graph.end_graph_capture()
        out = []
        for n in g:
            p = n.get("params", {}) if isinstance(n, dict) else {}
            nt = n.get("node_type", "") if isinstance(n, dict) else ""
            if nt == "function_start" and "DeviceOperation" in str(p.get("name", "")):
                out.append(str(p.get("name")).replace("ttnn::", "").replace("DeviceOperation", ""))
        return out
    except Exception as e:  # noqa
        print("graph capture failed:", e, flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/opt/ep0/port5/profile_ops_dev.json")
    ap.add_argument("--pieces", default="gdn_mixer,attn_mixer,moe_block")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.prompt import FrameEmbedder
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.weight_mapping import read_tensors, weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.tt_transformers.tt.common import Mode

    cfg = json.load(open(f"{MODEL}/config.json"))
    wmap = weight_map(MODEL)
    keys = ["model.language_model.embed_tokens.weight", "audio_decoder.codebook_embeddings.weight"]
    t = read_tensors(MODEL, wmap, keys)
    E = FrameEmbedder(t[keys[0]], t[keys[1]], cfg["semantic_start_token_id"], cfg["semantic_end_token_id"])(torch.load("/opt/ep0/day3/close.rows.pt"))
    cand = torch.tensor(sorted({cfg["eos_token_id"], *range(cfg["semantic_start_token_id"], cfg["semantic_end_token_id"] + 1)}))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    res = {}
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), wmap, cand)
        bb.reset()
        bb.prefill(E)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        T = E.shape[0]
        bb._stage(E[-1], T)
        gdn, att = bb.layers[0], bb.layers[3]

        def mixer(layer):
            def f():
                xn = layer.attention_norm(bb.x_buf, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
                if layer.is_full_attention:
                    return layer.attention.forward(xn, bb.cos_buf, bb.sin_buf, position_tensor=bb.pos_buf, page_table=bb.page_table)
                return layer.attention.forward(xn, mode="recurrent")
            return f

        def fused_mixer(layer):
            from models.demos.blackhole.drama3.tt.gdn_fused_decode import FusedGDNDecode

            fz = FusedGDNDecode(layer.attention, mesh)

            def f():
                xn = layer.attention_norm(bb.x_buf, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
                return fz.forward(xn, layer.attention.compute_kernel_config_decode)
            return f

        def moe(layer):
            def f():
                xn = layer.ffn_norm(bb.x_buf, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
                return layer.feed_forward.forward(xn, mode="decode")
            return f

        pieces = {"gdn_mixer": mixer(gdn), "attn_mixer": mixer(att), "moe_block": moe(gdn), "gdn_layer_full": lambda: gdn.forward(bb.x_buf, mode="decode"),
                  "gdn_mixer_fused": fused_mixer(gdn),
                  "attn_layer_full": lambda: att.forward(bb.x_buf, cos=bb.cos_buf, sin=bb.sin_buf, mode="decode", position_tensor=bb.pos_buf, page_table=bb.page_table)}
        tids, names = {}, {}
        for name in a.pieces.split(","):
            fn = pieces[name]
            fn()
            ttnn.synchronize_device(mesh)
            names[name] = device_op_names(fn)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)  # drain the device buffer (it holds a limited number of zones)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            for _ in range(2):  # the last replay is the measured one (few replays: the profiler buffer is finite)
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            ttnn.release_trace(mesh, tid)
            tids[name] = len(tids)
            print("captured", name, flush=True)
    finally:
        ttnn.close_device(mesh)
    spans = parse_log(len(tids))
    if len(spans) != len(tids):
        print("found", len(spans), "replays for", len(tids), "pieces", flush=True)
    for name, tid in tids.items():
        us = spans[tid] if tid < len(spans) else []
        if not us:
            print(name, "no profiler rows", flush=True)
            continue
        res[name] = dict(
            n_programs=len(us),
            span_ms=round((us[-1][1] + us[-1][2]) / 1e3, 4),
            sum_kernel_ms=round(sum(o_[2] for o_ in us) / 1e3, 4),
            op_names=len(names.get(name) or []),
            names=names.get(name),
            ops=[dict(i=i, run_id=rid, start_us=round(st, 2), dur_us=round(d, 2), risc=rk) for i, (rid, st, d, rk) in enumerate(us)],
        )
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "ops"}), flush=True)
        for o in res[name]["ops"]:
            print("  ", json.dumps(o), flush=True)
        print("  names", json.dumps(names.get(name)), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: op-by-op device time of one backbone decode layer (GDN or attention, with its MoE).

Every ttnn operation called during one eager decode forward of the chosen layer is ALSO captured as a one-op
trace with the same arguments and replayed N times (device time without host dispatch, like inside the real
decode trace). Prints the ops in call order with their caller line, and a total.

    (fast env) python -m models.demos.blackhole.drama3.tests.profile_ops [--layer 0] [--n 50]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)
MODEL = "/models/drama-3"

STATE = dict(depth=0, rows=[], mesh=None, n=50, on=False)


def _free(out):
    if isinstance(out, (list, tuple)):
        for o in out:
            _free(o)
    elif isinstance(out, ttnn.Tensor):
        try:
            ttnn.deallocate(out)
        except Exception:  # noqa: BLE001
            pass


def _wrap(name, fn):
    def w(*args, **kwargs):
        if not STATE["on"] or STATE["depth"] > 0:
            return fn(*args, **kwargs)
        STATE["depth"] += 1
        try:
            out = fn(*args, **kwargs)
            mesh = STATE["mesh"]
            f = sys._getframe(1)
            where = f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"
            ms = None
            try:
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                extra = fn(*args, **kwargs)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                for _ in range(STATE["n"]):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                ms = (time.time() - t0) / STATE["n"] * 1e3
                ttnn.release_trace(mesh, tid)
                STATE.setdefault("keep", []).append(extra)  # no free: views (reshape) share the buffer
            except Exception as e:  # noqa: BLE001
                ms = f"untraceable: {str(e)[:60]}"
            STATE["rows"].append((name, where, ms))
            return out
        finally:
            STATE["depth"] -= 1

    return w


def install():
    for mod, prefix in ((ttnn, "ttnn."), (ttnn.experimental, "ttnn.experimental."), (ttnn.transformer, "ttnn.transformer.")):
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if callable(obj) and "Operation" in type(obj).__name__ and name not in ("synchronize_device",):
                setattr(mod, name, _wrap(prefix + name, obj))
    gi = ttnn.Tensor.__getitem__
    ttnn.Tensor.__getitem__ = _wrap("Tensor.__getitem__", gi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tower", action="store_true", help="profile tower layer --layer (0-3 shared, 4-12 residual) instead")
    ap.add_argument("--prefill", type=int, default=0, help="profile a GDN-layer prefill of this many rows (bucket 128) instead")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.generate import TT_ENV

    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.weight_mapping import weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs

    cfg = json.load(open(f"{MODEL}/config.json"))
    cand = torch.tensor(sorted({cfg["eos_token_id"], *range(cfg["semantic_start_token_id"], cfg["semantic_end_token_id"] + 1)}))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    STATE["mesh"], STATE["n"] = mesh, a.n
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), weight_map(MODEL), cand)
        bb.reset()
        layer = bb.layers[a.layer]
        bb._stage(torch.randn(args.dim) * 0.3, 700)
        if a.tower:
            from models.demos.blackhole.drama3.host.tower import CodebookTower
            from models.demos.blackhole.drama3.tt.tower_device import TowerDevice

            ht = CodebookTower(MODEL, weight_map(MODEL), cfg)
            tw = TowerDevice(mesh, ht, bb.rope)
            tl = (tw.shared + tw.residual)[a.layer]
            tw._stage_pos(700)
            hx = ttnn.from_torch(torch.randn(1, 1, args.dim) * 0.3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            ex = ttnn.from_torch(torch.randn(1, 1, args.dim) * 0.3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            layer = type("L", (), {"is_full_attention": False})()

        if a.prefill:
            xp = ttnn.from_torch(torch.randn(1, 128, args.dim) * 0.3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            layer.feed_forward.experts.prefill_valid_len = a.prefill

        def fwd():
            if a.prefill:
                return layer.forward(xp, mode="prefill", chunk_size=layer.attention.long_prefill_chunk_size, valid_len=a.prefill)
            if a.tower:
                return tl(hx, ex, tw.cos_buf, tw.sin_buf, pos=tw.pos_buf)
            if layer.is_full_attention:
                return layer.forward(bb.x_buf, cos=bb.cos_buf, sin=bb.sin_buf, mode="decode", position_tensor=bb.pos_buf, page_table=bb.page_table)
            return layer.forward(bb.x_buf, mode="decode")

        for _ in range(2):  # compile
            _free(fwd())
        ttnn.synchronize_device(mesh)
        # whole layer, traced
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = fwd()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        for _ in range(a.n):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        whole = (time.time() - t0) / a.n * 1e3
        ttnn.release_trace(mesh, tid)
        _free(out)
        install()
        STATE["on"] = True
        _free(fwd())
        STATE["on"] = False
        agg = defaultdict(float)
        tot = 0.0
        for name, where, ms in STATE["rows"]:
            print(f"{ms if isinstance(ms, str) else f'{ms:8.4f}'}  {name:42s} {where}", flush=True)
            if not isinstance(ms, str):
                tot += ms
                agg[name] += ms
        print(f"layer {a.layer} ({'tower' if a.tower else 'attention' if layer.is_full_attention else 'GDN'}): whole layer traced {whole:.3f} ms, "
              f"{len(STATE['rows'])} ops, sum of one-op traces {tot:.3f} ms", flush=True)
        for k_, v_ in sorted(agg.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {v_:.4f} ms  {k_}", flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise

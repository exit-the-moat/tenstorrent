# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: where does the line-suffix prefill (after the cached voice prefix) spend its time? Per layer: token
mixer, router, routed experts (with the largest per-expert row count), shared expert; device sync after each.

    (fast env) python -m models.demos.blackhole.drama3.tests.profile_suffix [--line fix]
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV, VOICE, reference_codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="fix")
    ap.add_argument("--ref", default=str(VOICE / "ref_v2.wav"))
    a = ap.parse_args()
    os.environ.setdefault("DRAMA3_CARD_CODEC", "0")  # these checks open the card with the GDN L1_SMALL size
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    from models.demos.blackhole.drama3.host.generate_fast import FastSession
    from models.demos.blackhole.drama3.tt.backbone_paged import BLOCK
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
    from models.tt_transformers.tt.common import Mode

    lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
    refs = [(Path(a.ref).with_suffix(".txt").read_text().strip(), reference_codes(a.ref, "/opt/ep0/day3"))]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    try:
        s = FastSession(mesh)
        rows = s.prompt(lines[a.line]["text"], refs)
        P = s.split_point([s.prompt(x, refs) for x in ("A", "Z")] + [rows])
        s.set_voice(rows)
        bb = s.bb
        for _ in range(2):  # warm, then timed
            t0 = time.time()
            s._prefill_line(rows)
            ttnn.synchronize_device(mesh)
            print("suffix prefill (backbone + tower)", round(time.time() - t0, 3), flush=True)
        E = s.embedder(rows[P:])
        T = E.shape[0]
        bucket = bb.bucket_for(T)
        x = torch.zeros(1, bucket, E.shape[-1])
        x[0, :T] = E
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cos, sin = bb.rope.get_prefill_rot_mats(P, bucket)
        chunk_pt = ttnn.from_torch(bb.page_table_host[:, P // BLOCK : (P + bucket) // BLOCK].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        csi = ttnn.from_torch(torch.tensor([P], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        bb.restore_gdn(s.gdn_snap)
        ttnn.synchronize_device(mesh)
        out = []
        for i, layer in enumerate(bb.layers):
            layer.feed_forward.experts.prefill_valid_len = T
            t0 = time.time()
            xn = layer.attention_norm(x, mode=Mode.PREFILL, norm_config=None)
            if layer.is_full_attention:
                att = layer.attention.forward(xn, cos, sin, page_table=bb.page_table, chunk_page_table=chunk_pt, chunk_start_idx_tensor=csi)
            else:
                att = layer.attention.forward(xn, mode="chunk", chunk_size=layer.attention.long_prefill_chunk_size, valid_len=T)
            h = ttnn.add(x, att)
            ttnn.synchronize_device(mesh)
            t1 = time.time()
            moe = layer.feed_forward
            ff = layer.ffn_norm(h, mode=Mode.PREFILL, norm_config=None)
            ff4 = ttnn.reshape(ff, (1, 1, ff.shape[-2], ff.shape[-1]))
            r = moe.router(ff4)
            ttnn.synchronize_device(mesh)
            t2 = time.time()
            ex = moe.experts(ff4, r, mode="prefill")
            ttnn.synchronize_device(mesh)
            t3 = time.time()
            sh = moe.shared.forward(ff)
            ttnn.synchronize_device(mesh)
            t4 = time.time()
            rt = ttnn.to_torch(r).float().reshape(bucket, -1)[:T]
            need = int(torch.bincount(rt.topk(8, -1).indices.reshape(-1), minlength=rt.shape[-1]).max())
            y = ttnn.add(h, ttnn.add(ttnn.reshape(ex, h.shape), ttnn.reshape(sh, h.shape)))
            for t_ in (xn, att, ff, r, ex, sh, h, x):
                ttnn.deallocate(t_)
            x = y
            out.append(dict(layer=i, attn=layer.is_full_attention, mixer_ms=round(1e3 * (t1 - t0), 1), router_ms=round(1e3 * (t2 - t1), 1),
                            routed_ms=round(1e3 * (t3 - t2), 1), shared_ms=round(1e3 * (t4 - t3), 1), max_expert_rows=need))
            print(json.dumps(out[-1]), flush=True)
        tot = {k: round(sum(r_[k] for r_ in out), 1) for k in ("mixer_ms", "router_ms", "routed_ms", "shared_ms")}
        print("TOTAL", json.dumps(tot), flush=True)
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

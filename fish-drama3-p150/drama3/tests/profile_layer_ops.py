# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port6 Track A: synced wall time per ttnn call inside one suffix-prefill layer (a GDN layer and an attention
layer): every top-level ttnn.* call (and the GDN helpers) of the layer forward is wrapped with a device sync, so
each entry is that call's dispatch + device time. Nested calls count once (at the outermost wrapped call).

    (interactive env) python -m models.demos.blackhole.drama3.tests.profile_layer_ops [--line fix] [--layers 0,3]
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
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--ref", default=str(VOICE / "ref_v2.wav"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.environ.setdefault("DRAMA3_CARD_CODEC", "0")
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    from models.demos.blackhole.drama3.host.generate_fast import FastSession
    from models.demos.blackhole.drama3.tt.backbone_paged import BLOCK
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
    import models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet as gdn_mod
    import models.demos.blackhole.qwen36.tt.gdn.fused_chunk as fc_mod
    import models.demos.blackhole.qwen36.tt.moe.prefill_grouped as pg

    lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
    refs = [(Path(a.ref).with_suffix(".txt").read_text().strip(), reference_codes(a.ref, "/opt/ep0/day3"))]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    want = {int(i) for i in a.layers.split(",")}
    out = {}
    try:
        s = FastSession(mesh)
        rows = s.prompt(lines[a.line]["text"], refs)
        P = s.split_point([s.prompt(x, refs) for x in ("A", "Z")] + [rows])
        s.set_voice(rows)
        bb = s.bb
        for _ in range(2):
            s._prefill_line(rows)
        ttnn.synchronize_device(mesh)

        acc, depth = {}, [0]

        def wrap(mod, name, label):
            f = getattr(mod, name)

            def w(*args, **kw):
                if depth[0]:
                    return f(*args, **kw)
                depth[0] += 1
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                try:
                    return f(*args, **kw)
                finally:
                    ttnn.synchronize_device(mesh)
                    d = acc.setdefault(label, [0, 0.0])
                    d[0] += 1
                    d[1] += time.time() - t0
                    depth[0] -= 1

            setattr(mod, name, w)
            return (mod, name, f)

        patched = []
        for name in dir(ttnn):
            obj = getattr(ttnn, name)
            if callable(obj) and not isinstance(obj, type) and not name.startswith("_") and name not in (
                    "synchronize_device", "deallocate", "to_torch", "from_torch", "MeshShape", "CoreGrid", "CoreCoord"):
                try:
                    patched.append(wrap(ttnn, name, name))
                except Exception:
                    pass
        for name in ("transformer",):
            sub = getattr(ttnn, name)
            for n2 in dir(sub):
                if callable(getattr(sub, n2)) and not n2.startswith("_"):
                    patched.append(wrap(sub, n2, f"{name}.{n2}"))
        for mod, names in ((gdn_mod, ("_causal_conv1d_fir", "l2_norm_ttnn", "rms_norm_gated_ttnn")),
                           (gdn_mod, ("chunk_gated_delta_rule_fused_adapter",)) if hasattr(gdn_mod, "chunk_gated_delta_rule_fused_adapter") else (gdn_mod, ()),
                           (fc_mod, ("chunk_gated_delta_rule_fused_adapter",)),
                           (pg, ("grouped_prefill_forward",))):
            for n2 in names:
                if hasattr(mod, n2):
                    patched.append(wrap(mod, n2, f"{mod.__name__.split('.')[-1]}.{n2}"))
        # replicate bb.prefill up to the wanted layers
        E = s.embedder(rows[P:])
        T = E.shape[0]
        bucket = bb.bucket_for(T)
        bb.restore_gdn(s.gdn_snap)
        x = torch.zeros(1, bucket, E.shape[-1])
        x[0, :T] = E
        ttnn.synchronize_device(mesh)
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cos, sin = bb.rope.get_prefill_rot_mats(P, bucket)
        chunk_pt = ttnn.from_torch(bb.page_table_host[:, P // BLOCK : (P + bucket) // BLOCK].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        csi = ttnn.from_torch(torch.tensor([P], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        for i, layer in enumerate(bb.layers):
            if i > max(want):
                break
            layer.feed_forward.experts.prefill_valid_len = T
            acc.clear()
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            if layer.is_full_attention:
                y = layer.forward(x, cos=cos, sin=sin, mode="prefill", page_table=bb.page_table, chunk_page_table=chunk_pt, chunk_start_idx_tensor=csi)
            else:
                y = layer.forward(x, mode="prefill", chunk_size=layer.attention.long_prefill_chunk_size, valid_len=T)
            ttnn.synchronize_device(mesh)
            tot = time.time() - t0
            if len(y.shape) == 4:
                y = ttnn.reshape(y, (1, y.shape[-2], y.shape[-1]))
            x = y
            if i in want:
                ops = sorted(((k, v[0], round(1e3 * v[1], 3)) for k, v in acc.items()), key=lambda r: -r[2])
                out[i] = dict(attn=layer.is_full_attention, total_ms=round(1e3 * tot, 2), sum_ms=round(sum(r[2] for r in ops), 2),
                              ops=[dict(op=k, n=n, ms=ms) for k, n, ms in ops])
                print(i, json.dumps(out[i]), flush=True)
        for mod, name, f in patched:
            setattr(mod, name, f)
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1)
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

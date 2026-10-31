# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4: the codebook tower on the card vs the host FP32 tower (same inputs, same RNG).

Prefill: per-layer PCC over the prompt rows. Steps: the backbone (paged, traced) drives N frames; at every step
both towers get the same backbone row and logits and a generator in the same state; the host tower's frame is fed
back to both. Reports per-layer / head-logit PCC and how many sampled codes are identical.

    python -m models.demos.blackhole.drama3.tests.check_tower_device [--steps 20] [--mlp bf16|bfp8] [--fidelity HiFi4]
"""
import argparse
import json
import os
import time

import torch

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)
MODEL = "/models/drama-3"


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="/opt/ep0/day3/close.rows.pt")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--mlp", default="bf16")
    ap.add_argument("--fidelity", default="HiFi4")
    ap.add_argument("--live-rope", type=int, default=0, help="1: the backbone's cos / sin (full width with DRAMA3_ROPE_MATMUL) as generate_fast passes them, so the one-token attention takes the fused pre path")
    ap.add_argument("--out", default="/opt/ep0/day4/check_tower.json")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.prompt import FrameEmbedder
    from models.demos.blackhole.drama3.host.tower import CodebookTower
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.device_check import dram_used_gb
    from models.demos.blackhole.drama3.tt.tower_device import TowerDevice
    from models.demos.blackhole.drama3.tt.weight_mapping import read_tensors, weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs

    cfg = json.load(open(f"{MODEL}/config.json"))
    wmap = weight_map(MODEL)
    host = CodebookTower(MODEL, wmap, cfg)
    e = read_tensors(MODEL, wmap, ["model.language_model.embed_tokens.weight"])["model.language_model.embed_tokens.weight"]
    emb = FrameEmbedder(e, host.codebook_embeddings, cfg["semantic_start_token_id"], cfg["semantic_end_token_id"])
    rows = torch.load(a.rows)
    T = rows.shape[0]
    E = emb(rows)
    res = dict(mlp=a.mlp, fidelity=a.fidelity)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), wmap, host.candidates, log=log)
        d0 = dram_used_gb(mesh)[0]
        t0 = time.time()
        dev = TowerDevice(
            mesh, host, bb.rope, mlp_dtype={"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[a.mlp],
            fidelity=getattr(ttnn.MathFidelity, a.fidelity), cos_sin_host=bb.cos_sin_host if a.live_rope else None,
        )
        res["tower_build_s"] = round(time.time() - t0, 1)
        res["tower_dram_gb"] = round(dram_used_gb(mesh)[0] - d0, 3)
        log(f"device tower built in {res['tower_build_s']}s, +{res['tower_dram_gb']} GB DRAM (total {dram_used_gb(mesh)[0]:.2f})")

        # compile everything eagerly BEFORE the backbone decode trace is parked: backbone prefill, tower prefill,
        # one tower step; then capture, then the real sequence
        bb.reset()
        normed, last, _, xn = bb.prefill(E, keep_device=True)
        dev.prefill(xn, rows, T)
        hd = ttnn.from_torch(normed[T - 1].reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        dev.step(last, hd, T - 1, rows[T - 1], [], torch.Generator().manual_seed(0))
        ttnn.deallocate(hd)
        ttnn.deallocate(xn)
        bb.capture(pos=T)
        bb.reset()
        normed, last, _, xn = bb.prefill(E, keep_device=True)

        host.reset(T + a.steps + 8)
        t0 = time.time()
        host_pre = host.prefill(normed[: T - 1], torch.arange(T - 1), rows[: T - 1], rows[1:T], collect=True)
        res["host_tower_prefill_s"] = round(time.time() - t0, 2)
        t0 = time.time()
        dev_pre = dev.prefill(xn, rows, T, collect=True)
        res["device_tower_prefill_s"] = round(time.time() - t0, 2)
        ttnn.deallocate(xn)
        res["prefill_layer_pcc"] = [round(pcc(d[: T - 1], h), 6) for d, h in zip(dev_pre, host_pre)]
        log(f"prefill per-layer PCC {res['prefill_layer_pcc']} host {res['host_tower_prefill_s']}s device {res['device_tower_prefill_s']}s")

        h_row, logits, pos, inp, window = normed[T - 1], last, T - 1, rows[T - 1], []
        steps, t_dev = [], []
        for i in range(a.steps):
            g1, g2 = torch.Generator().manual_seed(1000 + i), torch.Generator().manual_seed(1000 + i)
            fh, _, oh, lh = host.step(logits, h_row, pos, inp, window, g1, collect=True)
            hd = ttnn.from_torch(h_row.reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            t0 = time.time()
            fd, _, od, ld = dev.step(logits, hd, pos, inp, window, g2, collect=True, force_codes=fh[1:])
            t_dev.append(time.time() - t0)
            ttnn.deallocate(hd)
            steps.append(
                dict(
                    step=i,
                    codes_equal=int((fh == fd).sum()),
                    head_top1_equal=sum(int(x.argmax()) == int(y.argmax()) for x, y in zip(ld, lh)),
                    head_top5_overlap=[len(set(torch.topk(x, 5).indices.tolist()) & set(torch.topk(y, 5).indices.tolist())) for x, y in zip(ld, lh)],
                    layer_pcc_min=round(min(pcc(x, y) for x, y in zip(od, oh)), 6),
                    layer_pcc=[round(pcc(x, y), 4) for x, y in zip(od, oh)],
                    head_pcc_min=round(min(pcc(x, y) for x, y in zip(ld, lh)), 6),
                    host=fh.tolist(),
                    device=fd.tolist(),
                )
            )
            if int(fh[0]) == host.eos:
                break
            window = (window + [int(fh[0])])[-16:]
            pos += 1
            h_row, logits = bb.decode(emb(fh[None])[0], pos)
            inp = fh
        res["steps"] = steps
        n = len(steps)
        res["codes_identical"] = f"{sum(s['codes_equal'] for s in steps)}/{11 * n}"
        res["frames_identical"] = f"{sum(s['codes_equal'] == 11 for s in steps)}/{n}"
        res["step_layer_pcc_min"] = min(s["layer_pcc_min"] for s in steps)
        res["step_head_pcc_min"] = min(s["head_pcc_min"] for s in steps)
        res["head_top1_equal"] = f"{sum(s['head_top1_equal'] for s in steps)}/{9 * n}"
        res["device_step_ms_median_eager"] = round(1000 * sorted(t_dev[1:])[len(t_dev[1:]) // 2], 1) if n > 1 else None
        log(json.dumps({k: v for k, v in res.items() if k != "steps"}))
        json.dump(res, open(a.out, "w"), indent=1)
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

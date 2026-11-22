"""Episode 4: one-card SDPA sweep at the Wan self-attention shapes (tensor split: 20 of 40 heads per card, head 128).

For each N (480p 32,760 / 720p 75,600 tokens, padded to tiles) and each (q_chunk, k_chunk, fidelity, dtype), times
the ttnn SDPA (median of R calls after 2 warm-ups, synchronized) and compares the output with the stock config
(256x256, HiFi2, bf16) on the same inputs (PCC). Appends rows to /out/sdpa_bench.jsonl.
"""
import argparse
import json
import time

import torch
import ttnn

ap = argparse.ArgumentParser()
ap.add_argument("--n", default="32768,75616")
ap.add_argument("--chunks", default="256x256,128x512,256x512,512x256,512x512,128x256,384x512,64x512")
ap.add_argument("--fid", default="HiFi2")
ap.add_argument("--dtypes", default="bf16")
ap.add_argument("--heads", type=int, default=20)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--tag", default="")
a = ap.parse_args()

dev = ttnn.open_device(device_id=0)
grid = dev.compute_with_storage_grid_size()
DT = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}
FID = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}


def pcc(x, y):
    x, y = x.flatten().double(), y.flatten().double()
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / (x.norm() * y.norm()))


for n in [int(v) for v in a.n.split(",")]:
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, a.heads, n, 128) for _ in range(3))
    ref = None
    configs = [("256x256", "HiFi2", "bf16")] + [(c, f, d) for c in a.chunks.split(",") for f in a.fid.split(",")
                                                for d in a.dtypes.split(",")]
    seen = set()
    for c, f, d in configs:
        if (c, f, d) in seen:
            continue
        seen.add((c, f, d))
        qc, kc = (int(x) for x in c.split("x"))
        try:
            tq, tk, tv = (ttnn.from_torch(x, dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=dev) for x in (q, k, v))
            pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc,
                                        exp_approx_mode=False)
            cc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=FID[f], math_approx_mode=False,
                                                        fp32_dest_acc_en=False)
            ts = []
            for i in range(a.reps + 2):
                ttnn.synchronize_device(dev)
                t0 = time.time()
                o = ttnn.transformer.scaled_dot_product_attention(tq, tk, tv, is_causal=False, program_config=pc,
                                                                  compute_kernel_config=cc)
                ttnn.synchronize_device(dev)
                ts.append(time.time() - t0)
                if i < a.reps + 1:
                    ttnn.deallocate(o)
            out = ttnn.to_torch(o).float()
            ttnn.deallocate(o)
            for t in (tq, tk, tv):
                ttnn.deallocate(t)
            if ref is None:
                ref = out
            ms = sorted(ts[2:])[len(ts[2:]) // 2] * 1000
            rec = dict(tag=a.tag, n=n, heads=a.heads, chunks=c, fid=f, dtype=d, ms=round(ms, 2),
                       pcc_vs_stock=round(pcc(out, ref), 6))
        except Exception as e:  # noqa: BLE001
            rec = dict(tag=a.tag, n=n, heads=a.heads, chunks=c, fid=f, dtype=d, error=str(e)[:200])
        print(json.dumps(rec), flush=True)
        with open("/out/sdpa_bench.jsonl", "a") as fo:
            fo.write(json.dumps(rec) + "\n")
ttnn.close_device(dev)
print("SDPA_BENCH_DONE")

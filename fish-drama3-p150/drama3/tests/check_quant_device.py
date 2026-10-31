# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port6 Track A: the quantizer decode on the card (tt/quant_device.py) vs Fish's CPU quantizer (fp32): latent PCC /
relative error per window length, time per call (synced), and the card decoder's audio from both latents (SNR).
Codes: a real line's frames (day4b/out_v2/fix.frames.pt), windows starting at frame 0.

    python -m models.demos.blackhole.drama3.tests.check_quant_device [--out f.json]
"""
import argparse
import json
import time

import torch

import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/opt/ep0/port6/check_quant_device.json")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.codec import load_codec
    from models.demos.blackhole.drama3.tt.codec_device import DeviceDecoder
    from models.demos.blackhole.drama3.tt.quant_device import DeviceQuantizer

    codec = load_codec()
    codec.fold_weight_norm()
    frames = torch.load("/opt/ep0/day4b/out_v2/fix.frames.pt")  # [n, 11]: main token, then the 10 codes
    codes_all = frames[:, 1:].T.contiguous().long()  # [10, n]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    res = {}
    try:
        t0 = time.time()
        dq = DeviceQuantizer(mesh, codec)
        res["init_s"] = round(time.time() - t0, 1)
        decs = {}
        for n in (4, 10, 20, 32, 42, 64, 96):
            T = next(w for w in (32, 64, 96) if w >= n)
            cd = codes_all[:, :n]
            with torch.inference_mode():
                ref = codec.quantizer.decode(cd[None].clone())[0].T  # [4n, 1024]
            z = dq(cd, T)
            ttnn.synchronize_device(mesh)
            zh = ttnn.to_torch(z).float().reshape(4 * T, 1024)[: 4 * n]
            r = dict(T=T, pcc=float(torch.corrcoef(torch.stack([zh.flatten(), ref.flatten()]))[0, 1]),
                     rel=float((zh - ref).norm() / ref.norm()))
            ts = []
            for _ in range(5):
                ttnn.synchronize_device(mesh)
                t1 = time.time()
                zz = dq(cd, T)
                ttnn.synchronize_device(mesh)
                ts.append(time.time() - t1)
                ttnn.deallocate(zz)
            r["ms"] = round(1e3 * sorted(ts)[2], 2)
            if T not in decs:
                decs[T] = DeviceDecoder(mesh, codec, 4 * T)
            zr = torch.zeros(1, 4 * T, 1024)
            zr[0, : 4 * n] = ref
            a_ref = decs[T](zr)[: n * 2048]
            a_dev = decs[T](z)[: n * 2048]
            err = a_dev - a_ref
            r["audio_snr_db"] = round(float(10 * torch.log10((a_ref**2).sum() / (err**2).sum().clamp(min=1e-12))), 1)
            ttnn.deallocate(z)
            res[n] = r
            print(n, json.dumps(r), flush=True)
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

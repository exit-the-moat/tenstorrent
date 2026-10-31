# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: the card codec decoder (tt/codec_device.py) vs the host fp32 decoder on a real 96-frame window:
per-stage PCC, audio PCC / SNR, time per call (warm)."""
import json
import time

import torch

import ttnn
from models.demos.blackhole.drama3.host.codec import load_codec
from models.demos.blackhole.drama3.tt.codec_device import DeviceDecoder


def pcc(a, b):
    a, b = a.flatten().double() - a.double().mean(), b.flatten().double() - b.double().mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    c = load_codec()
    c.fold_weight_norm()
    codes = torch.load("/opt/ep0/day4b/gate8/fix.frames.pt")[64:160, 1:].T.contiguous()[None]
    with torch.inference_mode():
        z = c.quantizer.decode(codes)  # [1, 1024, 384]
        refs, x = [], z
        for i, mod in enumerate(c.decoder.model):
            x = mod(x)
            refs.append(x)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    try:
        dd = DeviceDecoder(mesh, c, z.shape[-1])
        zt = z.permute(0, 2, 1).contiguous()
        t0 = time.time()
        audio, outs = dd(zt, collect=True)
        first = time.time() - t0
        ts = []
        for _ in range(3):
            t0 = time.time()
            audio = dd(zt)
            ttnn.synchronize_device(mesh)
            ts.append(time.time() - t0)
        ref = refs[-1][0, 0]
        # stage boundaries of the host model: 0 conv, 1-4 blocks, 5 snake, 6 conv, 7 tanh
        kinds = [k for k, _ in outs]
        idx = [0] + [i for i, k in enumerate(kinds) if k == "res"][2::3] + [len(kinds) - 3, len(kinds) - 2, len(kinds) - 1]
        stage_pcc, stage_gain, stage_snr = [], [], []
        for j, i in enumerate(idx):
            o = outs[i][1][0]
            r = refs[j][0].T
            o = o[: r.shape[0], : r.shape[1]]
            stage_pcc.append(round(pcc(o, r), 6))
            stage_gain.append(round(float((o * r).sum() / (r * r).sum()), 4))
            stage_snr.append(round(float(10 * torch.log10(r.pow(2).sum() / (o - r).pow(2).sum())), 1))
        # every residual unit / op, with the host reference recomputed op by op
        print(json.dumps(dict(stage_gain=stage_gain, stage_snr_db=stage_snr)), flush=True)
        err = audio - ref
        res = dict(first_call_s=round(first, 1), ms_per_call=round(min(ts) * 1e3, 1), frames=codes.shape[-1],
                   stage_pcc=stage_pcc, audio_pcc=round(pcc(audio, ref), 6),
                   snr_db=round(float(10 * torch.log10(ref.pow(2).sum() / err.pow(2).sum())), 2),
                   max_abs=float(err.abs().max()), peak=float(audio.abs().max()))
        print(json.dumps(res), flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

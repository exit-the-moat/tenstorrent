# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: the card codec decoder op by op, each op fed the HOST reference input (no accumulation): gain and SNR
of every op vs the host fp32 op, to find where the level error comes from."""
import json

import torch

import ttnn
from models.demos.blackhole.drama3.host.codec import load_codec
from models.demos.blackhole.drama3.tt.codec_device import ACT, DeviceDecoder


def stats(o, r):
    o, r = o.double(), r.double()
    return dict(gain=round(float((o * r).sum() / (r * r).sum()), 5), snr=round(float(10 * torch.log10(r.pow(2).sum() / (o - r).pow(2).sum())), 1))


def main():
    c = load_codec()
    c.fold_weight_norm()
    codes = torch.load("/opt/ep0/day4b/gate8/fix.frames.pt")[64:160, 1:].T.contiguous()[None]
    m = c.decoder.model
    with torch.inference_mode():
        z = c.quantizer.decode(codes)
        host_ops = [m[0]]  # same op order as DeviceDecoder.stages
        for blk in list(m)[1:5]:
            host_ops += list(blk.block)
        host_ops += [m[5], m[6], m[7]]
        ins, x = [], z
        for op in host_ops:
            ins.append(x)
            x = op(x)
        ins.append(x)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    try:
        dd = DeviceDecoder(mesh, c, z.shape[-1])
        L = z.shape[-1]
        for i, (kind, p) in enumerate(dd.stages):
            xin = ins[i][0].T.contiguous()[None]  # [1, L, C]
            Lin = xin.shape[1]
            xd = ttnn.from_torch(xin.float(), dtype=ACT, layout=ttnn.TILE_LAYOUT, device=mesh)
            if kind == "conv":
                y = dd._conv(xd, Lin, p)
            elif kind == "snake":
                y = dd._snake_op(xd, p)
            elif kind == "trans":
                cc, s, cout = p
                y = dd._conv(xd, Lin, cc)
                y = ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), (1, Lin * s, cout))
            elif kind == "res":
                s1, c7, s2, c1 = p
                a = dd._snake_op(xd, s1)
                a7 = dd._conv(a, Lin, c7)
                b = dd._snake_op(a7, s2)
                b1 = dd._conv(b, Lin, c1)
                y = ttnn.add(xd, b1)
                with torch.inference_mode():  # sub-op references
                    hs1, hc7, hs2, hc1 = list(host_ops[i].block)
                    ra = hs1(ins[i])
                    ra7 = hc7(ra)
                    rb = hs2(ra7)
                    rb1 = hc1(rb)
                sub = {}
                for nm, dv, hv in (("snake1", a, ra), ("conv7", a7, ra7), ("snake2", b, rb), ("conv1", b1, rb1)):
                    sub[nm] = stats(ttnn.to_torch(dv).float()[0, : hv.shape[-1], : hv.shape[1]], hv[0].T)
            elif kind == "tanh":
                y = ttnn.tanh(xd)
            out = ttnn.to_torch(y).float()[0]
            ref = ins[i + 1][0].T
            row = dict(i=i, kind=kind, L=Lin, **stats(out[: ref.shape[0], : ref.shape[1]], ref))
            if kind == "res":
                row["sub"] = sub
            print(json.dumps(row), flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

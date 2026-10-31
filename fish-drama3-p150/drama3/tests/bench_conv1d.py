# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b: can ttnn.conv1d run the codec decoder's causal dilated k=7 convs? One conv per decoder stage
(window of 96 frames), left-padded causally, vs torch fp32: time (eager, after warm-up) and PCC."""
import json
import time

import torch

import ttnn

import sys

SHAPES = [  # (length, channels, dilation) of the residual-unit convs in a 96-frame window
    (3072, 768, 9),
    (24576, 384, 9),
    (98304, 192, 9),
    (196608, 96, 9),
    (196608, 96, 1),
]
LAYOUT = {"height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED, "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
          "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED, "auto": None}[sys.argv[1] if len(sys.argv) > 1 else "height"]
if len(sys.argv) > 2:
    SHAPES = SHAPES[: int(sys.argv[2])]


def pcc(a, b):
    a, b = a.flatten().double() - a.double().mean(), b.flatten().double() - b.double().mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    try:
        for L, C, d in SHAPES:
            torch.manual_seed(0)
            x = torch.randn(1, C, L)
            w = torch.randn(C, C, 7) / (C * 7) ** 0.5
            b = torch.randn(C) * 0.1
            ref = torch.nn.functional.conv1d(torch.nn.functional.pad(x, (6 * d, 0)), w, b, dilation=d)  # causal
            xt = torch.nn.functional.pad(x, (6 * d, 0)).permute(0, 2, 1).contiguous()  # [1, L + 6d, C]
            xd = ttnn.from_torch(xt.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
            wt = ttnn.from_torch(w.to(torch.bfloat16), dtype=ttnn.bfloat16)
            bt = ttnn.from_torch(b.reshape(1, 1, 1, C).to(torch.bfloat16), dtype=ttnn.bfloat16)
            cfg = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, shard_layout=LAYOUT, deallocate_activation=False)
            ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True)
            res = dict(L=L, C=C, d=d)
            try:
                for rep in range(3):
                    t0 = time.time()
                    out, [wt, bt] = ttnn.conv1d(
                        input_tensor=xd, weight_tensor=wt, bias_tensor=bt, device=mesh, in_channels=C, out_channels=C, batch_size=1,
                        input_length=L + 6 * d, kernel_size=7, stride=1, padding=0, dilation=d, groups=1, dtype=ttnn.bfloat16,
                        conv_config=cfg, compute_config=ckc, return_weights_and_bias=True,
                    )
                    ttnn.synchronize_device(mesh)
                    res[f"ms_{rep}"] = round((time.time() - t0) * 1e3, 2)
                    o = ttnn.to_torch(ttnn.sharded_to_interleaved(out) if out.is_sharded() else out).float().reshape(L, C).T[None]
                    ttnn.deallocate(out)
                res["pcc"] = round(pcc(o, ref), 6)
                res["gflop"] = round(2 * L * C * C * 7 / 1e9, 2)
            except Exception as e:  # noqa: BLE001
                res["error"] = str(e)[:300]
            print(json.dumps(res), flush=True)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

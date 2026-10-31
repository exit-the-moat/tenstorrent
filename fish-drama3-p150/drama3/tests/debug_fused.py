# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 4b debug: one fused tower frame vs the eager tower step with the same codes forced."""
import json
import os
from pathlib import Path

import torch
import yaml

import ttnn
from models.demos.blackhole.drama3.host.generate import TT_ENV, VOICE, reference_codes


def pcc(a, b):
    a, b = a.flatten().double() - a.double().mean(), b.flatten().double() - b.double().mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    os.environ.setdefault("DRAMA3_CARD_CODEC", "0")  # these checks open the card with the GDN L1_SMALL size
    for k, v in TT_ENV.items():
        os.environ.setdefault(k, v)
    os.environ["DRAMA3_TOWER_DEVICE_SAMPLING"] = "1"
    from models.demos.blackhole.drama3.host.generate_fast import FastSession
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

    lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}
    ref = str(VOICE / "ref_v2.wav")
    refs = [(Path(ref).with_suffix(".txt").read_text().strip(), reference_codes(ref, "/opt/ep0/day3"))]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    try:
        s = FastSession(mesh)
        rows = s.prompt(lines["fix"]["text"], refs)
        s._warm_T = rows.shape[0]
        s.split_point([s.prompt(x, refs) for x in ("A", "Z")] + [rows])
        s.warmup(rows)
        T = rows.shape[0]
        s.set_voice(rows)
        normed, logits = s._prefill_line(rows)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(normed[-1].reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), s.bb.dec_out[0])
        frame, _, noise = s.tower.step_fused(logits, T - 1, rows[T - 1], [], torch.Generator().manual_seed(0), debug=True)
        lgs = [ttnn.to_torch(lg).float().reshape(-1)[:1024] for lg in s.tower.fused_lgs]
        e_res = [ttnn.to_torch(b).float().reshape(-1)[:8].tolist() for b in s.tower.e_bufs[4:6]]
        # eager reference on the same codes (tower KV at T-1 rewritten)
        s._prefill_line(rows)
        hd = s._row_dev(normed[-1])
        fr_e, _, _, hl = s.tower.step(logits, hd, T - 1, rows[T - 1], [], torch.Generator().manual_seed(0), collect=True, force_codes=frame[1:])
        print(json.dumps(dict(
            fused_frame=frame.tolist(), eager_frame=fr_e.tolist(),
            head_pcc=[round(pcc(a, b), 4) for a, b in zip(lgs, hl)],
            fused_argmax=[int(a.argmax()) for a in lgs], eager_argmax=[int(b.argmax()) for b in hl],
            fused_max=[round(float(a.max()), 2) for a in lgs], e_bufs_res=e_res,
        )), flush=True)
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

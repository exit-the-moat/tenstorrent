# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port10 cc13 step 2: the card codec decoder (tt/codec_device.py, the --interactive levers) restricted to a core
sub-grid that is its own sub-device, against the default full-grid decoder: audio equality (torch.equal) and traced
time per window size, on the same bf16 latent (the host quantizer's z of real codes, as the card quantizer feeds it).

--grids: "/"-separated: "y0" (sub-device 1 = rows y0..last of the full width, sub-device 0 = the rows above, every
decoder program on sub-device 1), "full" (one sub-device = the whole grid: the restricted code path on every core,
layout ops as default), "est<y0>" (timing estimate, no sub-device: convs, snakes, adds, typecasts on rows y0..last,
the matmuls on a grid of that size at (0, 0), the tile <-> row-major ops and the channel-0 tail on the full grid).

    (--interactive env) run_check.sh <log> tests.cc13_subgrid --grids "full/8/est8" --out f.json
"""
import argparse
import json
import time

import torch

import ttnn


def snr_db(ref, x):
    err = (x - ref).double()
    return float(10 * torch.log10((ref.double() ** 2).sum() / (err**2).sum().clamp(min=1e-30)))


def timed(mesh, dec, tid, out, reps):
    """Median synced ms of (replay + readback) and of the replay alone."""
    full, dev = [], []
    for _ in range(reps):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        t1 = time.perf_counter()
        dec.audio(out)
        t2 = time.perf_counter()
        dev.append(t1 - t0)
        full.append(t2 - t0)
    full.sort()
    dev.sort()
    return round(1e3 * full[len(full) // 2], 2), round(1e3 * dev[len(dev) // 2], 2)


def capture(mesh, dec, z):
    x, _ = dec._body(z)
    ttnn.deallocate(x)
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out, _ = dec._body(z)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    return tid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="full/8/est8")
    ap.add_argument("--sizes", default="32,64,96")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.codec import load_codec
    from models.demos.blackhole.drama3.tt.codec_device import DeviceDecoder

    codec = load_codec()
    codec.fold_weight_norm()
    frames = torch.load("/opt/ep0/day4b/out_v2/fix.frames.pt")  # [n, 11]
    codes_all = frames[:, 1:].T.contiguous().long()
    sizes = [int(v) for v in a.sizes.split(",")]
    zs = {}
    with torch.inference_mode():
        for w in sizes:
            zs[w] = codec.quantizer.decode(codes_all[None, :, :w])[0].T.contiguous()[None]  # [1, 4 w, 1024]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536, trace_region_size=200 << 20)
    res = {}
    try:
        g = mesh.compute_with_storage_grid_size()
        res["grid"] = [g.x, g.y]
        zd = {w: ttnn.from_torch(zs[w], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh) for w in sizes}
        ref = {}
        res["default"] = {}
        for w in sizes:  # the default decoder, full grid, default sub-device manager
            dec = DeviceDecoder(mesh, codec, 4 * w)
            ref[w] = dec(zd[w]).clone()
            tid, out = capture(mesh, dec, zd[w])
            tr = dec.audio(out)
            ms, dev_ms = timed(mesh, dec, tid, out, a.reps)
            res["default"][w] = dict(trace_ms=ms, trace_dev_ms=dev_ms, trace_equal_eager=bool(torch.equal(tr, ref[w])))
            print("default", w, json.dumps(res["default"][w]), flush=True)
            ttnn.release_trace(mesh, tid)
            del dec
        for spec in a.grids.split("/"):
            mgr, lf = None, False
            if spec == "full":
                rect = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
                sds, sid, lf = [ttnn.SubDevice([rect])], ttnn.SubDeviceId(0), True
            elif spec.startswith("est"):
                y0 = int(spec[3:])
                rect = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, y0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
                sds, sid, lf = None, None, True
            else:
                y0 = int(spec)
                rect = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, y0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
                top = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, y0 - 1))})
                sds, sid = [ttnn.SubDevice([top]), ttnn.SubDevice([rect])], ttnn.SubDeviceId(1)
            if sds is not None:
                mgr = mesh.create_sub_device_manager(sds, 0)
                mesh.load_sub_device_manager(mgr)
            r = res[spec] = {"n_cores": rect.num_cores()}
            try:
                for w in sizes:
                    t0 = time.time()
                    try:
                        dec = DeviceDecoder(mesh, codec, 4 * w, cores=rect, sub_device_id=sid, layout_full=lf)
                        x = dec(zd[w]).clone()
                    except Exception as e:
                        r[w] = dict(error=str(e)[:600])
                        print(spec, w, "FAILS:", str(e)[:600], flush=True)
                        continue
                    rw = r[w] = dict(first_call_s=round(time.time() - t0, 1), equal=bool(torch.equal(ref[w], x)),
                                     max_abs=float((ref[w] - x).abs().max()), snr_db=round(snr_db(ref[w], x), 2))
                    tid, out = capture(mesh, dec, zd[w])
                    tr = dec.audio(out)
                    rw["trace_equal"] = bool(torch.equal(ref[w], tr))
                    rw["trace_ms"], rw["trace_dev_ms"] = timed(mesh, dec, tid, out, a.reps)
                    rw["vs_default"] = round(rw["trace_dev_ms"] / res["default"][w]["trace_dev_ms"], 2)
                    print(spec, w, json.dumps(rw), flush=True)
                    ttnn.release_trace(mesh, tid)
                    del dec
            finally:
                ttnn.synchronize_device(mesh)
                if mgr is not None:
                    mesh.clear_loaded_sub_device_manager()
                    mesh.remove_sub_device_manager(mgr)
        if a.out:
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

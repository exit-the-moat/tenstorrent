# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port10 cc13: do two traces on disjoint sub-devices overlap? Synthetic (no model): trace A = decode-like one-row
matmuls on the 8x8 cores (0..7, 0..7) (sub-device 0 = rows 0..7), trace B = eltwise ops with sub_core_grids on
rows 8..9 (sub-device 1). Wall time of A alone, B alone, A + B serial, and A + B issued for overlap on one
command queue (both orders) and on two (B on CQ 1).

    run_check.sh <log> tests.cc13_conc --cqs 2
"""
import argparse
import json
import time

import torch

import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cqs", type=int, default=2)
    ap.add_argument("--na", type=int, default=300)
    ap.add_argument("--nb", type=int, default=30)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rows", type=int, default=8, help="sub-device 0 = rows 0..rows-1, sub-device 1 = the rest")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536, trace_region_size=64 << 20, num_command_queues=a.cqs)
    res = {}
    try:
        g = mesh.compute_with_storage_grid_size()
        res["grid"] = [g.x, g.y]
        print("cqs", a.cqs, "grid", g.x, g.y, flush=True)
        sd0 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, a.rows - 1))})
        sd1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, a.rows), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        mgr = mesh.create_sub_device_manager([ttnn.SubDevice([sd0]), ttnn.SubDevice([sd1])], 0)
        mesh.load_sub_device_manager(mgr)
        S0, S1 = ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)
        mesh.set_sub_device_stall_group([S0, S1])
        cqb = 1 if a.cqs > 1 else 0
        torch.manual_seed(0)
        xa = ttnn.from_torch(torch.randn(1, 1, 32, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        wa = [ttnn.from_torch(torch.randn(2048, 2048) * 0.02, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh) for _ in range(4)]
        xb = ttnn.from_torch(torch.rand(1, 1, 2048, 1024), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)

        def body_a():
            x = xa
            for i in range(a.na):
                y = ttnn.linear(x, wa[i % 4], core_grid=ttnn.CoreGrid(y=8, x=8), compute_kernel_config=ckc, dtype=ttnn.bfloat16)
                if x is not xa:
                    ttnn.deallocate(x)
                x = y
            return x

        def body_b():
            x = xb
            for i in range(a.nb):
                y = ttnn.sin(x, sub_core_grids=sd1)
                y2 = ttnn.multiply(y, x, sub_core_grids=sd1)
                ttnn.deallocate(y)
                if x is not xb:
                    ttnn.deallocate(x)
                x = y2
            return x

        oa = body_a()
        with ttnn.command_queue(cqb):  # the ops of B go to the queue of B (per-thread current queue)
            ob = body_b()
        for q in range(a.cqs):
            ttnn.synchronize_device(mesh, cq_id=q)
        ref_a, ref_b = ttnn.to_torch(oa), ttnn.to_torch(ob)
        ttnn.deallocate(oa)
        ttnn.deallocate(ob)
        for q in range(a.cqs):
            ttnn.synchronize_device(mesh, cq_id=q)
        ta = ttnn.begin_trace_capture(mesh, cq_id=0)
        oa = body_a()
        ttnn.end_trace_capture(mesh, ta, cq_id=0)
        tb = ttnn.begin_trace_capture(mesh, cq_id=cqb)
        with ttnn.command_queue(cqb):
            ob = body_b()
        ttnn.end_trace_capture(mesh, tb, cq_id=cqb)
        ttnn.synchronize_device(mesh)

        def sync():  # every queue (synchronize_device without cq_id waits for CQ 0 only)
            for q in range(a.cqs):
                ttnn.synchronize_device(mesh, cq_id=q)

        def run(seq):
            sync()
            t0 = time.perf_counter()
            for tid, cq in seq:
                ttnn.execute_trace(mesh, tid, cq_id=cq, blocking=False)
            sync()
            return (time.perf_counter() - t0) * 1e3

        modes = dict(
            a_alone=[(ta, 0)], b_alone=[(tb, cqb)],
            a_then_b=[(ta, 0), (tb, cqb)], b_then_a=[(tb, cqb), (ta, 0)],
        )
        tt = {m: [] for m in modes}
        for rep in range(a.reps + 1):
            for m, seq in modes.items():
                v = run(seq)
                if rep:
                    tt[m].append(v)
        for m, v in tt.items():
            v.sort()
            res[m] = dict(min_ms=round(v[0], 3), med_ms=round(v[len(v) // 2], 3))
        # serial on one queue by an explicit sync between them
        v = []
        for rep in range(a.reps):
            sync()
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, ta, cq_id=0, blocking=True)
            ttnn.execute_trace(mesh, tb, cq_id=cqb, blocking=True)
            v.append((time.perf_counter() - t0) * 1e3)
        v.sort()
        res["serial_synced"] = dict(min_ms=round(v[0], 3), med_ms=round(v[len(v) // 2], 3))
        sync()
        res["a_equal"] = bool(torch.equal(ref_a, ttnn.to_torch(oa)))
        res["b_equal"] = bool(torch.equal(ref_b, ttnn.to_torch(ob)))
        print(json.dumps(res), flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
        ttnn.release_trace(mesh, ta)
        ttnn.release_trace(mesh, tb)
        mesh.reset_sub_device_stall_group()
        mesh.clear_loaded_sub_device_manager()
        mesh.remove_sub_device_manager(mgr)
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

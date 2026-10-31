# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port10: the card codec window (CardCodec.decode_dev: quantizer + decoder) per window size: synced time per window
(quantizer, decoder, readback), a per-op profile of the decoder (each ttnn op synced, grouped by op and shape), and the
audio vs a reference file (bit-identical / SNR) so an opt-in lever can be checked against the default path.

    python -m models.demos.blackhole.drama3.tests.bench_codec10 [--save ref.pt | --ref ref.pt] [--profile] [--reps 7]
"""
import argparse
import collections
import json
import time

import torch

import ttnn


def snr_db(ref, x):
    err = (x - ref).double()
    return float(10 * torch.log10((ref.double() ** 2).sum() / (err**2).sum().clamp(min=1e-30)))


def profile(dec, z, out):
    """Each ttnn call in the decoder synced; ms per (op, detail)."""
    from models.demos.blackhole.drama3.tt import codec_device as cd

    mesh = dec.mesh
    acc = collections.defaultdict(lambda: [0.0, 0])
    names = ["conv1d", "linear", "multiply", "addcmul", "typecast", "to_layout", "reshape", "slice", "sharded_to_interleaved",
             "add", "tanh", "sin", "transpose", "generic_op"]
    orig = {}

    class Shim:
        def __getattr__(self, k):
            return getattr(ttnn, k)

    shim = Shim()
    stage = ["?"]

    def wrap(name, f):
        def g(*a, **kw):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            r = f(*a, **kw)
            ttnn.synchronize_device(mesh)
            dt = time.perf_counter() - t0
            if name == "conv1d":
                key = f"conv1d k={kw['kernel_size']} cin={kw['in_channels']} cout={kw['out_channels']} L={kw['input_length']} d={kw['dilation']}"
            elif name == "generic_op":
                key = f"snake_kernel {list(a[0][0].shape)}"
            elif name == "linear":
                key = f"linear {list(a[0].shape)}x{list(a[1].shape)}"
            else:
                t = a[0] if a else None
                key = f"{name} {list(t.shape) if hasattr(t, 'shape') else ''} {getattr(t, 'dtype', '')}"
            acc[key][0] += dt
            acc[key][1] += 1
            acc["[all ops]"][0] += dt
            acc["[all ops]"][1] += 1
            return r

        return g

    for n in names:
        setattr(shim, n, wrap(n, getattr(ttnn, n)))
    # stage labels: wrap the stage loop by counting stages via kind
    old = cd.ttnn
    cd.ttnn = shim
    try:
        x, _ = dec._body(z)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        a = dec.audio(x)
        acc["[readback to_torch]"][0] += time.perf_counter() - t0
        acc["[readback to_torch]"][1] += 1
    finally:
        cd.ttnn = old
    rows = sorted(acc.items(), key=lambda kv: -kv[1][0])
    out["profile"] = [(k, round(v[0] * 1e3, 2), v[1]) for k, v in rows]
    tot = sum(v[0] for k, v in acc.items() if k.startswith("[") and not k.startswith("[readback"))
    out["profile_synced_total_ms"] = round(tot * 1e3, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--cross", default=None)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--cfgs", default="def=", help="name=ATTR:val,ATTR:val/name2=... (tt/codec_device.py module flags)")
    ap.add_argument("--sizes", default="32,64,96")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.stream_codec import CardCodec

    frames = torch.load("/opt/ep0/day4b/out_v2/fix.frames.pt")  # [n, 11]
    codes_all = frames[:, 1:].T.contiguous().long()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536, trace_region_size=200 * 1024 * 1024)
    res = {}
    audios = {}
    try:
        t0 = time.time()
        cc = CardCodec(mesh)
        res["init_s"] = round(time.time() - t0, 1)
        cc.capture()
        from models.demos.blackhole.drama3.tt import codec_device as cdm

        cfgs = []  # (name, {module attr: value}); configs alternate rep by rep in this one process
        for spec in a.cfgs.split("/"):
            name, _, kv = spec.partition("=")
            ov = {}
            for item in [i for i in kv.split(",") if i]:
                k, _, v = item.partition(":")
                cur = getattr(cdm, k)
                ov[k] = (v == "1") if isinstance(cur, bool) else ([o for o in v.split("+") if o] if isinstance(cur, list) else type(cur)(v))
            cfgs.append((name, ov))
        base = {k: getattr(cdm, k) for _, ov in cfgs for k in ov}

        def setcfg(ov):
            for k, v in base.items():
                setattr(cdm, k, v)
            for k, v in ov.items():
                setattr(cdm, k, v)

        refs = torch.load(a.ref) if a.ref else None
        for w in [int(s) for s in a.sizes.split(",")]:
            n = w  # a full window of real codes (frames 0..w-1), kept from frame 0
            cd = codes_all[:, :n]
            tt = {nm: [] for nm, _ in cfgs}
            au = {}
            bad = set()
            for rep in range(a.reps + 1):  # rep 0: compile / warm-up of each config
                for nm, ov in cfgs:
                    if nm in bad:
                        continue
                    setcfg(ov)
                    ttnn.synchronize_device(mesh)
                    t2 = time.perf_counter()
                    try:
                        x = cc.decode_dev(cd, 0, n)  # quantizer trace + decoder + readback, as the generation loop sees it
                    except Exception as e:
                        print(w, nm, "FAILS:", str(e)[:300], flush=True)
                        bad.add(nm)
                        continue
                    t3 = time.perf_counter()
                    if rep:
                        tt[nm].append(t3 - t2)
                    else:
                        au[nm] = x.clone()
            res[w] = {}
            for nm, ov in cfgs:
                if nm in bad:
                    continue
                v = sorted(tt[nm])
                r = dict(min_ms=round(1e3 * v[0], 2), med_ms=round(1e3 * v[len(v) // 2], 2))
                r["med_ms_per_frame"] = round(r["med_ms"] / w, 3)
                ref = refs[w] if refs is not None and w in refs else au[cfgs[0][0]]
                r["bit_identical"] = bool(torch.equal(ref, au[nm]))
                r["max_abs"] = float((ref - au[nm]).abs().max())
                r["snr_db"] = round(snr_db(ref, au[nm]), 2)
                if a.profile:
                    setcfg(ov)
                    profile(cc.decs[w], cc.dq(cd, w), r)
                res[w][nm] = r
                print(w, nm, json.dumps({k: v for k, v in r.items() if k != "profile"}), flush=True)
                if a.profile:
                    for row in r["profile"][:45]:
                        print("   ", row, flush=True)
            if a.trace:  # each config's decoder as a trace on the quantizer trace's output (as DRAMA3_CODEC_DEV_TRACE)
                zq = cc.dq.traces[w][2]
                trs = {}
                for nm, ov in cfgs:
                    if nm in bad:
                        continue
                    setcfg(ov)
                    x, _ = cc.decs[w]._body(zq)
                    ttnn.deallocate(x)
                    ttnn.synchronize_device(mesh)
                    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                    out, _ = cc.decs[w]._body(zq)
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    ttnn.synchronize_device(mesh)
                    trs[nm] = (tid, out)
                ttr = {nm: [] for nm in trs}
                for rep in range(a.reps + 1):
                    for nm, ov in cfgs:
                        if nm not in trs:
                            continue
                        setcfg(ov)
                        ttnn.synchronize_device(mesh)
                        t2 = time.perf_counter()
                        cc.dq(cd, w)
                        ttnn.execute_trace(mesh, trs[nm][0], cq_id=0, blocking=False)
                        x = cc.decs[w].audio(trs[nm][1])[: n * cc.frame_length]
                        t3 = time.perf_counter()
                        if rep:
                            ttr[nm].append(t3 - t2)
                        else:
                            ref = refs[w] if refs is not None and w in refs else au[cfgs[0][0]]
                            res[w][nm]["trace_bit_identical"] = bool(torch.equal(ref, x))
                            res[w][nm]["trace_snr_db"] = round(snr_db(ref, x), 2)
                for nm, (tid, _) in trs.items():
                    v = sorted(ttr[nm])
                    res[w][nm]["trace_min_ms"] = round(1e3 * v[0], 2)
                    res[w][nm]["trace_med_ms"] = round(1e3 * v[len(v) // 2], 2)
                    print(w, nm, "trace", json.dumps({k: v for k, v in res[w][nm].items() if k.startswith("trace")}), flush=True)
                    ttnn.release_trace(mesh, tid)
            audios[w] = au[cfgs[0][0]]
        if a.cross:  # real samples of n frames from each window size >= n vs the default window (the smallest of 32/64/96)
            for n in [int(v) for v in a.cross.split(",")]:
                cd = codes_all[:, :n]
                outs = {}
                for w in cc.windows:
                    if w >= n:
                        outs[w] = cc.decs[w](cc.dq(cd, w))[: n * cc.frame_length].clone()
                w0 = next(w for w in (32, 64, 96) if w >= n)
                for w, x in outs.items():
                    print("cross n", n, "w", w, "vs w", w0, json.dumps(dict(bit_identical=bool(torch.equal(outs[w0], x)), snr_db=round(snr_db(outs[w0], x), 2))), flush=True)
        if a.save:
            torch.save(audios, a.save)
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

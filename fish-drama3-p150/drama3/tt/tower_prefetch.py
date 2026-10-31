# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAMA3_TOWER_PREFETCH=1 (port10, opt-in): the one-token tower decode takes its 6 x 13 matmul weights (eh_h,
q|gate|k|v, o_proj, w1, w3, w2) through the DRAM-core tensor prefetcher (port10/prefetch/README.md).

- Second weight copies in a receiver-contiguous DRAM layout (ND shards [K, N / 64], round robin over the 8 banks),
  made from the device weights' own values (bit-identical numbers; q|gate|k|v padded with zero tiles to 320,
  the attention pre kernel reads its tiles by index and never sees the pad). Prefill keeps the interleaved ones.
- Receivers = the matmul cores (0..7, 0..7); one 1D-mcast config per weight shape with num_global_cb_receivers 8;
  one global CB on those cores for all of them; the prefetcher is started once, right after the weights are built
  (before any trace capture) and stopped before the device closes (close_device / close_mesh_device wrapped).
- frame(capture): one request per tower frame for the 78 weights in call order; inside it, ttnn.linear calls on a
  one-row input whose weight is one of the 78 go to the prefetched matmul (same compute config as the call:
  tests/check_prefetch_bits.py: bit-identical for every in0_block_w, also LoFi + packer L1 acc). Outside a frame
  (prefill, batch rows, the heads) nothing changes. Every queued weight must be consumed exactly once, in order,
  or the stream desyncs and stop hangs: the wrapper checks the order and raises before a wrong matmul is queued
  (and drains the rest of the frame's stream with throw-away matmuls).
- NOT bit-identical to the flag off, although every prefetched matmul is: the GCB is a persistent L1 buffer, and the
  matmul / norm / reduction auto program configs size themselves from the lowest occupied L1 address
  (ttnn program_cache_l1.hpp), so the eager prefill programs change (port10: STAGE=2, GCB only, already differs).
Env: DRAMA3_TOWER_PREFETCH_GCB_KB (128: the live pipeline has only ~130 KB of L1 above the backbone ExpertMM CBs;
512 / 160 clash), DRAMA3_TOWER_PREFETCH_PAGE_KB (largest in0_block_w with an in1 page
<= this, default 35: port10 profile 5.54 ms vs 5.63 at 64), DRAMA3_TOWER_PREFETCH_IBW ("eh_h=64,qkvg=16,..." overrides).
"""

import contextlib
import math
import os

import torch

import ttnn

ENABLED = os.environ.get("DRAMA3_TOWER_PREFETCH", "0") == "1"
RING, ROWS, T = 64, 8, 32
TILE_B = {"BFLOAT16": 2048, "BFLOAT8_B": 1088, "BFLOAT4_B": 576}
NAMES = ("eh_h", "qkvg", "o_proj", "w1", "w3", "w2")

_active = None  # the TowerPrefetch whose frame is open
_running = []  # meshes with a started prefetcher
_wrapped = False
_orig_linear = None
_broken = False  # a frame raised: its stream was not consumed


def _dname(t):
    return str(t.dtype).split(".")[-1]


def _attn_qkv(attn):
    """The merged [q|gate | k | v] weight of attn_fused_decode's wrapper (DRAMA3_ATTN_QKV_MERGED), or None."""
    import inspect

    try:
        p = inspect.signature(attn.forward).parameters.get("_p")
    except (TypeError, ValueError):
        return None
    return None if p is None else getattr(p.default, "w_qkv", None)


class _Mat:
    def __init__(self, name, src, w, cfg):
        self.name, self.src, self.w, self.cfg = name, src, w, cfg


class TowerPrefetch:
    def __init__(self, mesh, layers, log=print):
        self.mesh = mesh
        nb = mesh.dram_grid_size().x
        assert nb * ROWS == RING, nb
        gcb_kb = int(os.environ.get("DRAMA3_TOWER_PREFETCH_GCB_KB", 128))
        page_kb = int(os.environ.get("DRAMA3_TOWER_PREFETCH_PAGE_KB", 35))
        ibw_env = dict(kv.split("=") for kv in os.environ.get("DRAMA3_TOWER_PREFETCH_IBW", "").split(",") if kv)
        dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})
        silu = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
        cfgs = {}

        def config(name, K, N, d):
            key = (name, K, N, d)
            if key not in cfgs:
                kt, pcn = K // T, N // T // RING
                page = pcn * TILE_B[d]
                ibw = int(ibw_env[name]) if name in ibw_env else max(
                    b for b in (128, 96, 64, 48, 32, 16, 8, 4, 2, 1) if kt % b == 0 and (b * page <= page_kb * 1024 or b == 1)
                )
                assert kt % ibw == 0 and 2 * ibw * page <= gcb_kb * 1024, (name, K, N, ibw, gcb_kb)
                cfgs[key] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(nb, ROWS), in0_block_w=ibw, out_subblock_h=1,
                    out_subblock_w=max(s for s in (4, 3, 2, 1) if pcn % s == 0), out_block_h=1, out_block_w=pcn,
                    per_core_M=1, per_core_N=pcn, fuse_batch=True, fused_activation=silu if name == "w1" else None,
                    mcast_in0=True, gather_in0=False, hop_cores=ttnn.CoreRangeSet([]), num_global_cb_receivers=ROWS,
                    untilize_out=False, stream_in1=False,
                )
            return cfgs[key]

        def copy(t):  # device weight -> receiver-contiguous copy of the same values, N padded to RING tiles
            K, N = int(t.shape[-2]), int(t.shape[-1])
            npad = math.ceil(N / T / RING) * RING * T
            x = ttnn.to_torch(t).float().reshape(K, N)
            if npad != N:
                x = torch.nn.functional.pad(x, (0, npad - N))
            spec = ttnn.NdShardSpec(ttnn.Shape([K, npad // RING]), dram, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D)
            return ttnn.from_torch(x.reshape(1, 1, K, npad), dtype=t.dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                                   memory_config=ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec)), K, npad

        self.order, self.map = [], {}
        for L in layers:
            qkv = _attn_qkv(L.attn)
            srcs = (L.w_h, qkv, L.attn.weights.o_proj, L.w1, L.w3, L.w2)
            for name, src in zip(NAMES, srcs):
                if src is None:
                    continue  # (no merged qkv: the q / k / v matmuls stay as they are)
                w, K, N = copy(src)
                m = _Mat(name, src, w, config(name, K, N, _dname(src)))
                m.K = K
                self.order.append(m)
                self.map[id(src)] = m
        # DRAMA3_TOWER_PREFETCH_STAGE (debug): 1 = weight copies only, 2 = + GCB, 3 = + started prefetcher, 4 = all
        self.stage = int(os.environ.get("DRAMA3_TOWER_PREFETCH_STAGE", 4))
        self.pos, self.frames = 0, 0
        if self.stage < 2:
            log(f"tower prefetch: stage {self.stage}, {len(self.order)} copies only")
            return
        b2r = [(b, ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, s), ttnn.CoreCoord(b, s)) for s in range(ROWS)])) for b in range(nb)]
        self.gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
            mesh, [m.cfg for m in self.order], [m.w for m in self.order], b2r, gcb_kb * 1024
        )
        self.req = [(m.w, ttnn.experimental.tensor_prefetcher_block_count_for_matmul_1d(m.cfg, m.w, self.gcb)) for m in self.order]
        _install()
        if self.stage >= 3:
            ttnn.experimental.start_tensor_prefetcher(mesh)
            _running.append(mesh)
        self.info = dict(
            matmuls=len(self.order), gcb_kb=gcb_kb,
            configs={k[0]: dict(K=k[1], N=k[2], ibw=c.in0_block_w, per_core_N=c.per_core_N, page_kb=round(c.in0_block_w * c.per_core_N * TILE_B[k[3]] / 1024, 1))
                     for k, c in cfgs.items()},
            dram_gb=round(sum(m.cfg.per_core_N * RING * m.K // T * TILE_B[_dname(m.src)] for m in self.order) / 1e9, 3),
        )
        log(f"tower prefetch: {self.info}")

    @contextlib.contextmanager
    def frame(self, capture=False):
        """One tower frame's matmuls; capture=True inside a trace capture (the request is replayed with it)."""
        global _active
        if self.stage < 4:
            yield
            return
        assert _active is None and self.pos == 0
        ttnn.experimental.queue_tensor_prefetcher_request(self.mesh, self.req, global_cb=self.gcb, capture_into_trace=capture)
        _active = self
        try:
            yield
        except BaseException:
            import traceback

            traceback.print_exc()  # now: a half-used stream makes the device close (auto stop) hang
            _active = None
            try:
                self.drain()
            except BaseException:
                traceback.print_exc()
                print("tower prefetch: frame failed and the drain too, the stream is out of step", flush=True)
                global _broken
                _broken = True
            raise
        finally:
            _active = None
        assert self.pos == len(self.order), f"tower prefetch: {self.pos} of {len(self.order)} weights consumed"
        self.pos = 0
        self.frames += 1

    def drain(self):
        """Consume the rest of the open frame's stream with throw-away matmuls (zero rows), so stop does not hang."""
        n = len(self.order) - self.pos
        print(f"tower prefetch: frame failed at {self.pos}, draining {n} weights", flush=True)
        zeros = {}
        while self.pos < len(self.order):
            m = self.order[self.pos]
            if m.K not in zeros:
                zeros[m.K] = ttnn.from_torch(torch.zeros(1, 1, m.K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)
            ttnn.deallocate(self._run(m, _orig_linear, zeros[m.K], {}))
        ttnn.synchronize_device(self.mesh)
        self.pos = 0
        print("tower prefetch: drained", flush=True)

    def _run(self, m, fn, x, kw):
        if self.pos >= len(self.order) or self.order[self.pos] is not m:
            want = self.order[self.pos].name if self.pos < len(self.order) else "none"
            raise RuntimeError(f"tower prefetch out of order at {self.pos}: {m.name}, expected {want}")
        self.pos += 1
        out = {k: kw[k] for k in ("compute_kernel_config", "memory_config", "dtype") if kw.get(k) is not None}
        return fn(x, m.w, program_config=m.cfg, global_cb=self.gcb, **out)


def _install():
    """Wrap ttnn.linear once: inside an open frame, a one-row call on a mapped weight takes the prefetched matmul
    (any program_config / activation of the call is replaced: w1's SiLU sits in its config)."""
    global _wrapped
    if _wrapped:
        return
    _wrapped = True
    global _orig_linear
    fn = _orig_linear = ttnn.linear

    def linear(*args, _fn=fn, **kw):
        pf = _active
        if pf is not None and len(args) >= 2 and isinstance(args[1], ttnn.Tensor):
            m = pf.map.get(id(args[1]))
            if m is not None and m.src is args[1] and args[0].shape[-2] == 1:
                return pf._run(m, _fn, args[0], kw)
        return _fn(*args, **kw)

    ttnn.linear = linear
    for name in ("close_device", "close_mesh_device"):
        close = getattr(ttnn, name)

        def closing(dev, *a, _close=close, **k):
            stop_all()
            return _close(dev, *a, **k)

        setattr(ttnn, name, closing)
    # exits without a device close: atexit stops it; SIGTERM (timeout) becomes SystemExit, so the callers' finally
    # blocks (device close -> stop) run. SIGKILL cannot be caught: the next device open finds the DRAM cores as left.
    import atexit
    import signal

    atexit.register(stop_all)
    if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, None):
        signal.signal(signal.SIGTERM, _sigterm)


def _sigterm(*_):
    raise SystemExit(143)


def stop_all():
    while _running:
        mesh = _running.pop()
        if _broken:
            print("tower prefetch: not stopping the prefetcher (stream out of step)", flush=True)
            continue
        ttnn.experimental.stop_tensor_prefetcher(mesh)
        ttnn.synchronize_device(mesh)


class SharedTower:
    """The same tower stream as group "tower" of the shared prefetcher (tt/weight_prefetch.py: ONE global CB and
    one prefetcher with the backbone's DRAMA3_BB_PREFETCH group; its requests are queued backbone, then tower, in
    trace execute order). Used with DRAMA3_BB_PREFETCH=1 (two GCBs do not fit the ~130 KB of L1) or
    DRAMA3_TOWER_PREFETCH_SHARED=1; else TowerPrefetch. Same weights, order, configs (DRAMA3_TOWER_PREFETCH_PAGE_KB /
    _IBW); the GCB size is DRAMA3_PREFETCH_GCB_KB, the start is lazy (the first frame of any group)."""

    def __init__(self, mesh, layers, log=print):
        from models.demos.blackhole.drama3.tt import weight_prefetch as wp

        page_kb = int(os.environ.get("DRAMA3_TOWER_PREFETCH_PAGE_KB", 35))
        ibw = dict(kv.split("=") for kv in os.environ.get("DRAMA3_TOWER_PREFETCH_IBW", "").split(",") if kv)
        self.pf = wp.get(mesh)
        self.g = self.pf.group("tower", page_kb=page_kb, ibw=ibw)
        for L in layers:
            srcs = (L.w_h, _attn_qkv(L.attn), L.attn.weights.o_proj, L.w1, L.w3, L.w2)
            for name, src in zip(NAMES, srcs):
                if src is not None:
                    self.g.add(src, name, silu=name == "w1")
        import atexit
        import signal

        atexit.register(wp.stop_all)
        if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, None):
            signal.signal(signal.SIGTERM, _sigterm)
        log(f"tower prefetch (shared, group tower): {len(self.g.order)} matmuls, {self.g.dram_bytes() / 1e9:.3f} GB, "
            f"GCB {self.pf.gcb_kb} KB, pages <= {page_kb} KB, groups {list(self.pf.groups)}")

    def frame(self, capture=False):
        return self.g.frame(capture)

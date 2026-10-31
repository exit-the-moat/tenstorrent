# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One DRAM-core tensor prefetcher per card for the one-token decode matmuls (port10, opt-in users:
DRAMA3_BB_PREFETCH=1 in tt/backbone_paged.py, the tower's DRAMA3_TOWER_PREFETCH=1). Recipe: port10/prefetch/README.md.

- Prefetcher.get(mesh): one per mesh. group(name): an ordered list of weights consumed by one traced body (the
  backbone decode, the tower frame). Every group adds its weights (Group.add) before the first frame of any group;
  the first Group.frame() builds ONE global CB on the receiver cores (0..7, 0..7) from every group's configs and
  starts the prefetcher (adding a weight after that raises).
- Group.add(src, name): a second copy of the device weight in the receiver-contiguous DRAM layout (ND shards
  [K, N / 64] round robin over the 8 banks, from the device values: the same numbers; N padded with zero tiles to
  a multiple of 64 tiles, read by index by the kernels after it, so they never see the pad) and its 8x8 1D-mcast
  config (num_global_cb_receivers 8, in0_block_w = the largest divisor of K tiles with an in1 page <= page_kb).
- Group.frame(capture): queues the group's request (captured into the open trace with capture=True: every replay
  re-sends it) and, while it is open, routes each one-row ttnn.linear call on one of the group's source weights to
  the prefetched matmul (the call's compute config / memory config / dtype kept; its program config replaced).
  Every queued weight must be consumed exactly once, in order, or the stream desyncs and stop hangs: an
  out-of-order call raises before it is queued, a short frame raises at its end.
- Requests are queued in trace execute order (backbone, then tower) and each trace consumes exactly its own list,
  so the one stream stays in order however the host interleaves other traces (embed, codec) between them.
Env: DRAMA3_PREFETCH_GCB_KB (global CB per receiver core, default 128: with the other L1 buffers it must fit above
every program's static CBs on those cores, and ExpertMM's reach ~1.35 MB of the ~1.52 MB on core (0, 0); 160 fails in
a full session), DRAMA3_PREFETCH_PAGE_KB (largest in1 page, default min(128, GCB / 2)), DRAMA3_PREFETCH_NICE (nice
value of the prefetcher's host thread, default 10; 0 = unchanged).
"""

import contextlib
import math
import os

import torch

import ttnn

RING, ROWS, T = 64, 8, 32
TILE_B = {"BFLOAT16": 2048, "BFLOAT8_B": 1088, "BFLOAT4_B": 576}

_by_mesh = {}
_wrapped = False
_linear = None  # ttnn.linear under the wrapper


def _dname(t):
    return str(t.dtype).split(".")[-1]


def supported(mesh):
    return ttnn.experimental.is_tensor_prefetcher_supported(mesh) and mesh.dram_grid_size().x * ROWS == RING


def batch_main():
    """True when the running program is host/batch.py or host/cbatch.py (python -m ...): the users stay off there."""
    import sys

    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return (getattr(spec, "name", "") or "").rsplit(".", 1)[-1] in ("batch", "cbatch")


def attn_qkv(attn):
    """The merged [q|gate | k | v] weight of attn_fused_decode's wrapper (DRAMA3_ATTN_QKV_MERGED), or None."""
    import inspect

    try:
        p = inspect.signature(attn.forward).parameters.get("_p")
    except (TypeError, ValueError):
        return None
    return None if p is None else getattr(p.default, "w_qkv", None)


class _Entry:
    def __init__(self, group, name, src, w, cfg, K):
        self.group, self.name, self.src, self.w, self.cfg, self.K = group, name, src, w, cfg, K


class Group:
    def __init__(self, pf, name, page_kb=None, ibw=None):
        page_kb = page_kb or int(os.environ.get("DRAMA3_PREFETCH_PAGE_KB", min(128, pf.gcb_kb // 2)))
        self.pf, self.name, self.page_kb, self.ibw = pf, name, page_kb, dict(ibw or {})
        self.order, self.req, self.pos, self.frames = [], None, 0, 0

    def add(self, src, name, silu=False):
        """Register the next weight this group's frame consumes (call order). -> the entry."""
        pf = self.pf
        assert not pf.started, f"prefetch group {self.name}: {name} added after the prefetcher started"
        assert id(src) not in pf.map, f"prefetch group {self.name}: {name} registered twice"
        K, N, d = int(src.shape[-2]), int(src.shape[-1]), _dname(src)
        npad = math.ceil(N / T / RING) * RING * T
        x = ttnn.to_torch(src).float().reshape(K, N)
        if npad != N:
            x = torch.nn.functional.pad(x, (0, npad - N))
        spec = ttnn.NdShardSpec(ttnn.Shape([K, npad // RING]), pf.dram, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D)
        w = ttnn.from_torch(x.reshape(1, 1, K, npad), dtype=src.dtype, layout=ttnn.TILE_LAYOUT, device=pf.mesh,
                            memory_config=ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec))
        e = _Entry(self, name, src, w, pf.config(K, npad, d, silu, self.ibw.get(name), self.page_kb), K)
        self.order.append(e)
        pf.map[id(src)] = e
        return e

    @contextlib.contextmanager
    def frame(self, capture=False):
        """One traced body's matmuls; capture=True inside a trace capture (the request is replayed with it)."""
        pf = self.pf
        if not pf.started:
            pf.start()
        assert pf.active is None and self.pos == 0, (self.name, pf.active and pf.active.name, self.pos)
        ttnn.experimental.queue_tensor_prefetcher_request(pf.mesh, self.req, global_cb=pf.gcb, capture_into_trace=capture)
        pf.active = self
        try:
            yield
        except BaseException:
            pf.active = None
            self._drain(capture)
            raise
        pf.active = None
        assert self.pos == len(self.order), f"prefetch group {self.name}: {self.pos} of {len(self.order)} weights consumed"
        self.pos = 0
        self.frames += 1

    def _drain(self, capture):
        """After an error inside a frame: consume the rest of the queued request with zero-row matmuls, so the stream
        stays in order and stop does not hang (inside a trace capture nothing ran: the prefetcher is marked broken)."""
        pf = self.pf
        try:
            if capture:
                raise RuntimeError("error inside a captured frame")
            for e in self.order[self.pos:]:
                x = ttnn.from_torch(torch.zeros(1, 1, e.K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=pf.mesh)
                ttnn.deallocate(_linear(x, e.w, program_config=e.cfg, global_cb=pf.gcb))
            ttnn.synchronize_device(pf.mesh)
        except Exception as err:  # noqa: BLE001
            pf.broken = True
            print(f"weight_prefetch: group {self.name} could not drain at {self.pos} ({err}); the prefetcher is not stopped", flush=True)
        self.pos = 0

    def _run(self, e, fn, x, kw):
        if self.pos >= len(self.order) or self.order[self.pos] is not e:
            want = self.order[self.pos].name if self.pos < len(self.order) else "none"
            raise RuntimeError(f"prefetch group {self.name} out of order at {self.pos}: {e.name}, expected {want}")
        self.pos += 1
        out = {k: kw[k] for k in ("compute_kernel_config", "memory_config", "dtype") if kw.get(k) is not None}
        return fn(x, e.w, program_config=e.cfg, global_cb=self.pf.gcb, **out)

    def dram_bytes(self):
        return sum(e.cfg.per_core_N * RING * (e.K // T) * TILE_B[_dname(e.src)] for e in self.order)


class Prefetcher:
    @classmethod
    def get(cls, mesh):
        pf = _by_mesh.get(id(mesh))
        if pf is None:
            pf = _by_mesh[id(mesh)] = cls(mesh)
        return pf

    def __init__(self, mesh):
        assert supported(mesh), "tensor prefetcher not supported on this device"
        self.mesh = mesh
        self.nb = mesh.dram_grid_size().x
        self.dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.nb - 1, 0))})
        self.gcb_kb = int(os.environ.get("DRAMA3_PREFETCH_GCB_KB", 128))
        self.groups, self.map, self.cfgs = {}, {}, {}
        self.started, self.gcb, self.active, self.broken = False, None, None, False
        # the global CB must sit at the top of L1, above every program's static CBs on the receiver cores (ExpertMM's
        # reach ~1.35 MB on core (0, 0)): a placeholder of its size holds that range from the first registration
        # (the backbone build, before the tower / codec allocate L1) until start() frees it and creates the CB there
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.nb - 1, ROWS - 1))])
        w = self.gcb_kb * 1024 // (T * 2)
        assert w % T == 0, self.gcb_kb
        spec = ttnn.ShardSpec(crs, [T, w], ttnn.ShardOrientation.ROW_MAJOR)
        self.hold = ttnn.allocate_tensor_on_device(
            ttnn.Shape([RING * T, w]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec),
        )
        _install()

    def group(self, name, **kw):
        if name not in self.groups:
            assert not self.started, f"prefetch group {name} created after the prefetcher started"
            self.groups[name] = Group(self, name, **kw)
        return self.groups[name]

    def config(self, K, N, d, silu, ibw, page_kb):
        kt, pcn = K // T, N // T // RING
        page = pcn * TILE_B[d]
        if ibw is None:
            ibw = max(b for b in (128, 96, 64, 48, 32, 16, 8, 4, 2, 1) if kt % b == 0 and (b * page <= page_kb * 1024 or b == 1))
        ibw = int(ibw)
        assert kt % ibw == 0 and 2 * ibw * page <= self.gcb_kb * 1024, (K, N, d, ibw, self.gcb_kb)
        key = (K, N, d, silu, ibw)
        if key not in self.cfgs:
            self.cfgs[key] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(self.nb, ROWS), in0_block_w=ibw, out_subblock_h=1,
                out_subblock_w=max(s for s in (4, 3, 2, 1) if pcn % s == 0), out_block_h=1, out_block_w=pcn,
                per_core_M=1, per_core_N=pcn, fuse_batch=True,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if silu else None,
                mcast_in0=True, gather_in0=False, hop_cores=ttnn.CoreRangeSet([]), num_global_cb_receivers=ROWS,
                untilize_out=False, stream_in1=False,
            )
        return self.cfgs[key]

    def start(self):
        """One global CB for every group's matmuls, block counts per group, then start the prefetcher."""
        assert not self.started
        ents = [e for g in self.groups.values() for e in g.order]
        assert ents, "prefetcher started without weights"
        b2r = [(b, ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, s), ttnn.CoreCoord(b, s)) for s in range(ROWS)])) for b in range(self.nb)]
        hold_addr = self.hold.buffer_address()
        ttnn.deallocate(self.hold)
        self.hold = None
        self.gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
            self.mesh, [e.cfg for e in ents], [e.w for e in ents], b2r, self.gcb_kb * 1024
        )
        for g in self.groups.values():
            g.req = [(e.w, ttnn.experimental.tensor_prefetcher_block_count_for_matmul_1d(e.cfg, e.w, self.gcb)) for e in g.order]
        self.hold_addr = hold_addr  # (the GCB is created right after it is freed: same top-of-L1 range)
        tasks = set(os.listdir("/proc/self/task"))
        ttnn.experimental.start_tensor_prefetcher(self.mesh)
        self.started = True
        # its host worker thread spin-yields while the request sockets are back-pressured (most of every frame):
        # DRAMA3_PREFETCH_NICE (default 10) lowers that thread's priority so the frame loop's own threads win
        nice = int(os.environ.get("DRAMA3_PREFETCH_NICE", "10"))
        self.threads = sorted(int(t) for t in set(os.listdir("/proc/self/task")) - tasks)
        for tid in self.threads if nice else ():
            try:
                os.setpriority(os.PRIO_PROCESS, tid, nice)
            except OSError:
                pass

    def stop(self):
        if self.started and not self.broken:  # (a broken stream would hang stop)
            self.started = False
            ttnn.experimental.stop_tensor_prefetcher(self.mesh)
            ttnn.synchronize_device(self.mesh)

    def info(self):
        return dict(
            gcb_kb=self.gcb_kb, l1_reserved_at=getattr(self, "hold_addr", None), host_threads=getattr(self, "threads", None),
            groups={n: dict(matmuls=len(g.order), dram_gb=round(g.dram_bytes() / 1e9, 3)) for n, g in self.groups.items()},
            configs=[dict(K=k[0], N=k[1], dtype=k[2], silu=k[3], ibw=k[4], per_core_N=c.per_core_N,
                          page_kb=round(k[4] * c.per_core_N * TILE_B[k[2]] / 1024, 1)) for k, c in self.cfgs.items()],
        )


def get(mesh):
    return Prefetcher.get(mesh)


def _install():
    """Wrap ttnn.linear once: inside an open group frame, a one-row call on one of its weights takes the prefetched
    matmul. Wrap close_device / close_mesh_device: the prefetcher stops before the device closes."""
    global _wrapped, _linear
    if _wrapped:
        return
    _wrapped = True
    fn = _linear = ttnn.linear

    def linear(*args, _fn=fn, **kw):
        if len(args) >= 2 and isinstance(args[1], ttnn.Tensor):
            for pf in _by_mesh.values():
                g = pf.active
                if g is None:
                    continue
                e = pf.map.get(id(args[1]))
                if e is not None and e.src is args[1] and e.group is g and args[0].shape[-2] == 1:
                    return g._run(e, _fn, args[0], kw)
        return _fn(*args, **kw)

    ttnn.linear = linear
    # a timeout / kill (SIGTERM) or a plain exit still stops the prefetcher (an unstopped one can leave the card busy)
    import atexit
    import signal

    atexit.register(stop_all)
    if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, None):
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    for name in ("close_device", "close_mesh_device"):
        close = getattr(ttnn, name)

        def closing(dev, *a, _close=close, **k):
            stop_all()
            return _close(dev, *a, **k)

        setattr(ttnn, name, closing)


def stop_all():
    for pf in list(_by_mesh.values()):
        pf.stop()
    _by_mesh.clear()

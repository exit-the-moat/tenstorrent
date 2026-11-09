# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Modified for Exit the Moat Episode 2 (2026-11-02): new file, port of Episode 0's drama3 tt/weight_prefetch.py.
"""QWEN36_PF=1 (default off): the DRAM-core tensor prefetcher streams the one-token decode weights into one global
circular buffer while the previous ops run; the decode matmuls read in1 from it instead of from interleaved DRAM.

Why: the BF4 decode matmuls read interleaved 576-byte tiles at ~280 of 512 GB/s on any core grid. The prefetcher's
DRAM-core (DRISC) senders read large contiguous blocks of a receiver-contiguous copy and push them to an 8 x rpb
receiver grid. Card 0 unit tests: gate/up 90 -> 57 us,
4-layer frame of 20 matmuls 1570 -> 1066 us; outputs bit-identical to the 1D path (the copy is built from the
device's own bfp4 values).

How (this build, tt-metal of the 0.20.0 image):
  - Prefetcher.get(mesh) (created before the layers: it reserves the top of L1 on the receiver cores with a
    placeholder of the global CB's size, so later L1 buffers sit below it), register(src, silu) per decode weight in
    execute order (model.py after the layers are built), frame() around the decode layer loop (model._forward_decode).
  - register: a second copy of the weight, ND-sharded [K, Npad / ring] round robin over the DRAM banks, from the
    device values (QWEN36_PF_COPY=device, default: typecast to bf16 -> zero-pad N to a multiple of ring tiles ->
    typecast back -> reshard, all on the card; =host: per device to_torch -> pad -> from_torch, ~2 s per weight), plus
    its 1D mcast_in0 program config (grid 8 x rpb, num_global_cb_receivers = rpb, in0_block_w = the largest divisor of the
    K tiles whose in1 page fits QWEN36_PF_PAGE_KB and divides the global CB: this build requires gcb.size % page == 0).
  - frame(): queues the frame's request (one entry per weight, in order). Inside a trace capture:
    QWEN36_PF_REQ=host (default) queues nothing then; the wrapped ttnn.execute_trace queues the request on the host
    right before every replay of that trace (the trace holds only the matmuls). =trace: the request is queued during
    the capture and so captured into the trace (this build has no capture_into_trace kwarg). While the frame is open,
    ttnn.linear calls on a registered weight with <= 32 rows take the prefetched matmul (compute_kernel_config,
    memory_config, dtype kept; program config replaced). Every queued weight must be consumed exactly once, in order,
    or the stream desyncs and stop hangs: out of order raises before the matmul is issued; an error inside an eager
    frame drains the rest with zero matmuls.
Env: QWEN36_PF_RPB (receivers per DRAM bank, ring = 8 x rpb, default 5), QWEN36_PF_GCB_KB (L1 budget of the global CB per receiver
core, default 400 (ring 40: a 387 KB GCB); the size is the largest multiple of every page's lcm within it; must leave room for every
program's CBs on those cores),
QWEN36_PF_PAGE_KB (largest in1 page, default 128), QWEN36_PF_ONLY (kinds to stream: mixer_in, mixer_out, mlp),
QWEN36_PF_NICE (prefetcher host threads, default 10), QWEN36_PF_COPY (device | host), QWEN36_PF_REQ (host | trace),
QWEN36_PF_GCB_MODE (resident: the GCB lives from the first frame on; decode: freed before every prefill by
release(), recreated at the same L1 address by the next frame / decode replay, for paths whose prefill needs the L1).
"""
import contextlib
import math
import os
import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import decode_mode as _dm

T = 32
TILE_B = {"BFLOAT16": 2048, "BFLOAT8_B": 1088, "BFLOAT4_B": 576}
# Stage A1: the rows a prefetcher frame serves. Every registered entry carries a program config built for ONE tile
# row, so a frame streams only <= PF_MAX_ROWS-row matmuls. A decode step wider than that (a DFlash2 verify at 8+
# streams) must run with the prefetcher off (QWEN36_PF=0) or with QWEN36_PF_MAX_ROWS raised, once Stage A2 has a
# 2+-tile-row GCB config. The flag already routes such a frame; the limit is the program, not the branch.
PF_MAX_ROWS = int(os.environ.get("QWEN36_PF_MAX_ROWS", str(T)))

_by_mesh = {}
_wrapped = False
_linear = None  # ttnn.linear under the wrapper
_capturing = {}  # mesh key -> trace id while a trace capture is open (QWEN36_PF_REQ=host)


def _key(mesh):
    """A stable key per mesh: the Python wrapper objects differ between callers (vLLM's Generator passes
    model_args.mesh_device, not model.mesh_device), so id() alone does not match; there is one prefetcher per process."""
    try:
        return ("mesh", int(mesh.id()))
    except Exception:  # noqa: BLE001
        return ("one", 0)


def _tk(tid):
    """A hashable key for a trace id (an int or a pybind MeshTraceId)."""
    try:
        return int(tid)
    except (TypeError, ValueError):
        return getattr(tid, "id", tid)


def pcn_to(p):
    """df1c QWEN36_PF_PCN_TO="a:b,...": ring N tiles per receiver core a -> b (default off)."""
    m = os.environ.get("QWEN36_PF_PCN_TO", "")
    if not m:
        return p
    d = dict((int(a), int(b)) for a, b in (kv.split(":") for kv in m.split(",") if ":" in kv))
    return d.get(int(p), int(p))


def enabled():
    return os.environ.get("QWEN36_PF", "0") == "1"


def _dname(t):
    return str(t.dtype).split(".")[-1]


class _Entry:
    def __init__(self, name, src, w, cfg, K, ndev):
        self.name, self.src, self.w, self.cfg, self.K, self.ndev = name, src, w, cfg, K, ndev


class Prefetcher:
    @classmethod
    def get(cls, mesh):
        pf = _by_mesh.get(_key(mesh))
        if pf is None:
            pf = _by_mesh[_key(mesh)] = cls(mesh)
        return pf

    def __init__(self, mesh):
        assert ttnn.experimental.is_tensor_prefetcher_supported(mesh), "QWEN36_PF: tensor prefetcher not supported"
        self.mesh = mesh
        self.nb = mesh.dram_grid_size().x
        self.rows = int(os.environ.get("QWEN36_PF_RPB", "5"))  # ring 40: N = 5120 (160 tiles) needs no pad / slice
        self.ring = self.nb * self.rows
        # the global CB's L1 budget per receiver core; its exact size is chosen in start() (a multiple of every page)
        self.gcb_max = int(os.environ.get("QWEN36_PF_GCB_KB", "400")) * 1024 // 2048 * 2048
        self.gcb_bytes = None
        self.page_kb = int(os.environ.get("QWEN36_PF_PAGE_KB", "128"))
        self.dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.nb - 1, 0))})
        self.order, self.map, self.cfgs = [], {}, {}
        self.started, self.gcb, self.req, self.active, self.pos, self.broken = False, None, None, False, 0, False
        self.frames = 0
        self.in_prefill = False
        self.recording, self.seen = None, {}
        self.req_mode = os.environ.get("QWEN36_PF_REQ", "host")
        self.copy_mode = os.environ.get("QWEN36_PF_COPY", "device")
        # ep2 r64w QWEN36_PF_RING_ONLY=1 (default off): the ring copy becomes the ONLY resident copy of the decode
        # weights. adopt() swaps every owning module's weight for its ring entry and frees the interleaved DRAM copy
        # after registration. Prefill reads the ring directly: the 2D prefill matmul, the fused all-gather matmul
        # (all_gather_minimal_matmul_async) and minimal_matmul all address in1 through the TensorAccessor, which
        # resolves the NdShardSpec layout (bit-identical to the interleaved
        # weight at M=2048 for gate/up, qkv, gdn_in, gdn_out and w2 incl. K padding; padded columns exactly zero).
        self.ring_only = os.environ.get("QWEN36_PF_RING_ONLY", "0") == "1" and enabled()
        self.trace_frames = {}  # trace id -> frames captured in it (QWEN36_PF_REQ=host)
        # ep2 host agent, QWEN36_PF_PREQ=1 (default off, decode mode only): right after a decode replay is enqueued, queue
        # the NEXT replay's request too (the host call costs 0.2-0.7 ms per token and then overlaps the card; the
        # prefetcher can start the next step's first weights as soon as the GCB has room, as in back-to-back replays).
        # `pending` = requests queued ahead and not yet consumed. A replay / eager frame consumes a pending one instead
        # of queueing; release() (prefill) and stop() first consume what is pending with the eager zero-input frame
        # that _warm compiled (same programs: nothing compiles next to the parked traces).
        self.preq = os.environ.get("QWEN36_PF_PREQ", "0") == "1" and os.environ.get("QWEN36_PF_GCB_MODE", "decode") == "decode"
        self.pending = 0
        self.drained = 0
        # pf6 QWEN36_PF_PREQ_HEAD=K (default 0 = the whole next request ahead): queue only the first K weights of the
        # next replay's request ahead, the rest (the tail) right before that replay. The weight stream is the same,
        # in the same order (the GCB holds only a few weights ahead anyway); a prefill then drains only those K
        # weights (a small zero-frame trace) instead of streaming every weight once (~20 ms).
        self.head = int(os.environ.get("QWEN36_PF_PREQ_HEAD", "0")) if self.preq else 0
        # placeholder of the global CB's size at the top of L1 on the receiver cores (freed right before the GCB is
        # created in start(), which then lands in the same range)
        self.gcb_mode = os.environ.get("QWEN36_PF_GCB_MODE", "decode")
        # resident: a placeholder holds the top of L1 from now on. decode: no placeholder (L1 buffers that later decode
        # code allocates would land below it, in the range prefill's CBs need: 2-card server 16:35); the GCB's place is
        # taken at the first prefetched frame and must be free again at every ensure()
        self.hold, self.hold_addr = None, None
        # lazy: like decode (no placeholder, first eager frame skipped, created at the first frame) but never released:
        # the GCB stays resident, so replayed decode traces always see the same GCB (decode mode's release + recreate
        # broke traces captured before a prefill: 64-layer replay after a prefill diverged at step 9, max abs 34.9)
        if self.gcb_mode not in ("decode", "lazy"):
            self.hold = self._hold_alloc()
            self.hold_addr = self.hold.buffer_address()
        # decode mode: the first QWEN36_PF_SKIP eager decode frames run without the prefetcher, so whatever the decode
        # path allocates lazily (persistent L1) is allocated before the GCB, above it
        self.skip = int(os.environ.get("QWEN36_PF_SKIP", "1")) if self.gcb_mode in ("decode", "lazy") else 0
        if self.ring_only:  # r64x: the skipped frame runs plain matmuls, and a plain 1D matmul cannot read the DRAM ring
            self.skip = 0
        _install()

    def _hold_alloc(self, extra=0):
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.nb - 1, self.rows - 1))])
        w = ((self.gcb_bytes or self.gcb_max) + extra) // (T * 2) // T * T
        spec = ttnn.ShardSpec(crs, [T, w], ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.ring * T, w]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec),
        )

    def _top_free(self, extra=0):
        """True when the placeholder's L1 range (the global CB's place), plus `extra` bytes below it, is free: a new
        placeholder of that size lands exactly there."""
        h = self._hold_alloc(extra)
        a = h.buffer_address()
        ttnn.deallocate(h)
        return a == self.hold_addr - extra, a

    def _choose(self):
        """in0_block_w per distinct (K, N, dtype) and the GCB size: this build needs gcb.size % page == 0 for every
        matmul's in1 page (in0_block_w x per_core_N tiles), so the GCB is a multiple of the pages' lcm. A small search
        (per shape the divisors of its K tiles with a page <= QWEN36_PF_PAGE_KB) picks the combination with the largest
        GCB <= QWEN36_PF_GCB_KB that holds 2 pages of each, preferring pages of >= 24 KB."""
        shapes = sorted({(e.K, e.npad, e.d) for e in self.order})
        forced = {(e.K, e.npad, e.d): e.ibw for e in self.order if getattr(e, "ibw", None)}
        states = {1: ((), 1 << 60)}  # lcm -> (ibw per shape, smallest page)
        for K, npad, d in shapes:
            kt, pcn, tb = K // T, npad // T // self.ring, TILE_B[d]
            cand = [b for b in range(kt, 0, -1) if kt % b == 0 and 2 * b * pcn * tb <= self.gcb_max
                    and b * pcn * tb <= self.page_kb * 1024]
            # QWEN36_PF_IBW=match (default): the 1D path's in0_block_w (largest divisor of the K tiles <= 8), i.e. the
            # same K blocking -> the same fp32 accumulation order (other blockings differed in rare roundings: 64
            # layers, 4k, logits off by up to 1.5 from step 19); =free: any block width (larger pages)
            if os.environ.get("QWEN36_PF_IBW", "match") == "match":
                cand = [max(d for d in range(1, 9) if kt % d == 0)]
            if (K, npad, d) in forced:
                cand = [forced[(K, npad, d)]]
            nxt = {}
            for L, (ch, mn) in states.items():
                for b in cand:
                    pg = b * pcn * tb
                    L2 = L * pg // math.gcd(L, pg)
                    if L2 > self.gcb_max:
                        continue
                    v = (ch + (b,), min(mn, pg))
                    if L2 not in nxt or nxt[L2][1] < v[1]:
                        nxt[L2] = v
            states = nxt
            if not states:
                raise ValueError(f"QWEN36_PF: no global CB <= {self.gcb_max} B fits the pages of {shapes}: raise QWEN36_PF_GCB_KB or change QWEN36_PF_RPB")
        best = None
        for L, (ch, mn) in states.items():
            g = self.gcb_max // L * L
            pages = [b * (npad // T // self.ring) * TILE_B[d] for b, (K, npad, d) in zip(ch, shapes)]
            score = (min(mn, 24 * 1024), g, mn)  # pages of >= 24 KB first (small pages are slow for the DRISC sender)
            if g >= 2 * max(pages) and (best is None or score > best[3]):
                best = (g, ch, mn, score)
        if best is None:
            raise ValueError(f"QWEN36_PF: no global CB <= {self.gcb_max} B holds 2 pages of every shape {shapes}")
        self.gcb_bytes = best[0]
        return dict(zip(shapes, best[1]))

    def config(self, K, npad, d, silu, ibw):
        pcn = npad // T // self.ring
        key = (K, npad, d, silu, ibw)
        if key not in self.cfgs:
            self.cfgs[key] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(self.nb, self.rows), in0_block_w=ibw, out_subblock_h=1,
                out_subblock_w=max(s for s in (4, 3, 2, 1) if pcn % s == 0), out_block_h=1, out_block_w=pcn,
                per_core_M=1, per_core_N=pcn, fuse_batch=True,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if silu else None,
                mcast_in0=True, gather_in0=False, hop_cores=ttnn.CoreRangeSet([]), num_global_cb_receivers=self.rows,
                untilize_out=False, stream_in1=False,
            )
        return self.cfgs[key]

    def ring_mc(self, dim):
        """ep2 QWEN36_AR_RING: a [*, dim] activation width-sharded on the receiver ring (row-major), as the
        prefetched matmuls with N = dim produce it."""
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.nb - 1, self.rows - 1))])
        assert dim % (self.ring * T) == 0, (dim, self.ring)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                                 ttnn.ShardSpec(crs, [T, dim // self.ring], ttnn.ShardOrientation.ROW_MAJOR))

    def register(self, src, name, silu=False, pad_k=None, slice_out=True, out_mc=None):
        """The next weight the decode frame consumes (execute order). N is zero-padded to a multiple of ring tiles;
        slice_out=False leaves the matmul output that wide (its consumer slices by offset, or it feeds a weight
        registered with pad_k = this N padded: zero rows, so the extra columns add nothing)."""
        assert not self.started, f"QWEN36_PF: {name} registered after the prefetcher started"
        assert id(src) not in self.map, f"QWEN36_PF: {name} registered twice"
        K, N, d = int(src.shape[-2]), int(src.shape[-1]), _dname(src)
        kp = pad_k or K
        npad = pcn_to(math.ceil(N / T / self.ring)) * self.ring * T  # df1c QWEN36_PF_PCN_TO
        spec = ttnn.NdShardSpec(ttnn.Shape([kp, npad // self.ring]), self.dram, ttnn.ShardOrientation.ROW_MAJOR,
                                ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D)
        mc = ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec)
        ndev = len(ttnn.get_device_tensors(src))
        if self.copy_mode == "device":
            w = self._copy_device(src, K, N, kp, npad, mc)
        else:
            w = self._copy_host(src, K, N, kp, npad, mc)
        e = _Entry(name, src, w, None, kp, ndev)
        e.N, e.npad, e.d, e.silu, e.slice_out = N, npad, d, silu, slice_out and npad != N
        e.out_mc = out_mc  # (QWEN36_AR_RING: the output layout, whatever the caller asked for)
        assert out_mc is None or npad == N, (name, N, npad)
        self.order.append(e)
        self.map[id(src)] = e
        self.map[id(w)] = e  # QWEN36_PF_RING_ONLY: the frame is handed the ring tensor once adopt() swapped it in
        return e

    def adopt(self, model, layers=None, quiet=False):
        """QWEN36_PF_RING_ONLY: make the ring copy the module's weight and free the interleaved DRAM copy.

        The registered entries already hold the values (the copy was built from the device's own bfp4 values), so
        nothing re-uploads. Every owning reference is swapped: layer.feed_forward.weights.{w1,w3,w2} and the
        attention dict tw[{wqkv_fused,wo,qkvz,out}]; the LM head keeps its own vocab-sharded copy (it is used by
        the eager/demo logits paths outside the frame)."""
        assert self.ring_only and not self.started, "QWEN36_PF_RING_ONLY: adopt after registration, before start"
        # r64z: QWEN36_PF_RING_ONLY_CLASSES selects which weight classes adopt (bisect; default all).
        # Classes: gate_up (w1,w3), w2, qkv (wqkv_fused), wo, gdn_in (qkvz), gdn_out (out).
        _sel = os.environ.get("QWEN36_PF_RING_ONLY_CLASSES", "all").strip()
        _only = None if _sel in ("", "all") else {c.strip() for c in _sel.split(",") if c.strip()}
        dead, swapped, skipped = [], 0, {}

        def take(owner, key, src, tag):
            nonlocal swapped
            if _only is not None and tag not in _only:
                skipped[tag] = skipped.get(tag, 0) + 1
                return
            e = self.map.get(id(src)) if src is not None else None
            if e is None or e.src is not src:
                return
            if isinstance(owner, dict):
                owner[key] = e.w
            else:
                object.__setattr__(owner, key, e.w)  # MLPWeights is a frozen dataclass
            dead.append(src)
            swapped += 1

        for _li, layer in enumerate(model.layers if layers is None else layers):
            w = layer.feed_forward.weights
            for nm, tag in (("w1", "gate_up"), ("w3", "gate_up"), ("w2", "w2")):
                _src = getattr(w, nm, None)
                _e = self.map.get(id(_src)) if _src is not None else None
                if os.environ.get("R64Y_DBG") == "1" and _li == 0 and _e is not None:
                    try:
                        print(f"R64Z_SHAPE {tag}.{nm} src={tuple(_src.shape)} ring={tuple(_e.w.shape)} K={_e.K}"
                              f" npad={_e.npad} src_dev={[tuple(t.shape) for t in ttnn.get_device_tensors(_src)]}"
                              f" ring_dev={[tuple(t.shape) for t in ttnn.get_device_tensors(_e.w)]}", flush=True)
                        if tag in ("w2", "gate_up"):
                            _dr, _ds = ttnn.get_device_tensors(_e.w), ttnn.get_device_tensors(_src)
                            for _di in range(len(_dr)):
                                _a = ttnn.to_torch(ttnn.to_memory_config(_dr[_di], ttnn.DRAM_MEMORY_CONFIG)).float()
                                _b = ttnn.to_torch(_ds[_di]).float()
                                _a = _a.reshape(-1, _a.shape[-1]); _b = _b.reshape(-1, _b.shape[-1])
                                Kb, Nb = _b.shape
                                _d = (_a[:Kb, :Nb] - _b).abs()
                                _kpad = _a[Kb:, :Nb] if _a.shape[0] > Kb else None
                                _npad = _a[:Kb, Nb:] if _a.shape[1] > Nb else None
                                print(f"R64Z_VAL {tag}.{nm} dev{_di} ring={tuple(_a.shape)} il={tuple(_b.shape)}"
                                      f" ndiff={int((_d > 0).sum())} maxabs={float(_d.max())}"
                                      f" kpad_max={float(_kpad.abs().max()) if _kpad is not None else 0.0}"
                                      f" npad_max={float(_npad.abs().max()) if _npad is not None else 0.0}", flush=True)
                    except Exception as _x:  # noqa: BLE001
                        print(f"R64Z_SHAPE_ERR {tag}.{nm} {repr(_x)[:220]}", flush=True)
                take(w, nm, _src, tag)
            at = layer.attention
            if model.tp:
                tw = at.tw
                for key, tag in (("wqkv_fused", "qkv"), ("wo", "wo"), ("qkvz", "gdn_in"), ("out", "gdn_out")):
                    if key in tw:
                        take(tw, key, tw[key], tag)
            else:
                for nm, tag in (("q_proj", "qkv"), ("k_proj", "qkv"), ("v_proj", "qkv"), ("o_proj", "wo"),
                                ("mega_fused_weight", "qkv"), ("o_proj_weight", "wo")):
                    if hasattr(at.weights, nm):
                        take(at.weights, nm, getattr(at.weights, nm), tag)
        for t in dead:
            ttnn.deallocate(t)
        self._adopted = getattr(self, "_adopted", 0) + swapped
        if quiet:
            return
        print(f"QWEN36_PF_RING_ONLY: total {self._adopted}; {swapped} module weights now read the ring, {len(dead)} interleaved copies freed,"
              f" classes={_sel} skipped={skipped}", flush=True)

    def register_lm_head(self, src, split, ibw=2):
        """ep2 QWEN36_PF_LMHEAD=1: the LM head [K, N] (vocab-sharded per device) as len(split) prefetched matmuls, piece i
        = split[i] output tiles per receiver core (the last piece zero-padded to that width, its output sliced back).
        Why pieces: one LM head matmul has 97 tiles per core at ring 40 (3880 tiles per device), and a page of 97 x ibw
        tiles never divides the global CB (this build needs gcb.size % page == 0); 7 x 14 (= 98) pages divide the
        387,072-byte GCB of the layer weights, a K row of a piece must fit the DRAM-core stage third (23,616 B: <= 41 tiles),
        and a K block (ibw x 14 tiles = 16 KB) within it is streamed without K-sub-splits (4 layers: 7 x 14 2.471 ms,
        16 x 6 + 1 2.513, 4 x 28 2.609, 8 x 12 + 1 2.551). ibw 2 = the in0_block_w of the plain ttnn.linear's auto 1D config
        (get_mcast_1d_config: 2 when the K tiles are even), so the K accumulation and the values stay the same."""
        assert not self.started, "QWEN36_PF: LM head registered after the prefetcher started"
        K, N, d = int(src.shape[-2]), int(src.shape[-1]), _dname(src)
        nt = N // T
        assert N % T == 0 and sum(split) * self.ring >= nt, (N, split, self.ring)
        x = ttnn.typecast(src, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if len(x.shape) != 4:
            x = ttnn.reshape(x, (1, 1, K, N))
        ents, n0 = [], 0
        for i, pcn in enumerate(split):
            npad = pcn * self.ring * T
            n1 = min(N, n0 + npad)
            xs = ttnn.slice(x, [0, 0, 0, n0], [1, 1, K, n1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if n1 - n0 != npad:
                xp = ttnn.pad(xs, padding=[(0, 0), (0, 0), (0, 0), (0, npad - (n1 - n0))], value=0.0)
                ttnn.deallocate(xs)
                xs = xp
            y = ttnn.typecast(xs, src.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(xs)
            spec = ttnn.NdShardSpec(ttnn.Shape([K, npad // self.ring]), self.dram, ttnn.ShardOrientation.ROW_MAJOR,
                                    ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D)
            w = ttnn.to_memory_config(y, ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec))
            ttnn.deallocate(y)
            key = object()  # stands for the piece in self.map / self.seen (no source tensor of its own)
            e = _Entry(f"lm_head.{i}", key, w, None, K, len(ttnn.get_device_tensors(src)))
            e.N, e.npad, e.d, e.silu, e.slice_out, e.ibw, e.n0 = n1 - n0, npad, d, False, npad != n1 - n0, ibw, n0
            self.order.append(e)
            self.map[id(key)] = e
            ents.append(e)
            n0 = n1
        ttnn.deallocate(x)
        assert n0 == N, (n0, N)
        self.lm_entries = ents
        return ents

    def lm_head(self, x, weight):
        """The LM head inside a frame: the pieces through the prefetcher (or, in a frame that does not prefetch, the plain
        matmul sliced into the same pieces: the eager frame compiles the same downstream programs), each untilized to
        row major (padding rows dropped, as QWEN36_COMPACT_LOGITS does for the whole), then concatenated."""
        ents = self.lm_entries
        if self.active:
            parts = [self.run(e, _linear, x, {}) for e in ents]
        else:
            if self.recording is not None:
                for e in ents:
                    self.recording[id(e.src)] = (list(x.shape), x.memory_config(), x.dtype, {})
            full = ttnn.linear(x, weight)
            sh = list(full.shape)
            parts = [ttnn.slice(full, [0] * (len(sh) - 1) + [e.n0], sh[:-1] + [e.n0 + e.N], memory_config=full.memory_config())
                     for e in ents]
            ttnn.deallocate(full)
        rm = []
        for p in parts:
            rm.append(ttnn.to_layout(p, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.deallocate(p)
        out = ttnn.concat(rm, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for r in rm:
            ttnn.deallocate(r)
        return out

    @staticmethod
    def _copy_device(src, K, N, kp, npad, mc):
        """On the card, per device: bfp -> bf16 (exact), zero-pad K / N, bf16 -> the same bfp (exact: the values and
        their 16-value exponent groups are unchanged; the zero pad forms its own groups), reshard."""
        x = ttnn.typecast(src, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if npad != N or kp != K:
            r = len(x.shape)
            xp = ttnn.pad(x, padding=[(0, 0)] * (r - 2) + [(0, kp - K), (0, npad - N)], value=0.0)
            ttnn.deallocate(x)
            x = xp
        y = ttnn.typecast(x, src.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        if len(y.shape) != 4:
            y = ttnn.reshape(y, (1, 1, kp, npad))
        w = ttnn.to_memory_config(y, mc)
        ttnn.deallocate(y)
        return w

    def _copy_host(self, src, K, N, kp, npad, mc):
        xs = []
        for p in ttnn.get_device_tensors(src):
            x = ttnn.to_torch(p).float().reshape(K, N)
            xs.append(torch.nn.functional.pad(x, (0, npad - N, 0, kp - K)))
        return ttnn.from_torch(torch.stack(xs).reshape(len(xs), 1, kp, npad), dtype=src.dtype, layout=ttnn.TILE_LAYOUT,
                               device=self.mesh, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0), memory_config=mc)

    def start(self):
        assert not self.started and self.order, "QWEN36_PF: nothing registered"
        ibw = self._choose()
        for e in self.order:
            e.cfg = self.config(e.K, e.npad, e.d, e.silu, ibw[(e.K, e.npad, e.d)])
        b2r = [(b, ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, s), ttnn.CoreCoord(b, s)) for s in range(self.rows)]))
               for b in range(self.nb)]
        self.b2r = b2r
        if self.hold is not None:
            ttnn.deallocate(self.hold)
            self.hold = None
            self._make_gcb()
        else:  # decode mode: the GCB's place = where a placeholder of its size lands now (checked again by ensure)
            h = self._hold_alloc()
            self.hold_addr = h.buffer_address()
            ttnn.deallocate(h)
            self._make_gcb()
        self._launch()
        self.started = True
        self._warm()

    def _warm(self):
        """Compile every prefetched matmul program once, eagerly, with zero inputs (one full request): a decode trace
        capture must not be the first use of a program (compiling writes to the device: TT_FATAL inside a capture)."""
        self.queue()
        self._zero_frame()
        ttnn.synchronize_device(self.mesh)
        # pf4 QWEN36_PF_DRAIN_TRACE=1: the same zero-input frame as a trace (its zero inputs are made inside it), so the
        # PF_PREQ drain before every prefill is a replay instead of ~320 eagerly dispatched matmuls (~24 ms of host).
        # Captured here, at the first start (warm-up), right after the eager frame compiled every program; captures
        # consume no request (nothing executes), replays consume one each, as the eager frame does.
        self.zf_tid = None
        if self.preq and os.environ.get("QWEN36_PF_DRAIN_TRACE", "0") == "1" and _capturing.get(_key(self.mesh)) is None:
            # the zero inputs are persistent (ttnn.zeros writes from the host: not allowed inside a capture); their values
            # do not matter (the frame only consumes the streamed weights, its outputs are dropped)
            self.zf_inputs = {}
            self.queue()  # the eager frame that allocates the inputs consumes one request
            self._zero_frame(keep=self.zf_inputs)
            ttnn.synchronize_device(self.mesh)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self._zero_frame(keep=self.zf_inputs)
            finally:
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            ttnn.synchronize_device(self.mesh)
            self.zf_tid = tid
            print("QWEN36_PF_DRAIN_TRACE: zero frame captured", flush=True)
            if self.head:  # pf6 PREQ_HEAD: the first `head` weights only (the same persistent inputs: nothing allocated)
                self.head = min(self.head, len(self.order))
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                try:
                    self._zero_frame(keep=self.zf_inputs, limit=self.head)
                finally:
                    ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                ttnn.synchronize_device(self.mesh)
                self.zfh_tid = tid
                print(f"QWEN36_PF_PREQ_HEAD: head zero frame captured ({self.head} of {len(self.order)} weights)", flush=True)

    def _zero_frame(self, keep=None, limit=None):
        """One full request consumed by the prefetched matmuls with zero inputs (the recorded activation shapes).
        keep (pf4 QWEN36_PF_DRAIN_TRACE): a dict of persistent zero inputs, filled on first use and not freed."""
        trim = os.environ.get("QWEN36_HOST_TRIM", "0") == "1" or keep is not None
        zs = keep if keep is not None else {}  # (A32) the frame-local zero inputs, freed at the end of the frame
        for ei, e in enumerate(self.order if limit is None else self.order[:limit]):
            shape, mc, dt, kw = self.seen.get(id(e.src), ([1, 1, T, e.K], ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16, {}))
            shape = list(shape[:-1]) + [e.K]  # (w2: K padded)
            if trim:  # ep2 t3c QWEN36_HOST_TRIM: zeros made on the card, one per input shape for the whole frame (A32)
                zk = (tuple(shape), str(dt), str(mc) if not mc.is_sharded() else "L1")
                x = zs.get(zk)
                if x is None:
                    x = zs[zk] = ttnn.zeros(shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG if keep is not None  # persistent: not in L1
                                            else (mc if not mc.is_sharded() else ttnn.L1_MEMORY_CONFIG))
            else:
                x = ttnn.from_torch(torch.zeros(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                    memory_config=mc if not mc.is_sharded() else ttnn.L1_MEMORY_CONFIG,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
            self.pos = ei if trim else self.order.index(e)
            ttnn.deallocate(self.run(e, _linear, x, kw))
            if not trim:
                ttnn.deallocate(x)
        if keep is None:
            for x in zs.values():
                ttnn.deallocate(x)
        self.pos = 0

    def drain_pending(self):
        """QWEN36_PF_PREQ: consume the requests queued ahead (eager, between frames, never inside a capture)."""
        if not self.pending:
            return
        assert not self.active and _capturing.get(_key(self.mesh)) is None, "QWEN36_PF_PREQ: drain inside a frame/capture"
        n, self.pending = self.pending, 0
        for _ in range(n):
            if self.head:  # pf6: only the head of a request is ever queued ahead
                if getattr(self, "zfh_tid", None) is not None:
                    ttnn.execute_trace(self.mesh, self.zfh_tid, cq_id=0, blocking=False)
                else:
                    self._zero_frame(limit=self.head)
            elif getattr(self, "zf_tid", None) is not None:
                ttnn.execute_trace(self.mesh, self.zf_tid, cq_id=0, blocking=False)
            else:
                self._zero_frame()
        ttnn.synchronize_device(self.mesh)
        self.drained += n

    def _launch(self):
        tasks = set(os.listdir("/proc/self/task"))
        ttnn.experimental.start_tensor_prefetcher(self.mesh)
        self.running = True
        nice = int(os.environ.get("QWEN36_PF_NICE", "10"))
        self.threads = sorted(int(t) for t in set(os.listdir("/proc/self/task")) - tasks)
        for tid in self.threads if nice else ():
            try:
                os.setpriority(os.PRIO_PROCESS, tid, nice)
            except OSError:
                pass

    def _make_gcb(self):
        self.gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
            self.mesh, [e.cfg for e in self.order], [e.w for e in self.order], self.b2r, self.gcb_bytes)
        bc = {}
        self.req = []
        for e in self.order:
            k = id(e.cfg)
            if k not in bc:
                bc[k] = ttnn.experimental.tensor_prefetcher_block_count_for_matmul_1d(e.cfg, e.w, self.gcb)
            self.req.append((e.w, bc[k]))

    def release(self):
        """QWEN36_PF_GCB_MODE=decode: free the global CB before a prefill (its programs need that L1). Between frames
        only, every queued request consumed (synchronize). The prefetcher is stopped too: once running it holds a small
        L1 buffer of its own just below the GCB (2-card server, 16:35: a prefill CB clash at hold - 2304 B with the
        GCB freed and the prefetcher idle). Raises if the GCB range did not come free; notes other L1 within 64 KB below it."""
        if self.gcb_mode == "lazy":
            self.in_prefill = True
            return
        if self.gcb_mode != "decode" or self.broken:
            return
        self.in_prefill = True  # until the next decode frame / replay
        if self.hold is not None:  # not started yet: the placeholder goes too (it is an L1 buffer like the GCB)
            ttnn.deallocate(self.hold)
            self.hold = None
            return
        if self.gcb is None:
            return
        assert not self.active, "QWEN36_PF: release inside a frame"
        _tq = [time.perf_counter()]
        self.drain_pending()
        _tq.append(time.perf_counter())
        ttnn.synchronize_device(self.mesh)
        _tq.append(time.perf_counter())
        self.gcb, self.req = None, None
        import gc

        _t0 = time.perf_counter()
        if os.environ.get("QWEN36_PF_NO_GC", "0") != "1":  # t3c: the global CB is freed by refcount; gc.collect is the slow part?
            gc.collect()
        self.release_t = {"gc": round(1000 * (time.perf_counter() - _t0), 1)}
        if getattr(self, "running", False):
            ttnn.experimental.stop_tensor_prefetcher(self.mesh)
            ttnn.synchronize_device(self.mesh)
            self.running = False
        _tq.append(time.perf_counter())
        self.releases = getattr(self, "releases", 0) + 1
        ok, a = self._top_free()
        _tq.append(time.perf_counter())
        self.release_t.update(drain=round(1000 * (_tq[1] - _tq[0]), 1), sync=round(1000 * (_tq[2] - _tq[1]), 1),
                              stop=round(1000 * (_tq[3] - _tq[2]), 1), topfree=round(1000 * (_tq[4] - _tq[3]), 1))
        if not ok:
            raise RuntimeError(f"QWEN36_PF: global CB not freed (placeholder lands at {a}, not {self.hold_addr})")
        ok, a = self._top_free(64 * 1024)
        if not ok and self.releases <= 3:  # diagnostics: some other L1 buffer sits within 64 KB below the GCB
            print(f"QWEN36_PF: note: an L1 buffer within 64 KB below the global CB after release (placeholder of +64 KB "
                  f"at {a}, free would be {self.hold_addr - 64 * 1024})", flush=True)

    def ensure(self, launch=True):
        """Recreate the global CB after a release, at the same L1 address the decode traces were captured with, and
        restart the prefetcher."""
        self.in_prefill = False
        if self.gcb is None:
            ok, a = self._top_free()
            if not ok:
                raise RuntimeError(f"QWEN36_PF: the global CB's L1 range is taken (placeholder at {a}, traces use {self.hold_addr})")
            self._make_gcb()
        if launch and not getattr(self, "running", False):
            self._launch()

    def prepare(self):
        """Before a trace capture (eager): start the prefetcher, or recreate a released GCB."""
        if self.in_prefill:  # a prefill trace capture (decode mode): the GCB stays freed
            return
        self.skip = 0
        if not self.started:
            self.start()
        else:
            self.ensure()

    @contextlib.contextmanager
    def frame(self):
        """One decode step's weight matmuls (all registered weights, in order)."""
        if self.skip > 0 and _capturing.get(_key(self.mesh)) is None:
            self.skip -= 1  # (see __init__: lazy decode allocations first)
            self.recording = {}  # the calls' activation shapes / kwargs, for _warm (program cache keys)
            try:
                yield
            finally:
                self.seen, self.recording = self.recording, None
            return
        if not self.started:
            self.start()
        self.ensure()
        assert not self.active and self.pos == 0 and not self.broken, (self.active, self.pos, self.broken)
        tid = _capturing.get(_key(self.mesh))
        if tid is not None and self.req_mode == "host":
            self.trace_frames[tid] = self.trace_frames.get(tid, 0) + 1  # (tid already a key)  # queued by execute_trace at every replay
        elif self.pending:
            self.pending -= 1  # QWEN36_PF_PREQ: this frame's request is already queued
            if self.head:  # pf6: only its head; the tail now
                ttnn.experimental.queue_tensor_prefetcher_request(self.mesh, self.req[self.head:], global_cb=self.gcb)
        else:
            self.queue()
        self.active = True
        try:
            yield
        except BaseException:
            self.active = False
            self._drain()
            raise
        self.active = False
        n = self.pos
        self.pos = 0
        assert n == len(self.order), (
            f"QWEN36_PF: {n} of {len(self.order)} weights consumed in a frame "
            f"(prefetcher max {PF_MAX_ROWS} rows, decode step={_dm.active()}, max_rows={_dm.max_rows()})"
        )
        self.frames += 1

    def queue(self, n=1):
        ttnn.experimental.queue_tensor_prefetcher_request(self.mesh, self.req * n, global_cb=self.gcb)

    def _drain(self):
        """After an error inside a frame: consume the rest of the request with zero-row matmuls so the stream stays in
        order (inside a trace capture this cannot help: the prefetcher is marked broken and never stopped)."""
        try:
            for e in self.order[self.pos:]:
                x = ttnn.from_torch(torch.zeros(e.ndev, 1, T, e.K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=self.mesh, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0))
                ttnn.deallocate(_linear(x, e.w, program_config=e.cfg, global_cb=self.gcb))
            ttnn.synchronize_device(self.mesh)
        except Exception as err:  # noqa: BLE001
            self.broken = True
            print(f"QWEN36_PF: could not drain at {self.pos} ({err}); the prefetcher is not stopped", flush=True)
        self.pos = 0

    def run(self, e, fn, x, kw):
        if self.pos >= len(self.order) or self.order[self.pos] is not e:
            want = self.order[self.pos].name if self.pos < len(self.order) else "none"
            raise RuntimeError(f"QWEN36_PF: out of order at {self.pos}: {e.name}, expected {want}")
        rank = len(x.shape)
        if x.is_sharded():
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        # (an activation kwarg is never passed on: a sharded program config rejects it; SiLU is fused at registration)
        out = {k: kw[k] for k in ("compute_kernel_config", "memory_config", "dtype") if kw.get(k) is not None}
        if getattr(e, "out_mc", None) is not None:
            out["memory_config"] = e.out_mc
        y = fn(x, e.w, program_config=e.cfg, global_cb=self.gcb, **out)
        self.pos += 1  # only once issued: a matmul that raises (host-side validation) leaves its entry to _drain
        if os.environ.get("QWEN36_PF_KEEP_RANK") == "1" and len(y.shape) != rank:
            # the GCB matmul returns 4-D for a 3-D activation (the 1D path keeps 3-D); DFlash2's verify slices by rank
            ys = list(y.shape)
            y = ttnn.reshape(y, ys[len(ys) - rank:])
        if e.slice_out:
            sh = list(y.shape)
            ys = ttnn.slice(y, [0] * len(sh), sh[:-1] + [e.N], memory_config=y.memory_config())
            ttnn.deallocate(y)
            y = ys
        return y

    def stop(self):
        if getattr(self, "running", False) and not self.broken:
            try:
                self.drain_pending()
            except Exception as err:  # noqa: BLE001
                print(f"QWEN36_PF_PREQ: drain before stop failed ({err})", flush=True)
            self.running = False
            ttnn.experimental.stop_tensor_prefetcher(self.mesh)
            ttnn.synchronize_device(self.mesh)

    def info(self):
        return dict(
            ring=self.ring, hold_addr=self.hold_addr, preq=self.preq, head=self.head, pending=self.pending, drained=self.drained, gcb_mode=self.gcb_mode, releases=getattr(self, "releases", 0), recaptures=getattr(self, "recaptures", 0), gcb_bytes=self.gcb_bytes, gcb_max=self.gcb_max, weights=len(self.order), frames=self.frames,
            dram_gb_per_device=round(sum(e.npad * e.K * TILE_B[e.d] / 1024 for e in self.order) / 1e9, 3),
            sliced=sum(e.slice_out for e in self.order), ring_only=self.ring_only,
            configs=[dict(K=k[0], N=k[1], dtype=k[2], silu=k[3], ibw=k[4], per_core_N=c.per_core_N) for k, c in self.cfgs.items()],
        )


def get(mesh):
    return Prefetcher.get(mesh)


def register_model(model):
    """Register every decode weight matmul in execute order, per layer the mixer's in- and out-projections, then the
    MLP w1 (SiLU fused), w3, w2.
    TP (2+ cards): GDN tw["qkvz"], tw["out"]; attention tw["wqkv_fused"], tw["wo"] (the 1D decode paths of
    model_config, the defaults). 1 card: needs the fused decode kernels (QWEN36_GDN_FUSED_DECODE=1: mega in-projection
    + o_proj; QWEN36_ATTN_FUSED=1: q / k / v / o_proj); on 1 card the copy doubles the weights' DRAM, so only for a
    truncated model (tests)."""
    args = model.args
    pf = get(model.mesh_device)
    # QWEN36_PF_ONLY (checks): comma list of the kinds to stream (mixer_in, mixer_out, mlp); default all
    only = set(os.environ.get("QWEN36_PF_ONLY", "mixer_in,mixer_out,mlp").split(","))
    reg = pf.register

    # QWEN36_PF_WIDE=0 (default in decode mode): every output sliced back to its real N (no N-padded tensors downstream,
    # so the skipped eager frame compiles every downstream op with the shapes the capture sees); =1: wide outputs
    wide = os.environ.get("QWEN36_PF_WIDE", "0" if pf.gcb_mode in ("decode", "lazy") else "1") == "1"

    # ep2 QWEN36_AR_RING=1: the row-parallel outputs (N = dim) straight into the ring residual layout
    rmc = pf.ring_mc(args.dim) if os.environ.get("QWEN36_AR_RING") == "1" and model.tp else None

    def register(src, name, kind, keep_wide=False, **kw):
        if not wide and not keep_wide:
            kw["slice_out"], kw["pad_k"] = True, None
        return reg(src, name, **kw) if kind in only else None

    for i, layer in enumerate(model.layers):
        at = layer.attention
        if model.tp:
            assert getattr(args, "proj_1d_decode", False) and getattr(args, "mlp_1d_decode", False), "QWEN36_PF needs the 1D decode matmuls"
            if layer.is_full_attention:
                register(at.tw["wqkv_fused"], f"L{i}.qkv", "mixer_in", slice_out=False,
                         keep_wide=os.environ.get("QWEN36_PF_QKV_WIDE") == "1")  # sliced by offset after
                register(at.tw["wo"], f"L{i}.wo", "mixer_out", out_mc=rmc)
            else:
                assert getattr(at, "_fuse_ab", True), "QWEN36_PF: GDN a/b not fused into the in-projection"
                register(at.tw["qkvz"], f"L{i}.gdn_in", "mixer_in", slice_out=False, keep_wide=os.environ.get("QWEN36_PF_GDNIN_WIDE", "1") == "1")  # read by offset
                # (always wide: N = 8240 is not tile-aligned; suspected: a slice back to it changes the last tile padding, which
                # the fused GDN kernel may read; server ep2-pf-a with the slice: exact 2/5 vs the reference, pf2 wide: bit-equal)
                register(at.tw["out"], f"L{i}.gdn_out", "mixer_out", out_mc=rmc)
        elif layer.is_full_attention:
            assert getattr(at, "_fused", None) is not None, "QWEN36_PF on 1 card needs QWEN36_ATTN_FUSED=1"
            for nm, kind in (("q_proj", "mixer_in"), ("k_proj", "mixer_in"), ("v_proj", "mixer_in"), ("o_proj", "mixer_out")):
                register(getattr(at.weights, nm), f"L{i}.{nm}", kind)
        else:
            assert getattr(at, "_fused1", None) is not None, "QWEN36_PF on 1 card needs QWEN36_GDN_FUSED_DECODE=1"
            register(at.weights.mega_fused_weight, f"L{i}.gdn_in", "mixer_in")
            register(at.weights.o_proj_weight, f"L{i}.gdn_out", "mixer_out")
        w = layer.feed_forward.weights
        # gate / up outputs stay N-padded (SiLU(0) * 0 = 0); w2 gets the same number of zero K rows
        # ep2 QWEN36_PF_MLP_WIDE=1: gate / up outputs stay N-padded in decode mode too (no slice per matmul; the MLP
        # pads them the same way in a frame that does not prefetch), w2 reads them with zero K rows
        mw = os.environ.get("QWEN36_PF_MLP_WIDE") == "1"
        e1 = register(w.w1, f"L{i}.w1", "mlp", keep_wide=mw, silu=True, slice_out=False)  # (1 card: replaces activation="silu")
        register(w.w3, f"L{i}.w3", "mlp", keep_wide=mw, slice_out=False)
        register(w.w2, f"L{i}.w2", "mlp", keep_wide=mw, pad_k=e1.npad if e1 is not None else None, out_mc=rmc)
        # df1c QWEN36_PF_ADOPT_EACH=1 (ring-only): free the interleaved copies of this layer right away (1 card: the
        # full model plus all ring copies do not fit at once)
        if pf.ring_only and os.environ.get("QWEN36_PF_ADOPT_EACH") == "1":
            pf.adopt(model, [layer], quiet=True)
    # ep2 QWEN36_PF_LMHEAD=1 (default off; needs QWEN36_COMPACT_LOGITS=1, a vocab-sharded LM head): the LM head too, last
    # in the frame (model._forward_decode runs the final norm and the LM head inside it); QWEN36_PF_LMHEAD_SPLIT = the
    # pieces' output tiles per receiver core
    if os.environ.get("QWEN36_PF_LMHEAD") == "1":
        assert os.environ.get("QWEN36_COMPACT_LOGITS") == "1" and model._lmhead_vocab_sharded, "QWEN36_PF_LMHEAD needs QWEN36_COMPACT_LOGITS=1 on a mesh"
        split = [int(v) for v in os.environ.get("QWEN36_PF_LMHEAD_SPLIT", "14,14,14,14,14,14,14").split(",")]
        pf.register_lm_head(model.lm_head_weight, split, ibw=int(os.environ.get("QWEN36_PF_LMHEAD_IBW", "2")))
    if pf.ring_only:
        pf.adopt(model)
    return pf


def _install():
    """Wrap ttnn.linear once: inside an open frame, a <= 32-row call on a registered weight takes the prefetched
    matmul. Wrap close_device / close_mesh_device and atexit / SIGTERM: the prefetcher stops before the device closes."""
    global _wrapped, _linear
    if _wrapped:
        return
    _wrapped = True
    fn = _linear = ttnn.linear

    def linear(*args, _fn=fn, **kw):
        if len(args) >= 2 and isinstance(args[1], ttnn.Tensor):
            for pf in _by_mesh.values():
                if not pf.active:
                    continue
                e = pf.map.get(id(args[1]))
                if e is not None and (e.src is args[1] or e.w is args[1]) and args[0].shape[-2] <= PF_MAX_ROWS and _dm.decode_rows(args[0].shape[-2], "pf.frame"):
                    return pf.run(e, _fn, args[0], kw)
            for pf in _by_mesh.values():
                e = pf.map.get(id(args[1]))
                if pf.recording is not None and e is not None and args[0].shape[-2] <= T:
                    # keyed by the ENTRY (id(e.src)): _zero_frame looks it up that way, and with QWEN36_PF_RING_ONLY
                    # the caller passes e.w, not e.src
                    pf.recording[id(e.src)] = (list(args[0].shape), args[0].memory_config(), args[0].dtype, dict(kw))
        return _fn(*args, **kw)

    ttnn.linear = linear

    # QWEN36_PF_REQ=host: remember which trace a frame was captured in; queue its requests before every replay
    begin, end, execute, release = ttnn.begin_trace_capture, ttnn.end_trace_capture, ttnn.execute_trace, ttnn.release_trace

    def begin_trace_capture(mesh, *a, **k):
        # nothing may write to the device during a capture: start the prefetcher / recreate the GCB before it
        pf = _by_mesh.get(_key(mesh))
        if pf is not None and pf.order and not pf.broken:
            pf.prepare()
        tid = begin(mesh, *a, **k)
        _capturing[_key(mesh)] = _tk(tid)
        return tid

    def end_trace_capture(mesh, tid, *a, **k):
        _capturing.pop(_key(mesh), None)
        return end(mesh, tid, *a, **k)

    def execute_trace(mesh, tid, *a, **k):
        pf = _by_mesh.get(_key(mesh))
        n = pf.trace_frames.get(_tk(tid), 0) if pf is not None else 0
        if n:
            pf.ensure()
            tail = None
            if pf.head:  # pf6 PREQ_HEAD: at most one head queued ahead; the rest of this replay's requests right
                if pf.pending:  # after the replay is enqueued (the card streams the head meanwhile), as PREQ does
                    pf.pending = 0
                    tail = pf.req[pf.head:] + pf.req * (n - 1)
                    if k.get("blocking", a[1] if len(a) > 1 else True):  # a blocking replay would wait on it
                        ttnn.experimental.queue_tensor_prefetcher_request(mesh, tail, global_cb=pf.gcb)
                        tail = None
                else:
                    pf.queue(n)
            elif pf.pending >= n:
                pf.pending -= n  # QWEN36_PF_PREQ: queued after the previous replay
            else:
                pf.queue(n - pf.pending)
                pf.pending = 0
        r = execute(mesh, tid, *a, **k)
        if n and pf.head:
            nxt = pf.preq and not pf.in_prefill
            if tail or nxt:  # this replay's tail + the next replay's head, in one host call
                ttnn.experimental.queue_tensor_prefetcher_request(
                    mesh, (tail or []) + (pf.req[:pf.head] if nxt else []), global_cb=pf.gcb)
            pf.pending = 1 if nxt else 0
        elif n and pf.preq and not pf.in_prefill:
            pf.queue(n)  # the next replay's request, while this one runs
            pf.pending += n
        return r

    def release_trace(mesh, tid, *a, **k):
        pf = _by_mesh.get(_key(mesh))
        if pf is not None:
            pf.trace_frames.pop(_tk(tid), None)
        return release(mesh, tid, *a, **k)

    ttnn.begin_trace_capture, ttnn.end_trace_capture = begin_trace_capture, end_trace_capture
    ttnn.execute_trace, ttnn.release_trace = execute_trace, release_trace
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


def wide_n(mesh, src):
    """ep2 QWEN36_PF_MLP_WIDE: the padded output width of a registered weight that is not sliced back, else None."""
    pf = _by_mesh.get(_key(mesh))
    e = pf.map.get(id(src)) if pf is not None else None
    return e.npad if e is not None and not e.slice_out and e.npad != e.N else None


def recreate_pending():
    """True when a started prefetcher's GCB is freed (decode mode, after a prefill): the decode traces must be
    re-captured after it is recreated."""
    return any(pf.started and pf.gcb is None and not pf.broken for pf in _by_mesh.values())


def recreated():
    for pf in list(_by_mesh.values()):
        if pf.started and pf.gcb is None:
            pf.ensure()
            pf.recaptures = getattr(pf, "recaptures", 0) + 1


def decode_all():
    """Called at every decode entry (qwen36_vllm decode_forward): from here on trace captures are decode captures."""
    for pf in list(_by_mesh.values()):
        pf.in_prefill = False


def release_all():
    for pf in list(_by_mesh.values()):
        pf.release()


def stop_all():
    for pf in list(_by_mesh.values()):
        pf.stop()
    _by_mesh.clear()

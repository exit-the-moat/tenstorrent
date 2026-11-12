"""DFlash2 multi-stream serving (ep2 night 4-5 Nov, agent ms1): S streams x T = 8 verify rows in ONE verify pass.

Off by default (DF_STREAMS unset or 1: df_server.py runs exactly as before; this module is not imported).

Layout (S = 4: R = 32 rows, the 32-row decode shapes the model already has for 32 plain users):
- the model is built with max_batch_size = R; stream s owns rows 8 s .. 8 s + 7, KV pool blocks [s * nbs, (s + 1) * nbs)
  (its own page-table row, width nbs), its own prefill slot s (rec_state / conv_states row s);
- verify = the model's own decode code on R rows (matmuls, norms, all-reduces, SDPA batched over R rows with the
  streams' aliased page tables), with two changes per call (as df_verify.Verifier, which this extends):
  * GDN layers: the in-projection runs once over R rows, then ONE chained program per stream (chain_all, the same
    kernel and buffers as the single-stream verify: per-stream base / bufs / window Xp / history), the gated rows of
    the streams concatenated for the out projection;
  * full attention: the KV write is the (fused K+V) update program once per slot k = 0..7 over all R rows, the
    position input holding pos[r] for the rows r % 8 == k (all others -1 = skipped): rows of different streams write
    different pages, so the streams' slot-k rows go in one program;
- DF_NOCOMMIT is the only commit mode: the accepted slot a of a stream is staged into that stream's `a` tensor
  (element T*T) and the stream's next verify starts from bufs[a];
- drafter: one TtDrafter per stream (weights shared, own ring / inputs / trace), writing [anchor, drafts] into its own
  token buffer; the verify trace concatenates the S token buffers.
"""
import contextlib
import functools
import os
import queue
import sys
import time
import traceback

import torch

import ttnn
from models.tt_transformers.tt.common import copy_host_to_device

from df_verify import Verifier, dev_rows

T = 8
_L1 = ttnn.L1_MEMORY_CONFIG
# DF_AUTO : ONE flag for the gated configuration; the whole set lives in df_auto.py (dfa: it is
# applied at the top of df_server.py, before the qwen36 import; this call is a no-op then).
import df_auto  # noqa: E402

df_auto.apply()

_DRAM = ttnn.DRAM_MEMORY_CONFIG
A_NONE = T + 247  # kernel convention: a >= T = start from base / hist
PRESTAGE = os.environ.get("DF_PRESTAGE", "0") == "1"  # the preset's host staging of the next draft during the verify
ROWS = os.environ.get("DF_MS_ROWS", "1") == "1"  # ms1o: GDN chains on rows of the shared tensors (no slice / concat)
MULTI = ROWS and os.environ.get("DF_MS_MULTI", "1") == "1"  # ms1o: + all streams' chains as one program per GDN layer
# ms4 (12 Nov): DF_MS_ROWS_AUTO=1 (default off) = the row kernels (+ DF_MS_MULTI's one chain program for all streams)
# per verifier, only where they are valid: S > 1 and depth k == T = 8 (2-4 live under DF_AUTO, rows 8 s .. 8 s + 7 inside
# one 16-row tile face; the window kernel copies GDNW_T = 8 rows, which at k < 8 reads the next stream's rows / crosses
# a face). DF_AUTO runs DF_MS_ROWS=0 because the R=30 (6 x 5) verifier is broken under the row kernels (r64t).
ROWS_AUTO = os.environ.get("DF_MS_ROWS_AUTO", "0") == "1"
MULTI_AUTO = os.environ.get("DF_MS_MULTI", "1") == "1"
MULTI_G = int(os.environ.get("DF_MS_MULTI_G", "4"))  # streams per chain program (smaller groups: more cores per head)
BD_MIN = int(os.environ.get("DF_MS_BDRAFT", "2"))  # ms1o: one batched drafter graph when >= this many streams are live (0 = off)
XT = os.environ.get("DF_XFER_T", "0") == "1"  # xfer: per-step host timers in the log (measurement only, default off)
CHAIN = int(os.environ.get("DF_MS_CHAIN", "0"))  # r64d: streams per verify pass (4 = two chains of 4; 0 = off)
# ms2: a long prompt's prefill (> 2048 tokens) in its chunk segments with a step of the live streams between them
# (df_server.prefill_gen; needs DF_MTRACE). Default off: the prefill runs whole between steps
ILV = os.environ.get("DF_MS_ILV", "0") == "1"
# ms2: requests admitted back to back (a burst) recreate the prefetcher's global CB only after the last one (the step
# recreates it anyway); each recreate costs ~29 ms (the next prefill frees it again). Default off
GCB_BATCH = os.environ.get("DF_MS_GCB_BATCH", "0") == "1"
# ms2: the prefill's GDN scratch straight into the stream's chain base / conv history (one slot-copy program) instead of
# scratch -> batched slot row s -> base / hist (begin_sequence: ~17 ms of copies on the card per admission). Default off
BEGIN_DIRECT = os.environ.get("DF_MS_BEGIN_DIRECT", "0") == "1"
# r64t: stream-count-aware speculative depth. Off by default (DF_SPEC_KCAP unset = the legacy fixed T=8 path).
# KCAP = the server's max verify depth per stream; the step's depth is k = max(1, min(KCAP, ROWS // live)); k = 1 is
# plain decode (no drafter, one row per stream). Every pass is NS * k <= ROWS rows (one 32-row tile row), so no pass is
# ever wider than a tile row. NS * KCAP <= the model's max_batch_size.
SPEC_KCAP = int(os.environ.get("DF_SPEC_KCAP", "0") or 0)
SPEC_ROWS = int(os.environ.get("DF_SPEC_ROWS", "32") or 32)
SPEC_FORCE = os.environ.get("DF_SPEC_K", "auto")  # auto | <int>: a fixed depth (controls / the sweep's k rows)
# the verifiers to build: "slots:depth,..." (a slot owns `depth` rows, so slots*depth <= SPEC_ROWS). The step picks the
# smallest entry that covers its live slots and whose depth is the scheduled one.
SPEC_PLAN = os.environ.get("DF_SPEC_PLAN", "")
# r64u: the automatic spec/plain engine switch. Armed by DF_PLAIN_LANE=1 (default off); once armed the server picks
# the engine per step from the live stream count, with no further operator input: live <= PLAIN_LIVE - 1 speculates
# at k = ROWS // live, live >= PLAIN_LIVE runs the batched plain lane. Buckets = the plain lane's row widths.
PLAIN_LANE = os.environ.get("DF_PLAIN_LANE", "0") == "1"
PLAIN_LIVE = int(os.environ.get("DF_PLAIN_LIVE", "7") or 7)
PLAIN_BUCKETS = [int(x) for x in os.environ.get("DF_PLAIN_BUCKETS", "8,16,32").split(",") if x.strip()]
LANE_SLOTS = os.environ.get("DF_MS_LANE_SLOTS", "0") == "1"  # dfa: lane-only slots keep base/hist only (StreamState)
PCACHE = os.environ.get("DF_PREFIX_CACHE", "0") == "1"  # ep2 dfpc (df_pcache.py)
# ms4 DF_MS_BSDPA=1 (default off): the batched drafter's per-stream ring SDPA calls (one per stream and layer, each
# parallel over 16 heads only, ~145 us) as ONE SDPA with the streams as the batch dim (rings concatenated per layer).
# Drafter-only (drafts change acceptance, never the output).
BSDPA = os.environ.get("DF_MS_BSDPA", "0") == "1"
# ms4 DF_GDN_WY=1 (default off): the k = 8 verifiers' GDN chains in block (WY) form (tt/gdn/fused_wy.py,
# kernels/gdnwy_*): S0 -> base and the block's K / U / R / gamma -> blk per stream and layer instead of 8 slot states
# (bufs). A stream's state convention follows the last step's engine (st.proto): 'wy' = recompute from base + blk at
# the accepted slot, 'old' = bufs[a]; switch_proto() converts when a step uses the other engine (k < 8 / plain lane).
WY = os.environ.get("DF_GDN_WY", "0") == "1"


def _rep(mesh):
    return ttnn.ReplicateTensorToMesh(mesh)


def _dev(t, mesh, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=_rep(mesh))


class StreamState(Verifier):
    """The GDN state of one stream: per GDN layer hist / base / bufs[T] / window Xp, and the stream's `a` tensor.
    Reuses Verifier's program builders (_win_prog, _alloc_chain) through the attributes they read."""

    def __init__(self, parent, s):
        self.parent, self.s = parent, s
        self.model, self.mesh, self.log, self.nd = parent.model, parent.mesh, parent.log, parent.nd
        # r64t: the chain-state CAPACITY (the max per-step depth this server will use). The buffers are shared by every
        # per-depth verifier, so a step at a smaller depth simply uses fewer of them and the state stays continuous.
        self.CAP = int(getattr(parent, "kcap", 0)) or T  # r64t: bufs = the deepest k this server will run (<= T)
        # dfa DF_MS_LANE_SLOTS=1 (default off): a slot >= PLAIN_LIVE - 1 never speculates (slots fill lowest first and a
        # PLAIN_LIVE-th live request moves every request to the plain lane until it drains), so its chain state is only
        # base / hist (the fresh-admission migration source): one state buffer per GDN layer instead of 1 + KCAP
        # (8 x 48 x 1.5 MiB = 576 MiB per slot and card saved). Lets NS > 8 slots fit next to the KV pool.
        self.lane_only = LANE_SLOTS and PLAIN_LANE and s >= PLAIN_LIVE - 1
        if self.lane_only:
            self.CAP = 1
        self.T = self.CAP
        self.gdn = parent.gdn
        self.kv_skip = True
        self.sdpa_rows = False
        self.hist, self.X = {}, {}
        for dn in self.gdn:
            self.hist[id(dn)] = ttnn.from_torch(
                torch.zeros(1, dn.K - 1, dn.qkv_dim_tp, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=_DRAM,
                mesh_mapper=_rep(self.mesh))
        self._alloc_fused_stash()
        self._alloc_chain()
        self.proto = "old"
        if WY and not self.lane_only and s < _wy_slots():
            from models.demos.blackhole.qwen36.tt.gdn import fused_wy

            self.blk = {id(dn): fused_wy.alloc_blk(self._fused(dn), self.mesh) for dn in self.gdn}
        assert self.nocommit and self.chain1 and self.xpersist, "DF_NOCOMMIT=1 DF_XWIN=1 (preset) required"
        assert os.environ.get("DF_XWIN", "0") == "1"
        # one `a` tensor for every depth: the kernels read element k*k of it (a < k = start from bufs[a])
        self.a_dev = _dev(torch.zeros(8 * 8 + 16, dtype=torch.int32), self.mesh, ttnn.int32)
        self._kvm = self.a_dev  # _win_prog reads self._kvm as the `a` tensor
        self.pending_a = None
        self._fdc = {}

    @property
    def _capturing(self):
        return self.parent._capturing

    def a_host(self):
        t = torch.zeros(8 * 8 + 16, dtype=torch.int32)
        v = self.pending_a if self.pending_a is not None else A_NONE
        for k in range(1, T + 1):  # r64t: depth k reads its own element k*k (the legacy T=8 path reads 64)
            t[k * k] = v
        return t

    def chain_stream(self, dn, fd, qs):
        """qs [1, k, W] (this stream's rows of the in-projection) -> gated [1, k, Nv*128] (L1).

        r64t: the conv window (Xl / Xp) is allocated at the CAPACITY depth for every step, whatever the step's depth k.
        With DF_NOCOMMIT the next step's conv history is read from the window rows a+1..a+3, so the window has to be ONE
        tensor shared by all depths: a window per (depth, layer) would make the first step after a depth switch read a
        window from an older step at that depth (mixed live counts). The rows this step needs (<= k+2) are exactly the
        ones the window program writes (3 + k rows); the capacity tail is never read."""
        D = dn.qkv_dim_tp
        k, w = int(qs.shape[1]), int(self.CAP)  # k = the step's depth (bufs / out rows), w = the window capacity
        shp = ttnn.Shape([1, dn.K - 1 + 2 * w, D])
        if id(dn) not in self.Xp:  # first (eager) verify, before any trace
            self.Xp[id(dn)] = ttnn.from_torch(torch.zeros(1, dn.K - 1 + 2 * w, D, dtype=torch.bfloat16),
                                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                              memory_config=_DRAM, mesh_mapper=_rep(self.mesh))
        Xl = ttnn.allocate_tensor_on_device(shp, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, _L1)
        ttnn.generic_op([self.hist[id(dn)], qs, Xl, self.Xp[id(dn)], self.a_dev], self._win_prog(dn, qs, Xl))
        self.X[id(dn)] = self.Xp[id(dn)]
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, k, dn.Nv * 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, _L1)
        fd._pcache = self._fdc.setdefault(id(fd), {})  # chain_all's program cache is one entry: keep one per stream
        fd.chain_all(qs, out, self.base[id(dn)], self.bufs[id(dn)][:k], Xl, self.a_dev)
        ttnn.deallocate(Xl)
        return out

    def window_rows(self, dn, qkvzab, row0):
        """DF_MS_MULTI: this stream's L1 verify window (+ the persistent window) from its rows of the shared
        in-projection; the caller frees the returned window after the chain program. r64t: the window is sized by
        the step's depth (self._k), not by the buffer capacity."""
        D, w = dn.qkv_dim_tp, int(self.CAP)  # r64t: one capacity window shared by every depth
        if id(dn) not in self.Xp:  # first (eager) verify, before any trace
            self.Xp[id(dn)] = ttnn.from_torch(torch.zeros(1, dn.K - 1 + 2 * w, D, dtype=torch.bfloat16),
                                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                              memory_config=_DRAM, mesh_mapper=_rep(self.mesh))
        Xl = ttnn.allocate_tensor_on_device(ttnn.Shape([1, dn.K - 1 + 2 * w, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                            self.mesh, _L1)
        ttnn.generic_op([self.hist[id(dn)], qkvzab, Xl, self.Xp[id(dn)], self.a_dev],
                        self._win_prog_rows(dn, qkvzab, Xl, row0))
        self.X[id(dn)] = self.Xp[id(dn)]
        return Xl

    def chain_rows(self, dn, fd, qkvzab, row0, out, oz0):
        """DF_MS_ROWS: this stream's rows row0 .. row0 + 7 of the shared in-projection qkvzab [1, R, W] -> rows
        row0 .. of the shared gated output [1, R, Nv*128] (no row slice, no concat; the kernels read / write the rows
        in place). oz0: output rows oz0 .. 31 zeroed by this program (32 = none)."""
        D, k, w = dn.qkv_dim_tp, int(self._k), int(self.CAP)  # k = state slots, w = the capacity window
        shp = ttnn.Shape([1, dn.K - 1 + 2 * w, D])
        if id(dn) not in self.Xp:  # first (eager) verify, before any trace
            self.Xp[id(dn)] = ttnn.from_torch(torch.zeros(1, dn.K - 1 + 2 * w, D, dtype=torch.bfloat16),
                                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                              memory_config=_DRAM, mesh_mapper=_rep(self.mesh))
        Xl = ttnn.allocate_tensor_on_device(shp, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, _L1)
        ttnn.generic_op([self.hist[id(dn)], qkvzab, Xl, self.Xp[id(dn)], self.a_dev],
                        self._win_prog_rows(dn, qkvzab, Xl, row0))
        self.X[id(dn)] = self.Xp[id(dn)]
        fd._pcache = self._fdc.setdefault((id(fd), row0, oz0), {})
        fd._ms_rows = (row0, row0, oz0)
        try:
            fd.chain_all(qkvzab, out, self.base[id(dn)], self.bufs[id(dn)][:k], Xl, self.a_dev)
        finally:
            fd._ms_rows = None
        ttnn.deallocate(Xl)

    def _win_prog_rows(self, dn, qkvzab, Xl, row0):
        """Verifier._win_prog with the stream's row offset (GDNW_XROW0) in the shared in-projection tile."""
        hist, Xp = self.hist[id(dn)], self.Xp[getattr(self, "_xkey", id(dn))]
        ts = (hist, qkvzab, Xl, Xp, self.a_dev)
        key = tuple(t.buffer_address() for t in ts) + (row0,)
        wc = self.__dict__.setdefault("_wk_rows", {})
        hit = wc.get((id(dn), row0))
        if hit is not None and hit[0] == key:
            return hit[1]
        from models.demos.blackhole.qwen36.tt.gdn.fused_decode import KDIR

        nh = dn.qkv_dim_tp // 32
        ncores = min(32, nh)
        grid = self.mesh.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(ncores)]
        crs = ttnn.num_cores_to_corerangeset(ncores, grid, True)
        ct = []
        for t in ts:
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        hper = -(-nh // ncores)
        for k, c in enumerate(cores):
            h0 = min(nh, k * hper)
            rt[c.x][c.y] = [hist.buffer_address(), qkvzab.buffer_address(), Xl.buffer_address(), Xp.buffer_address(),
                            h0, min(hper, nh - h0), self.a_dev.buffer_address()]
        kd = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/dfwin_reader.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=ttnn.ReaderConfigDescriptor())
        kd.defines = list(kd.defines) + [("GDNW_T", str(int(T))), ("GDNW_AIDX", "1"), ("GDNW_XROW0", str(int(row0)))]
        cbs = [ttnn.CBDescriptor(total_size=3 * 2048, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)])]
        prog = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=cbs)
        wc[(id(dn), row0)] = (key, prog)
        return prog


class MsVerifier(Verifier):
    """streams / tokbufs given (ms1o, DF_MS_SOLO): a verifier over a subset of another MsVerifier's streams, sharing
    their state buffers (base / bufs / window / hist / `a`, KV pages, the drafters' token buffers), with its own inputs
    and trace at R = 8 * len(streams) rows (the model's bucketed decode width): a step may use either verifier."""

    def __init__(self, model, tap_layer_ids, page_rows, log=print, streams=None, tokbufs=None, k=None):
        self.model = model
        self.mesh = model.mesh_device
        self.S = len(page_rows)
        # r64t: per-verifier depth. k set = a per-step-depth verifier (rows = slot * k); None = the legacy T=8 layout.
        self.T = T if k is None else int(k)
        self.R = self.S * self.T
        self.log = log
        self.nd = model.num_devices
        self.gdn = [l.attention for l in model.layers if not l.is_full_attention]
        self.attn = [l.attention for l in model.layers if l.is_full_attention]
        for dn in self.gdn:
            assert dn.B >= self.R, f"GDN batch rows {dn.B} < {self.R}: start the model with max_batch_size={self.R}"
        idx = list(model.layer_indices)
        self.tap_pos = [idx.index(t) for t in tap_layer_ids if t in idx]
        self.page_rows = [p.to(torch.int32).reshape(1, -1) for p in page_rows]
        self.pt_host = torch.cat([p.expand(self.T, -1) for p in self.page_rows], dim=0).contiguous()  # [R, nbs]
        # r64v: re-stage the page table on the device every step for this verifier (the plain lane sets this; see
        # _stage_ms). Default off: the speculative verifiers' page tables are staged once and stay correct.
        self.restage_pt = False
        self.sdpa_rows = False
        self.kv_skip = True
        self._kvm = None
        self.shard_argmax = self.nd > 1 and model._lmhead_vocab_sharded and os.environ.get("DF_SHARD_ARGMAX", "1") == "1"
        self._capturing = False
        self.streams = streams if streams is not None else [StreamState(self, s) for s in range(self.S)]
        self.tokbufs = tokbufs if tokbufs is not None else [
            _dev(torch.zeros(T, 1, dtype=torch.int32), self.mesh, ttnn.uint32) for _ in range(self.S)]
        self.begin_tids = {}
        self.sub = {}  # DF_MS_SUB: {live slots: verifier over just those streams} (set by bring_up_ms)
        self.inputs = None
        self.trace_id = None
        self.out = None
        self.timing = {}
        self.commit_tids = None
        self.begin_tid = None

    # ---------------------------------------------------------------- r64t: per-step depth
    def pick(self, live):
        """The step's speculative depth and the verifier that runs it. Schedule k = max(1, min(KCAP, ROWS // live));
        k = 1 is plain decode. DF_SPEC_K=<int> forces a depth (control rows); ms.force_k overrides. The verifier is the
        built (slots, depth) entry whose depth is the closest match and which covers this step's live slots: rows =
        slots*depth <= ROWS, a non-live slot is a stale row (bit-inert for the live rows, Stage C)."""
        k = max(1, min(int(getattr(self, "kcap", T)), SPEC_ROWS // max(1, len(live))))
        if SPEC_FORCE not in ("auto", ""):
            k = int(SPEC_FORCE)
        if getattr(self, "force_k", None):
            k = int(self.force_k)
        need = max([len(live)] + [s + 1 for s in live])  # the row layout is by slot index
        c = [p for p in self.plans if p[0] >= need] or [max(self.plans)]
        pair = min(c, key=lambda p: (abs(p[1] - int(k)), p[0]))
        return pair[1], self.kver[pair]

    # ---------------------------------------------------------------- per stream state
    def begin_sequence(self, s):
        """After the prompt prefill into slot s: stream s's base <- rec_state row s, hist <- conv_states rows."""
        st = self.streams[s]
        st.pending_a = None
        st.proto = "wy" if WY else "old"
        if BEGIN_DIRECT:  # the prefill wrote base / hist already (install_begin_direct)
            return
        if self.begin_tids.get(s) is not None:  # DF_BEGIN_TRACE: the copies as one trace per stream
            ttnn.execute_trace(self.mesh, self.begin_tids[s], cq_id=0, blocking=False)
            return
        self._begin_ops(s)

    def capture_begin_traces(self):
        for s in range(self.S):
            self._begin_ops(s)  # eager first: every program compiled
            ttnn.synchronize_device(self.mesh)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self._begin_ops(s)
            finally:
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.begin_tids[s] = tid
        ttnn.synchronize_device(self.mesh)
        self.log(f"[dflash-ms] {self.S} begin_sequence traces captured")

    def _begin_ops(self, s):
        st = self.streams[s]
        for dn in self.gdn:
            r0 = dn._slice_along(dn.rec_state, 0, s, s + 1)
            ttnn.copy(r0, st.base[id(dn)])
            ttnn.deallocate(r0)
            rows = [dn._slice_along(dn.conv_states[m], 1, s, s + 1) for m in range(1, dn.K)]
            h = ttnn.concat(rows, dim=1)
            ttnn.copy(h, st.hist[id(dn)])
            ttnn.deallocate(h)
            for r in rows:
                ttnn.deallocate(r)

    def commit(self, s, a):
        self.streams[s].pending_a = int(a)

    def release_traces(self):
        for v in self.sub.values():
            v.release_trace()
        for v in (getattr(self, "plain", None) or {}).values():
            v.release_trace()
        for tid in self.begin_tids.values():
            ttnn.release_trace(self.mesh, tid)
        self.begin_tids = {}
        if getattr(self, "bdraft", None) is not None:
            self.bdraft.release_trace()
        for bd in (getattr(self, "bdrafts", None) or {}).values():
            bd.release_trace()
        self.release_trace()

    # ---------------------------------------------------------------- GDN verify (S chains)
    def _gdn_verify_ms(self, dn, x):
        fd = self._fused(dn)
        assert fd is not None
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        R = x.shape[-2]
        tw = dn.tw
        qkvzab = dn._qkvzab_proj(x, R, _L1)
        W = qkvzab.shape[-1]
        if WY and self.T == T:
            from models.demos.blackhole.qwen36.tt.gdn import fused_wy

            gated = ttnn.allocate_tensor_on_device(ttnn.Shape([1, R, dn.Nv * 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                                   self.mesh, _L1)
            ml = []
            for s, st in enumerate(self.streams):
                st._k = self.T
                ml.append(dict(Xl=st.window_rows(dn, qkvzab, self.T * s), base=st.base[id(dn)], blk=st.blk[id(dn)],
                               kvm=st.a_dev, xrow0=self.T * s, orow0=self.T * s, oz0=R if s == 0 else 32))
            for g0 in range(0, len(ml), MULTI_G):
                fused_wy.chain_wy(fd, qkvzab, gated, ml[g0:g0 + MULTI_G], self.T)
            for m_ in ml:
                ttnn.deallocate(m_["Xl"])
            ttnn.deallocate(qkvzab)
            return self._gdn_out(dn, gated, R)
        rows_ok = ROWS or (ROWS_AUTO and self.S > 1 and self.T == T)
        multi_ok = MULTI or (ROWS_AUTO and rows_ok and MULTI_AUTO)
        if rows_ok:  # ms1o DF_MS_ROWS: the chains read / write their rows of the shared tensors in place
            gated = ttnn.allocate_tensor_on_device(ttnn.Shape([1, R, dn.Nv * 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                                   self.mesh, _L1)
            if multi_ok and self.S > 1:  # ms1o DF_MS_MULTI: all streams' chains in ONE program (a core group each)
                ml = []
                for s, st in enumerate(self.streams):
                    st._k = self.T
                    ml.append(dict(Xl=st.window_rows(dn, qkvzab, self.T * s), base=st.base[id(dn)],
                                   bufs=st.bufs[id(dn)][:self.T], kvm=st.a_dev, xrow0=self.T * s,
                                   orow0=self.T * s, oz0=R if s == 0 else 32))
                for g0 in range(0, len(ml), MULTI_G):  # DF_MS_MULTI_G: at most this many streams per program
                    fd.chain_multi(qkvzab, gated, ml[g0:g0 + MULTI_G])
                for m_ in ml:
                    ttnn.deallocate(m_["Xl"])
            else:
                for s, st in enumerate(self.streams):
                    st._k = self.T
                    st.chain_rows(dn, fd, qkvzab, self.T * s, gated, R if s == 0 else 32)
            ttnn.deallocate(qkvzab)
            outs = []
        else:
            gated = None
            outs = []
        for s, st in enumerate(self.streams if gated is None else ()):
            qs = ttnn.slice(qkvzab, (0, self.T * s, 0), (1, self.T * s + self.T, W), memory_config=_L1)
            outs.append(st.chain_stream(dn, fd, qs))
            ttnn.deallocate(qs)
        if gated is None:
            ttnn.deallocate(qkvzab)
            gated = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=1, memory_config=_L1)
        if len(outs) > 1:
            for o in outs:
                ttnn.deallocate(o)
        return self._gdn_out(dn, gated, R)

    def _gdn_out(self, dn, gated, R):
        partial = dn._row_proj(gated, dn.tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, R, partial.shape[-1]))
        from models.demos.blackhole.qwen36.tt import tp_common as _tpc
        from models.tt_transformers.tt.ccl import tt_all_reduce

        if _tpc.ar_active():
            return _tpc.decode_all_reduce(partial, dn.tt_ccl, "gdn")
        return tt_all_reduce(partial, dn.mesh, dn.tt_ccl, cluster_axis=0, dim=3, topology=dn.args.ccl_topology(),
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ---------------------------------------------------------------- attention row-wise wrappers
    @contextlib.contextmanager
    def _patched(self):
        orig_upd = ttnn.experimental.paged_update_cache
        orig_fused = ttnn.experimental.paged_fused_update_cache
        cache = {}
        R = self.R
        kvf_skip = os.environ.get("DF_KVF_SKIP", "0") == "1"
        pend = []

        def row_of(t, k):  # slot k's position mask [R]
            key = (id(t), k)
            if key not in cache:
                cache[key] = ttnn.slice(t, (k * R,), ((k + 1) * R,))
            return cache[key]

        def upd(kv_cache, x_sh, update_idxs_tensor=None, page_table=None, **kw):
            B = x_sh.shape[1]
            if page_table is None or B == 1:
                return orig_upd(kv_cache, x_sh, update_idxs_tensor=update_idxs_tensor, page_table=page_table, **kw)
            assert B == R, (B, R)
            if kvf_skip:
                if not pend:
                    pend.append((kv_cache, x_sh))
                    return
                k_cache, k_sh = pend.pop()
                sp = x_sh.memory_config().shard_spec
                key = ("vcfg", tuple(sp.shape))
                if key not in cache:
                    rs = list(sp.grid.ranges())
                    lo, hi = rs[0].start, rs[-1].end
                    gsz = self.mesh.compute_with_storage_grid_size()
                    if hi.y + 1 + (hi.y - lo.y) >= gsz.y:
                        # r64g: the V shard's range is stacked BELOW the K range. At 64 rows the K range is 8x8, so the
                        # stack wants y 8..15 on an 11x10 device (TT_FATAL: CB core range exceeds the compute grid).
                        # No room: write K and V with two paged_update_cache calls instead (same rows, same data).
                        cache[key] = None
                        self.log("[ms] DF_KVF_SKIP: no grid row left for the V range at R=%d (K grid %dx%d on grid %dx%d)"
                                 " - K/V written with two update calls" % (R, hi.x - lo.x + 1, hi.y - lo.y + 1, gsz.x, gsz.y))
                    else:
                        cr = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(lo.x, hi.y + 1),
                                                               ttnn.CoreCoord(hi.x, hi.y + 1 + (hi.y - lo.y)))})
                        cache[key] = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                                       ttnn.ShardSpec(cr, tuple(sp.shape), sp.orientation))
                if cache[key] is not None:
                    v2 = ttnn.to_memory_config(x_sh, cache[key])
                    for k in range(self.T):  # r64t: one update program per verified slot at this step's depth
                        orig_fused(k_cache, k_sh, kv_cache, v2, update_idxs_tensor=row_of(self._kvm, k),
                                   page_table=page_table)
                    ttnn.deallocate(v2)
                else:
                    for k in range(self.T):
                        cu = row_of(self._kvm, k)
                        orig_upd(k_cache, k_sh, update_idxs_tensor=cu, page_table=page_table, **kw)
                        orig_upd(kv_cache, x_sh, update_idxs_tensor=cu, page_table=page_table, **kw)
                return
            for k in range(self.T):
                orig_upd(kv_cache, x_sh, update_idxs_tensor=row_of(self._kvm, k), page_table=page_table, **kw)

        orig_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode
        split = self.S > 1 and os.environ.get("DF_MS_SDPA_SPLIT", "1") == "1"

        def sdpa(q, keys, values, page_table_tensor=None, cur_pos_tensor=None, **kw):
            # ms1o DF_MS_SDPA_SPLIT: one SDPA program per stream (8 rows, its page-table rows): the paged decode SDPA
            # splits each row's keys over grid / rows cores, so at R = 32 the reduction order differs from the
            # single-stream R = 8 program (4 live streams: GSM8K answers changed after ~150-350 tokens, 18:45)
            B = q.shape[1]
            if not split or B != R:
                return orig_sdpa(q, keys, values, page_table_tensor=page_table_tensor, cur_pos_tensor=cur_pos_tensor,
                                 **kw)
            qi = ttnn.sharded_to_interleaved(q, _DRAM) if q.is_sharded() else q  # interleaved Q must be in DRAM
            outs = []
            for k in range(self.S):
                qs = ttnn.slice(qi, (0, self.T * k, 0, 0), (1, self.T * k + self.T, qi.shape[2], qi.shape[3]),
                                memory_config=_DRAM)
                key = ("pt", k)
                if key not in cache:
                    cache[key] = (ttnn.slice(page_table_tensor, (self.T * k, 0),
                                             (self.T * k + self.T, page_table_tensor.shape[-1])),
                                  ttnn.slice(cur_pos_tensor, (self.T * k,), (self.T * k + self.T,)))
                pt, cp = cache[key]
                outs.append(orig_sdpa(qs, keys, values, page_table_tensor=pt, cur_pos_tensor=cp, **kw))
                ttnn.deallocate(qs)
            if qi is not q:
                ttnn.deallocate(qi)
            out = ttnn.concat(outs, dim=1, memory_config=kw.get("memory_config", _L1))
            for t in outs:
                ttnn.deallocate(t)
            return out

        # r64u: which of the three patches this verifier wants. The plain lane keeps the model's own batched GDN
        # decode and batched SDPA (patch_gdn / patch_sdpa off) but must take the KV write from here: the model's
        # native two-call paged_update_cache at R rows lands the new row in the wrong place (measured at 8 rows: the KV
        # at the new position pcc 0.61 / 0.44 against the prefill's own write, while the prefill's rows match exactly);
        # the fused masked write below is exact (the speculative verifier has used it for every counted number).
        p_gdn = getattr(self, "patch_gdn", True)
        p_kv = getattr(self, "patch_kv", True)
        p_sd = getattr(self, "patch_sdpa", True)
        if p_gdn:
            for dn in self.gdn:
                dn.forward_decode = functools.partial(self._gdn_verify_ms, dn)
        if p_kv:
            ttnn.experimental.paged_update_cache = upd
        if p_sd:
            ttnn.transformer.paged_scaled_dot_product_attention_decode = sdpa
        try:
            yield
            if p_kv:
                assert not pend, "fused KV write: a K row write had no V partner"
        finally:
            if p_kv:
                ttnn.experimental.paged_update_cache = orig_upd
            if p_sd:
                ttnn.transformer.paged_scaled_dot_product_attention_decode = orig_sdpa
            if p_gdn:
                for dn in self.gdn:
                    del dn.forward_decode

    # ---------------------------------------------------------------- forward / trace
    def _forward_ms(self, cur_pos, rope, page_table):
        # [R, 1] uint32 = the drafters' blocks (S = 1 and T = 8: the drafter's buffer itself, persistent).
        # r64t: at depth k < 8 only the first k rows of each drafter's block are verified, packed at slot * k.
        if self.T == T:
            toks = self.tokbufs[0] if self.S == 1 else ttnn.concat(self.tokbufs, dim=0)
        else:
            parts = [ttnn.slice(t, (0, 0), (self.T, 1)) for t in self.tokbufs]
            toks = parts[0] if self.S == 1 else ttnn.concat(parts, dim=0)
        ids, logits, tap_cat, rm = Verifier._forward(self, toks, cur_pos, rope, page_table)
        w = tap_cat.shape[-1]
        taps_s = []
        for s in range(self.S):
            t = ttnn.slice(tap_cat, (0, 0, self.T * s, 0), (1, 1, self.T * s + self.T, w))
            if self.T < T:
                # r64t: the drafter's context buffer is NCTX = T rows; pad this depth's taps with zero rows so the
                # prestage copy (prepare_all -> ttnn.copy into in_ctx) keeps its shape. Rows >= the accepted count
                # are stale/ignored exactly as they are on the legacy 8-row path.
                if getattr(self, "_tz", None) is None:
                    self._tz = ttnn.from_torch(torch.zeros(1, 1, T - self.T, w, dtype=torch.bfloat16),
                                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                               memory_config=_L1, mesh_mapper=_rep(self.mesh))
                t = ttnn.concat([t, self._tz], dim=2)
            taps_s.append(t)
        return (ids, logits, tap_cat, rm), toks, taps_s

    def _kvm_host(self, pos):
        m = torch.full((T, self.R), -1, dtype=torch.int32)
        live = getattr(self, "_live", None) if PCACHE else None  # ep2 dfpc: dead rows skip their KV write
        for k in range(self.T):
            for s in range(self.S):
                if live is not None and s not in live:
                    continue
                r = s * self.T + k
                m[k, r] = pos[r]
        return m.reshape(-1)

    def _stage_ms(self, starts):
        """Inputs of one verify step: stream s verifies positions starts[s] .. +T-1 (its tokens are in tokbufs[s])."""
        pos = torch.cat([torch.arange(st, st + self.T, dtype=torch.int32) for st in starts])
        i32 = lambda t: ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        if self.inputs is None:
            host = self.model.prepare_decode_inputs_host(torch.zeros(self.R, 1, dtype=torch.int32), pos,
                                                         page_table=self.pt_host)
            self.inputs = copy_host_to_device(host, mesh_device=self.mesh)
            self._kvm = _dev(self._kvm_host(pos), self.mesh, ttnn.int32)
        else:
            ttnn.copy_host_to_device_tensor(i32(pos), self.inputs[1])
            ttnn.copy_host_to_device_tensor(self.model.decode_rope_host(pos), self.inputs[2])
            ttnn.copy_host_to_device_tensor(i32(self._kvm_host(pos)), self._kvm)
        for st in self.streams:
            ttnn.copy_host_to_device_tensor(i32(st.a_host()), st.a_dev)
        if getattr(self, "restage_pt", False):
            # r64v (the plain lane): the page table the lane handed paged_fused_update_cache was a BROADCAST of
            # page-table row 0 on the device (host pt_host [0, 1024, 2048, ...]; device readback all row 0), so every
            # lane step wrote its K/V into slot 0's pages and the lane read a stale row for the position it had just
            # produced - exact-looking at the first step, garbage from the second on. The creation path is fine in
            # isolation (a fresh ttnn.to_device of the same host tensor is correct), so something later in the warm-up
            # clobbers the buffer; not diagnosed further. Re-stage it per step, as cur_pos / rope / _kvm are.
            if getattr(self, "_pt_tt", None) is None:
                self._pt_tt = ttnn.from_torch(self.pt_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(self._pt_tt, self.inputs[3])

    def capture_trace(self, starts):
        from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

        self._stage_ms(starts)
        _wp.decode_all()
        self._capturing = True
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.out, self.tok_out, self.taps_s = self._forward_ms(*self.inputs[1:4])
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self._capturing = False
        ttnn.synchronize_device(self.mesh)
        self.trace_id = tid
        self.log(f"[dflash-ms] verify trace captured (S={self.S}, R={self.R})")

    def run_eager(self, starts):
        """Eager verify (programs compiled before a capture); outputs kept in self.out like the trace's."""
        self._stage_ms(starts)
        if self.out is not None:
            for t in self.out:
                for u in (t if isinstance(t, tuple) else (t,)):
                    ttnn.deallocate(u)
            if self.S > 1:
                ttnn.deallocate(self.tok_out)
            for t in self.taps_s:
                ttnn.deallocate(t)
        self.out, self.tok_out, self.taps_s = self._forward_ms(*self.inputs[1:4])
        return self._read_ids()

    def run_chained(self, starts, pre=None, staged=False):
        """The drafters have written the blocks into the token buffers: verify. Returns (blocks [R], ids [R]).
        pre(): host work while the replay runs, before the read back (DF_PRESTAGE). staged: _stage_ms(starts) was
        already called (before the drafts, so the replay follows the drafter replays without a host gap)."""
        from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

        t0 = time.perf_counter()
        _wp.decode_all()
        if _wp.recreate_pending():
            _wp.recreated()  # DF_NO_RECAP=1: the GCB comes back at the same L1 address, the trace is replayed
        _t1 = time.perf_counter()
        if not staged:
            self._stage_ms(starts)
        _t2 = time.perf_counter()
        ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
        _t3 = time.perf_counter()
        if pre is not None:
            pre()
        _t4 = time.perf_counter()
        post = self._read_ids()
        _t5 = time.perf_counter()
        blk = dev_rows(self.tok_out, self.mesh)[0].reshape(-1)[: self.R].to(torch.int64) & 0xFFFFFFFF
        _t6 = time.perf_counter()
        self.timing["verify_ms"] = 1000 * (_t6 - t0)
        self.timing["xt"] = (_t1 - t0, _t2 - _t1, _t3 - _t2, _t4 - _t3, _t5 - _t4, _t6 - _t5)
        if _T4:
            for k_, d_ in (("v_gcb", _t1 - t0), ("v_stage", _t2 - _t1), ("v_exec", _t3 - _t2), ("v_pre", _t4 - _t3),
                           ("v_wait_ids", _t5 - _t4), ("v_blk", _t6 - _t5)):
                _t4add((k_, self.S), d_)
        return [int(v) for v in blk.tolist()], post

    def _read_ids(self):
        R = self.R
        idx_t, mx_t = self.out[0]
        ii_all, vv_all = dev_rows(idx_t, self.mesh), dev_rows(mx_t, self.mesh)
        per = []
        for d in range(self.nd):
            per.append((ii_all[d].reshape(-1)[:R].to(torch.int64) & 0xFFFFFFFF, vv_all[d].float().reshape(-1)[:R]))
        shard = self.model.vocab_size // self.nd
        ids, best = per[0][0].clone(), per[0][1].clone()
        for d in range(1, self.nd):
            better = per[d][1] > best
            ids = torch.where(better, per[d][0] + d * shard, ids)
            best = torch.where(better, per[d][1], best)
        return [int(v) for v in ids.tolist()]

    def host_logits(self):
        rm = self.out[3]
        parts = [t.reshape(-1, t.shape[-1])[: self.R] for t in dev_rows(rm, self.mesh)]
        return torch.cat(parts, dim=-1)[:, : self.model.vocab_size].float()


# ---------------------------------------------------------------------------------------------- drafters
_BLK_ID = {}


def _wy_slots():
    """The stream slots a k = T (WY) verifier can hold: the largest slot count of a depth-T entry of DF_SPEC_PLAN
    ("1:8,2:8,3:8,4:8,5:6,6:5" -> 4). Only these slots get a WY block store (~46 MB per slot and card); a stream in a
    higher slot never runs a depth-T step (pick() needs every live slot < the verifier's slot count)."""
    n = 0
    for it in os.environ.get("DF_SPEC_PLAN", "").split(","):
        if ":" in it:
            a_, k_ = it.split(":")
            if int(k_) == T:
                n = max(n, int(a_))
    return n or int(os.environ.get("DF_STREAMS", "8") or 8)


def _blk_identity(ms, dn):
    """A blk whose recompute gives S0 = base exactly (gamma = 1, K = U = R = 0): after an old-engine step."""
    key = id(dn)
    if key not in _BLK_ID:
        Nv = dn.Nv
        t = torch.zeros(1, Nv * 10 * 32, 32, dtype=torch.float32)
        for hv in range(Nv):
            t[0, (hv * 10 + 9) * 32:(hv * 10 + 10) * 32] = 1.0
        _BLK_ID[key] = ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=ms.mesh,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=_rep(ms.mesh))
    return _BLK_ID[key]


def _stage_a(st):
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(st.a_host(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                    st.a_dev)


def wy_commit(ms, st):
    """base <- the state at the accepted slot (recomputed from base + blk); pending_a is kept (the conv history of the
    next step still comes from window rows a + 1 ..)."""
    from models.demos.blackhole.qwen36.tt.gdn import fused_wy

    _stage_a(st)
    for dn in ms.gdn:
        fused_wy.commit(ms._fused(dn), st.base[id(dn)], st.blk[id(dn)], st.a_dev, T)


def switch_proto(ms, st, want):
    """Make stream st's state readable by a step of engine `want` ('wy' / 'old'); no-op without a pending slot."""
    if not WY or st.proto == want:
        return
    a = st.pending_a
    if a is not None and int(a) < T:
        if want == "wy":  # old -> wy: base <- bufs[a], blk <- identity
            for dn in ms.gdn:
                ttnn.copy(st.bufs[id(dn)][int(a)], st.base[id(dn)])
                ttnn.copy(_blk_identity(ms, dn), st.blk[id(dn)])
        else:  # wy -> old: base <- S_a, bufs[a] <- base
            wy_commit(ms, st)
            for dn in ms.gdn:
                ttnn.copy(st.base[id(dn)], st.bufs[id(dn)][int(a)])
        switch_proto.n = getattr(switch_proto, "n", 0) + 1
    st.proto = want


def make_drafters(d0, S, tokbufs):
    """S drafters: d0 + clones sharing d0's weights (own ring, inputs, trace); drafter s writes tokbufs[s]."""
    ds = [d0]
    for s in range(1, S):
        d = object.__new__(type(d0))
        d.__dict__.update(d0.__dict__)
        d._alloc_state()  # fresh ring / inputs / trace id
        ds.append(d)
    for d, tb in zip(ds, tokbufs):
        d.link_verify(tb)
    return ds


class MsDraft:
    """ms1o DF_MS_BDRAFT: the S drafters' step as ONE device graph over R = 8 S rows (the drafter is latency bound:
    ~7 ms per replay at 8 rows). Every row-wise op (norms, matmuls, conv, rope, LM head, argmax, combine) runs once
    on the stacked rows; per stream only the ring updates and the SDPA over its own ring (and the context
    permutation). Host bookkeeping and the persistent inputs stay in the S TtDrafter objects (their _stage /
    prestage); the rings stay theirs (prefill_context unchanged). Streams without a request compute on stale inputs
    (their outputs are never read; their ring rows are rewritten by the next prefill_context). Draft numerics: the
    same ops on more rows (the drafts only change acceptance, never the output)."""

    def __init__(self, ds):
        import df_drafter as D

        self.D = D
        self.ds = ds
        self.d0 = ds[0]
        self.mesh = self.d0.mesh
        self.S = len(ds)
        self.R = D.BLK * self.S
        assert D.STAGE_PACK and D.SHIFT_MM and D.RING_POS and self.d0.greedy, "preset drafter switches expected"
        R, B = self.R, D.BLK
        sh = torch.zeros(R, R, dtype=torch.bfloat16)
        for s in range(self.S):
            for t in range(1, B):
                sh[s * B + t, s * B + t - 1] = 1  # block-diagonal: no row crosses a stream boundary
        self.shift_m = D._up(sh.reshape(1, 1, R, R), self.mesh)
        r0 = torch.zeros(R, dtype=torch.float32)
        r0[::B] = 1.0
        self.row0 = D._up(r0.reshape(1, 1, R, 1), self.mesh, dtype=ttnn.float32)
        self.trace_id = None
        self.outs = None

    def _conv_dyn(self, L, c, x):
        D, d0 = self.D, self.d0
        _pc = d0._pc1d(x, L[c + ".proj"], key="conv")
        proj = ttnn.linear(x, L[c + ".proj"], compute_kernel_config=d0.hifi2 if D.CONV_HIFI2 else d0.hifi4,
                           **({"program_config": _pc} if _pc is not None else {}))
        if D.CONV_TP:
            proj = d0._gather(proj)
        d = [[None, None], [None, None]]
        for a in range(2):
            for o in range(2):
                lo = a * 640 + o * 320
                part = ttnn.slice(proj, (0, 0, 0, lo), (1, 1, self.R, lo + D.GROUPS))
                if D.CONV_BIAS:
                    d[a][o] = ttnn.linear(part, d0.E, bias=L[c + ".base"][a][o], compute_kernel_config=d0.hifi4)
                else:
                    e = ttnn.matmul(part, d0.E, compute_kernel_config=d0.hifi4)
                    d[a][o] = ttnn.add(e, L[c + ".base"][a][o])
                    ttnn.deallocate(e)
                ttnn.deallocate(part)
        ttnn.deallocate(proj)
        return d

    def _conv_apply(self, x, coef):
        prev = ttnn.matmul(self.shift_m, x, compute_kernel_config=self.d0.hifi4)
        out = ttnn.add(ttnn.multiply(x, coef[0]), ttnn.multiply(prev, coef[1]))
        ttnn.deallocate(prev)
        return out

    def _rows(self, x, s, dim=2):
        """Rows 8 s .. 8 s + 7 along dim (4-D)."""
        B = self.D.BLK
        lo, hi = [0, 0, 0, 0], list(x.shape)
        lo[dim], hi[dim] = B * s, B * s + B
        return ttnn.slice(x, tuple(lo), tuple(hi))

    def _layer(self, li, L, x, ctx_h, per, cos_b, sin_b, cos_c, sin_c):
        D, d0, R, B = self.D, self.d0, self.R, self.D.BLK
        res = x
        hn = d0._rms(x, L["in_norm"])
        da = self._conv_dyn(L, "attention_conv", hn)
        hc = self._conv_apply(hn, da[0])
        ttnn.deallocate(hn)
        q = d0._lin(hc, L["q"], key="q")
        qh = d0._heads(q, R, D.NH_L)
        ttnn.deallocate(q)
        q = d0._rope(d0._rms(qh, L["q_norm"]), cos_b, sin_b)  # [1, NH_L, R, 128]
        kvb = d0._lin(hc, L["kv"], key="kv")
        ttnn.deallocate(hc)
        kb = ttnn.slice(kvb, (0, 0, 0, 0), (1, 1, R, D.NKV_L * D.HD))
        vb = ttnn.slice(kvb, (0, 0, 0, D.NKV_L * D.HD), (1, 1, R, 2 * D.NKV_L * D.HD))
        ttnn.deallocate(kvb)
        kb = d0._rope(d0._rms(d0._heads(kb, R, D.NKV_L), L["k_norm"]), cos_b, sin_b)  # [1, NKV_L, R, 128]
        vb = d0._heads(vb, R, D.NKV_L)
        kc, vc = d0._ctx_kv(L, ctx_h, R, cos_c, sin_c)  # [1, NKV_L, R, 128]
        outs = []
        qps, ks, vs = [], [], []
        for s, d in enumerate(self.ds):
            for src, ring, ix in ((kc, d.ring_k[li], d.in_idx), (vc, d.ring_v[li], d.in_idx),
                                  (kb, d.ring_k[li], d.blk_idx), (vb, d.ring_v[li], d.blk_idx)):
                part = self._rows(src, s)
                r = ttnn.reshape(part, (1, D.NKV_L * 8, 1, D.HD))
                r = ttnn.pad(r, [1, D.NKV_L * 8, 32, D.HD], [0, 0, 0, 0], 0.0)
                r = ttnn.to_memory_config(r, d.kv_upd_cfg)
                ttnn.experimental.paged_update_cache(ring, r, update_idxs_tensor=ix)
                ttnn.deallocate(r)
                ttnn.deallocate(part)
            kfull = ttnn.reshape(d.ring_k[li], (1, D.NKV_L, D.RING, D.HD))
            vfull = ttnn.reshape(d.ring_v[li], (1, D.NKV_L, D.RING, D.HD))
            qs = self._rows(q, s)
            qp = ttnn.pad(qs, [1, D.NH_L, D.QPAD, D.HD], [0, 0, 0, 0], 0.0)
            if BSDPA:
                qps.append(qp)
                ks.append(kfull)
                vs.append(vfull)
                continue
            attn = ttnn.transformer.scaled_dot_product_attention(
                qp, kfull, vfull, attn_mask=per[s]["mask"], is_causal=False, scale=D.HD ** -0.5,
                program_config=d0.sdpa_cfg, compute_kernel_config=d0.hifi2)
            outs.append(ttnn.slice(attn, (0, 0, 0, 0), (1, D.NH_L, B, D.HD)))
            ttnn.deallocate(attn)
        if BSDPA:
            qa = ttnn.concat(qps, dim=0)  # [S, NH_L, QPAD, HD]
            ka = ttnn.concat(ks, dim=0)  # [S, NKV_L, RING, HD]
            va = ttnn.concat(vs, dim=0)
            attn = ttnn.transformer.scaled_dot_product_attention(
                qa, ka, va, attn_mask=self._mask_all, is_causal=False, scale=D.HD ** -0.5,
                program_config=d0.sdpa_cfg, compute_kernel_config=d0.hifi2)
            for t in [qa, ka, va] + qps:
                ttnn.deallocate(t)
            for s in range(self.S):
                outs.append(ttnn.slice(attn, (s, 0, 0, 0), (s + 1, D.NH_L, B, D.HD)))
            ttnn.deallocate(attn)
        for t in (kc, vc, kb, vb, q):
            ttnn.deallocate(t)
        a = ttnn.concat(outs, dim=2)
        for t in outs:
            ttnn.deallocate(t)
        a = ttnn.reshape(ttnn.transpose(a, 1, 2), (1, 1, R, D.NH_L * D.HD))
        o = d0._lin(a, L["o"], key="o")
        ttnn.deallocate(a)
        if D.TP:
            o = d0._allreduce(o)
        o2 = self._conv_apply(o, da[1])
        ttnn.deallocate(o)
        x = ttnn.add(res, o2)
        ttnn.deallocate(o2)
        ttnn.deallocate(res)
        for row in da:
            for t in row:
                ttnn.deallocate(t)
        res = x
        hn = d0._rms(x, L["post_norm"])
        dm = self._conv_dyn(L, "mlp_conv", hn)
        hc = self._conv_apply(hn, dm[0])
        ttnn.deallocate(hn)
        _pc = d0._pc1d(hc, L["gate"], act=ttnn.UnaryOpType.SILU, key="gate")
        if _pc is not None:
            gt = ttnn.linear(hc, L["gate"], compute_kernel_config=d0.hifi2, program_config=_pc)
        else:
            gt = ttnn.linear(hc, L["gate"], compute_kernel_config=d0.hifi2, activation="silu")
        up = d0._lin(hc, L["up"], key="up")
        ttnn.deallocate(hc)
        act = ttnn.multiply(gt, up)
        ttnn.deallocate(gt)
        ttnn.deallocate(up)
        dn = d0._lin(act, L["down"], key="down")
        ttnn.deallocate(act)
        if D.TP:
            dn = d0._allreduce(dn)
        m2 = self._conv_apply(dn, dm[1])
        ttnn.deallocate(dn)
        x = ttnn.add(res, m2)
        ttnn.deallocate(m2)
        ttnn.deallocate(res)
        for row in dm:
            for t in row:
                ttnn.deallocate(t)
        return x

    def _graph(self):
        # r64t: the drafter graph is a decode step of R = BLK * streams rows (64 at 8 streams). Without this context
        # the all-reduce sees no active decode step and refuses rows > 32 ("a wider grid is Stage A2").
        from models.demos.blackhole.qwen36.tt import decode_mode as _dm

        with _dm.step(max_rows=self.R):
            return self._graph_inner()

    def _graph_inner(self):
        D, d0, R, B = self.D, self.d0, self.R, self.D.BLK
        per = []
        for d in self.ds:  # the stream's packed inputs (DF_STAGE_PACK blob): mask, rope tables, context permutation
            b = d.in_blob
            p = {"mask": ttnn.slice(b, (0, 0, 0, 0), (1, 1, D.QPAD, D.KLEN))}
            p["rope"] = [ttnn.slice(b, (0, 0, D.QPAD, i * D.HD), (1, 1, D.QPAD + B, (i + 1) * D.HD)) for i in range(4)]
            p["perm"] = ttnn.slice(b, (0, 0, D.QPAD, 4 * D.HD), (1, 1, D.QPAD + D.NCTX, 4 * D.HD + D.NCTX))
            per.append(p)
        cos_b, sin_b, cos_c, sin_c = [ttnn.concat([p["rope"][i] for p in per], dim=2) for i in range(4)]
        self._mask_all = ttnn.concat([p["mask"] for p in per], dim=0) if BSDPA else None  # [S, 1, QPAD, KLEN]
        tok = ttnn.concat([d.in_tok for d in self.ds], dim=0)  # [R] uint32
        x = d0.target.embd(tok)
        x = ttnn.reshape(x, (1, 1, R, x.shape[-1]))
        if self.mesh.get_num_devices() > 1 and x.shape[-1] < D.H:
            from models.tt_transformers.tt.ccl import tt_all_gather

            x = tt_all_gather(x, self.mesh, d0.target.tt_ccl, cluster_axis=None, dim=3, topology=ttnn.Topology.Linear)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        ctx = ttnn.concat([d.in_ctx for d in self.ds], dim=2)  # [1, 1, R, 25600]
        ch = d0.ctx_hidden(ctx)
        ttnn.deallocate(ctx)
        parts = []
        for s, p in enumerate(per):  # DF_RING_POS: context row j -> lane (p_ctx + j) % 8, per stream
            hs = self._rows(ch, s)
            parts.append(ttnn.matmul(p["perm"], hs, compute_kernel_config=d0.hifi4))
            ttnn.deallocate(hs)
        ttnn.deallocate(ch)
        ctx_h = ttnn.concat(parts, dim=2)
        for t in parts:
            ttnn.deallocate(t)
        for li, L in enumerate(d0.layers):
            x = self._layer(li, L, x, ctx_h, per, cos_b, sin_b, cos_c, sin_c)
        ttnn.deallocate(ctx_h)
        for p in per:
            for t in [p["mask"], p["perm"]] + p["rope"]:
                ttnn.deallocate(t)
        for t in (cos_b, sin_b, cos_c, sin_c):
            ttnn.deallocate(t)
        if self._mask_all is not None:
            ttnn.deallocate(self._mask_all)
            self._mask_all = None
        hn = d0._rms(x, d0.norm)
        ttnn.deallocate(x)
        _pc = d0._pc1d(hn, d0.lm_w, key="lm")
        logits = ttnn.linear(hn, d0.lm_w, compute_kernel_config=d0.hifi2,
                             **({"program_config": _pc} if _pc is not None else {}))
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(logits)
        idx = ttnn.argmax(rm, dim=-1)
        gi = ttnn.reshape(idx, list(idx.shape) + [1]) if len(idx.shape) < len(rm.shape) else idx
        mx = ttnn.gather(rm, -1, gi)
        import df_chain

        blk = df_chain.combine_dev(self.mesh, d0.target.tt_ccl, idx, mx, tok,
                                   d0.lm_shard, self.row0)  # [R, 1] uint32
        for s, d in enumerate(self.ds):
            bs = ttnn.slice(blk, (B * s, 0), (B * s + B, 1))
            ttnn.copy(bs, d.chain_dst)
            ttnn.deallocate(bs)
        ttnn.deallocate(blk)
        ttnn.deallocate(tok)
        return hn, idx, mx, rm

    def run_eager(self):
        if self.outs is not None:
            for t in self.outs:
                ttnn.deallocate(t)
        self.outs = self._graph()

    def capture_trace(self):
        hold = None
        try:
            from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

            for pf in _wp._by_mesh.values():
                if pf.started and pf.gcb is None and not pf.broken:
                    hold = pf._hold_alloc(64 * 1024)
        except Exception:  # noqa: BLE001
            pass
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self.outs = self._graph()
        finally:
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        if hold is not None:
            ttnn.deallocate(hold)
        self.trace_id = tid

    def run(self):
        ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)

    def release_trace(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None


def stage_only(d, anchor, start, prev_a):
    """A drafter's inputs for this step without its replay (DF_MS_BDRAFT): draft_async / draft_async_pre minus the
    execute_trace."""
    if PRESTAGE and prev_a is not None and getattr(d, "pre", None):
        taps, pos, (idx_h, blob_h) = d.pre.pop(prev_a)
        d.pre = None
        d.pending_ctx = None
        d.pos = pos
        ttnn.copy(taps, d.in_ctx)
        if getattr(d, "preup", False):  # ms4: DF_PREUP=1 = the variants were uploaded during the verify (prepare_all)
            ttnn.copy(d.pre_idx[prev_a], d.in_idx)
            ttnn.copy(d.pre_blob[prev_a], d.in_blob)
        else:
            ttnn.copy_host_to_device_tensor(idx_h, d.in_idx)
            ttnn.copy_host_to_device_tensor(blob_h, d.in_blob)
        d._h2d(torch.tensor([int(anchor)] + [d.mask_id] * (T - 1), dtype=torch.int32), d.in_tok, ttnn.uint32,
               ttnn.ROW_MAJOR_LAYOUT)
    else:
        d._stage(anchor, start)


# ---------------------------------------------------------------------------------------------- scheduler
class Active:
    def __init__(self, req, s, out, start, t0, t_pf):
        self.req, self.s, self.out, self.start = req, s, out, start
        self.t0, self.t_pf = t0, t_pf
        self.steps, self.acc = 0, []
        self.prev_a = None  # DF_PRESTAGE: the accepted slot of the previous step (its drafter inputs are staged)


def srv_():
    return sys.modules["__main__"]


def admit(ms, ds, s, req, gcb=True, plan=None):
    """Prefill req into slot s. Returns an Active, or None when the request is already complete. gcb=False: leave the
    prefetcher's global CB freed (another admission follows; DF_MS_GCB_BATCH). plan (ep2 dfpc DF_PREFIX_CACHE): the
    prefix cache's plan for this request in slot s (df_pcache.PrefixCache.plan)."""
    srv = srv_()
    S = srv.S
    ids = req.ids
    t0 = time.perf_counter()
    srv.log(f"[ms] admit slot {s}: prompt {len(ids)} tokens")
    pc = getattr(S, "pcache", None)
    if pc is not None and plan is not None:
        cl = pc.before_prefill(s, plan)
        ok = False
        try:
            logits, taps, tap0 = srv.prefill(ids, slot=s, page_row=S.page_rows[s], cached_len=cl)
            pc.after_prefill(s, plan)  # (checks the model resumed where the plan said)
            ok = True
        finally:
            if not ok:
                pc.after_prefill(s, plan, ok=False)  # the slot's pages are unknown now: nothing valid
    else:
        if pc is not None:
            pc.after_prefill(s, None, ok=False)  # (no plan: a warm-up admission) the slot's pages are rewritten
        logits, taps, tap0 = srv.prefill(ids, slot=s, page_row=S.page_rows[s])
    return _admit_post(ms, ds, s, req, t0, logits, taps, tap0, gcb)


def admit_gen(ms, ds, s, req):
    """ms2 DF_MS_ILV: admit() as a generator; yields between the chunk segments of a long prompt (srv.prefill_gen)."""
    srv = srv_()
    t0 = time.perf_counter()
    srv.log(f"[ms] admit slot {s}: prompt {len(req.ids)} tokens")
    logits, taps, tap0 = yield from srv.prefill_gen(req.ids, s, srv.S.page_rows[s])
    return _admit_post(ms, ds, s, req, t0, logits, taps, tap0)


def _admit_post(ms, ds, s, req, t0, logits, taps, tap0, gcb=True):
    srv = srv_()
    S = srv.S
    ids = req.ids
    first = srv.host_pick(logits, 0, req)
    t_pf = time.perf_counter() - t0
    srv.log(f"[ms] slot {s} prefilled in {t_pf:.2f}s")
    req.out = [first]
    if req.q:
        req.q.put([first])
    a = Active(req, s, [first], len(ids), t0, t_pf)
    if (first in S.eos and not req.ignore_eos) or req.max_new <= 1:
        finish(a)
        return None
    ms.begin_sequence(s)
    ds[s].prefill_context(taps, list(range(tap0, a.start)))
    ds[s].pre = None
    if gcb:
        gcb_up()  # DF_EARLY_GCB: the GCB recreated while the card still runs the context ops queued above
    srv.log(f"[ms] slot {s} ready ({time.perf_counter() - t0:.2f}s)")
    return a


def finish(a):
    srv = srv_()
    S = srv.S
    req = a.req
    wall = time.perf_counter() - a.t0
    req.out = a.out
    req.stats = dict(
        t=time.strftime("%Y-%m-%dT%H:%M:%S"), mode="ms", prompt_tokens=len(req.ids), completion_tokens=len(a.out),
        steps=a.steps, accept_mean=round(sum(a.acc) / len(a.acc), 3) if a.acc else None,
        tokens_per_step=round((len(a.out) - 1) / a.steps, 3) if a.steps else None,
        prefill_s=round(a.t_pf, 3), decode_s=round(wall - a.t_pf, 3),
        decode_tps=round((len(a.out) - 1) / (wall - a.t_pf), 2) if wall > a.t_pf and len(a.out) > 1 else None,
        accept_hist=[a.acc.count(i) for i in range(1, T + 1)], slot=a.s, tag=getattr(req, "tag", ""),
    )
    try:
        import json

        with open(srv.STATS, "a") as f:
            f.write(json.dumps(req.stats) + "\n")
    except Exception:
        pass
    S.last_stats = (S.last_stats + [req.stats])[-50:]
    if getattr(S, "pcache", None) is not None:
        S.pcache.finished(a.s)
    srv.log("req", req.stats)
    if req.q:
        req.q.put(None)
    req.done.set()


_T4 = os.environ.get("DF_MS4_T", "0") == "1"  # ms4: host timers per step section (log every 50 steps per live count)
_T4A = {}


def _t4add(key, dt):
    if _T4:
        a = _T4A.setdefault(key, [0, 0.0])
        a[0] += 1
        a[1] += dt


def gcb_up():
    """The prefetcher's global CB back (same L1 address) and the prefetcher running, as DF_EARLY_GCB does before the
    single-stream server's first draft: the drafter traces are captured with it up (ms1o 18:22: a drafter trace
    captured with the prefetcher stopped hung the device when replayed with it running, in step 2)."""
    from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

    _wp.decode_all()
    if _wp.recreate_pending():
        _wp.recreated()


def step(ms, ds, active):
    """One DFlash2 step over every active stream: drafters, one verify pass, per-stream accept and commit."""
    srv = srv_()
    S = srv.S
    starts = [a.start if a else 0 for a in active]
    ta = time.perf_counter()
    if XT:
        step.gap = ta - getattr(step, "tend", ta)
    gcb_up()
    if XT:
        step.tg = time.perf_counter() - ta
    live = tuple(s for s, a in enumerate(active) if a)
    # r64t DF_SPEC_KCAP: the step's speculative depth from the live stream count (k = ROWS // live, capped by the
    # server's KCAP); k = 1 is plain decode. The verifier is chosen by k; every pass is NS * k <= ROWS rows.
    spec = getattr(ms, "kver", None)
    kstep, vspec = ms.pick(live) if spec else (None, None)
    # r64d Stage B DF_MS_CHAIN: verify each chain's live streams in its OWN pass (<= CHAIN*8 rows, the row-exact
    # width). The NS*8-row parent pass is never run - above 32 rows the model is not row-exact (binv 33 = 0/5).
    chains = getattr(ms, "chains", None)
    if chains:
        groups = [g for g in (tuple(s for s in live if s in ch) for ch in chains) if g]
    else:
        groups = [live]
    if WY and spec:
        want_ = "wy" if int(kstep) == T else "old"
        for s in live:
            switch_proto(ms, ms.streams[s], want_)
    bd = (getattr(ms, "bdrafts", None) or {}).get(live) or getattr(ms, "bdraft", None)  # DF_MS_BDSUB: live streams only
    if spec and live and ms.bdrafts and live not in ms.bdrafts:
        # ms4 DF_MS_BDPREFIX: the smallest prefix graph (slots 0 .. n-1) that holds every live slot
        pre_ = [g for g in ms.bdrafts if g == tuple(range(len(g))) and max(live) < len(g)]
        if pre_:
            bd = ms.bdrafts[min(pre_, key=len)]
    if kstep == 1:
        # r64t k = 1: plain decode. No drafter replay - the committed token alone is row 0 of each stream's block.
        for s in live:
            d, a = ds[s], active[s]
            v0 = torch.tensor([[int(a.out[-1])] + [d.mask_id] * (T - 1)], dtype=torch.int32).reshape(T, 1)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(v0, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                            d.chain_dst)
        step.plain = getattr(step, "plain", 0) + 1
    elif getattr(ms, "bgroups", None) and BD_MIN:
        for g, bdg in ms.bgroups:
            gl = [s for s in g if s in live]
            if not gl or bdg.trace_id is None:
                continue
            for s in gl:
                stage_only(ds[s], active[s].out[-1], active[s].start, active[s].prev_a)
            bdg.run()
        step.bd = getattr(step, "bd", 0) + 1
    elif bd is not None and bd.trace_id is not None and BD_MIN and len(live) >= BD_MIN:
        t4 = time.perf_counter()
        for s in live:  # DF_MS_BDRAFT: the live drafters' inputs, then one replay for all streams
            stage_only(ds[s], active[s].out[-1], active[s].start, active[s].prev_a)
        t5 = time.perf_counter()
        bd.run()
        _t4add(("dstage", len(live)), t5 - t4)
        _t4add(("drun", len(live)), time.perf_counter() - t5)
        step.bd = getattr(step, "bd", 0) + 1
    else:
        for s in live:
            a = active[s]
            if PRESTAGE and a.prev_a is not None and getattr(ds[s], "pre", None):
                ds[s].draft_async_pre(a.prev_a, a.out[-1])  # inputs staged during the previous verify
            else:
                ds[s].draft_async(a.out[-1], a.start)
    tb = time.perf_counter()
    runs = []
    for g in groups:
        if spec:
            v, vrow, vstarts = vspec, {s: s for s in range(vspec.S)}, [starts[s] for s in range(vspec.S)]
        elif chains:
            v = ms.sub.get(g)
            assert v is not None, (g, "no chain verifier for this live set")
            vrow = {s: i for i, s in enumerate(g)}
            vstarts = [starts[s] for s in g]
        else:
            v = ms.sub.get(g)
            if v is None:
                v, vrow = ms, {s: s for s in range(ms.S)}
                vstarts = starts
            else:
                vrow = {s: i for i, s in enumerate(g)}
                vstarts = [starts[s] for s in g]
        if PCACHE:
            v._live = {vrow[s] for s in live if s in vrow}  # ep2 dfpc: the pass's live rows (dead rows: no KV write)
        pre = None
        if PRESTAGE:
            # (staging the verify inputs before the drafts was slower: run 5, code 114 vs 116 tok/s; the staging
            # overlaps the drafter replay here)
            pre = lambda v=v, g=g, vrow=vrow: [ds[s].prepare_all(starts[s], v.taps_s[vrow[s]]) for s in g]
        blocks, post = v.run_chained(vstarts, pre=pre)
        step.sub = getattr(step, "sub", 0) + (v is not ms)
        if spec:
            step.k = kstep
        runs.append((v, vrow, blocks, post))
    tc = time.perf_counter()
    any_pen = any(a is not None and a.req.pen is not None for a in active)
    run_of = {}
    for v, vrow, blocks, post in runs:
        host = v.host_logits() if any_pen else None
        for s, i in vrow.items():
            if s < len(active) and active[s] is not None:
                run_of[s] = (v, i, blocks, post, host)
    for s, a in enumerate(active):
        if not a:
            continue
        req = a.req
        v_s, i_s, blocks_s, post_s, host_s = run_of[s]
        nrow = v_s.T  # r64t: the depth this verifier ran (T = 8 on the legacy path)
        r0 = i_s * nrow  # this stream's first row in the verifier that ran it
        blk = blocks_s[r0:r0 + nrow]
        ids = post_s[r0:r0 + nrow]
        assert blk[0] == a.out[-1], (s, blk[0], a.out[-1])
        rows = host_s[r0:r0 + nrow] if (host_s is not None and req.pen is not None) else None
        k = 0
        new = []
        while True:
            n_out = len(a.out) + len(new)
            if rows is not None:
                t = srv.host_pick(rows[k], n_out, req)
            else:
                t = ids[k]
                if not req.ignore_eos and n_out < req.min_new and t in S.eos:
                    lg = v_s.host_logits()[r0 + k]
                    for e in S.eos:
                        lg[e] = -float("inf")
                    t = int(torch.argmax(lg))
            new.append(t)
            if k < nrow - 1 and t == blk[k + 1]:
                k += 1
                continue
            break
        room = req.max_new - len(a.out)
        stop = False
        if not req.ignore_eos:
            for j, t in enumerate(new):
                if t in S.eos and len(a.out) + j + 1 >= req.min_new:
                    new = new[: j + 1]
                    stop = True
                    break
        if len(new) > room:
            new = new[:room]
        ds[s].set_step_context(v_s.taps_s[i_s], k + 1, a.start)
        ms.commit(s, k)
        a.prev_a = k
        a.out.extend(new)
        if req.q:
            req.q.put(new)
        a.steps += 1
        a.acc.append(k + 1)
        a.start += k + 1
        if stop or len(a.out) >= req.max_new or a.start + nrow >= S.max_len:
            finish(a)
            active[s] = None
    step.t = (tb - ta, tc - tb, time.perf_counter() - tc)
    if _T4:
        L_ = len(live)
        _t4add(("step_draft", L_), tb - ta)
        _t4add(("step_verify", L_), tc - tb)
        _t4add(("step_post", L_), step.t[2])
        n_ = _T4A[("step_post", L_)][0]
        if n_ % 50 == 0:
            vs_ = getattr(vspec, "S", L_)
            srv.log(f"[ms4-T] live {L_} (verifier S {vs_}) n {n_}: " + " ".join(
                f"{k[0]} {1e3 * v[1] / max(1, v[0]):.2f}" for k, v in sorted(_T4A.items())
                if (k[0].startswith("v_") and k[1] == vs_) or (not k[0].startswith("v_") and k[1] == L_)))
    step.n = getattr(step, "n", 0) + 1
    if XT:
        step.tend = time.perf_counter()
        # accumulate over 40 steps: gap (host between steps), gcb, draft stage, verify pieces, post
        xa = step.__dict__.setdefault("xa", [0.0] * 10)
        x = getattr(runs[0][0], "timing", {}).get("xt", (0,) * 6)
        for i, v in enumerate((step.gap, step.tg, step.t[0] - step.tg, *x, step.t[2])):
            xa[i] += v
        if step.n % 40 == 0:
            srv.log("[xt] step " + str(step.n) + " ms/step gap %.2f gcb %.2f draftstage %.2f | decode_all %.2f stage %.2f exec %.2f pre %.2f readids %.2f readblk %.2f | post %.2f" % tuple(1e3 * v / 40 for v in xa))
            step.xa = [0.0] * 10
    if step.n % 20 == 1 or os.environ.get("DF_MS_VERBOSE") == "1":
        srv.log(f"[ms] step {step.n}: draft {1e3 * step.t[0]:.1f} verify {1e3 * step.t[1]:.1f} post {1e3 * step.t[2]:.1f} ms; "
                f"active {[None if a is None else (len(a.out), a.start) for a in active]}")


def ms_worker(srv):
    """Replaces df_server.worker when DF_STREAMS > 1: bring-up / exec jobs as before; requests get a free stream slot
    (prefill between steps), and every step serves all active streams."""
    S = srv.S
    pend, active = [], []
    adm = None  # DF_MS_ILV: the admission in progress (slot, req, generator)
    while True:
        idle = not pend and all(a is None for a in active) and adm is None
        job = None
        try:
            job = S.jobs.get(block=idle)
        except queue.Empty:
            job = None
        while job is not None:
            try:
                if job[0] == "gen":
                    pend.append(job[1])
                else:
                    srv._dispatch(job)
            except Exception:
                srv.log("worker error", traceback.format_exc())
            try:
                job = S.jobs.get_nowait()
            except queue.Empty:
                job = None
        ms = getattr(S, "ms", None)
        if ms is None or not S.ready:
            continue
        NSW = getattr(ms, "nstreams", ms.S)  # r64d: chains share the model; ms.S is one chain's stream count
        if len(active) != NSW:
            active = [None] * NSW
        ds = S.drafters
        while ILV:
            # ms2: one admission at a time; each next() runs one prefill segment. Mid-prefill with live streams: step
            # them first; with none live the prefill just continues.
            if adm is None:
                free = [s for s in range(NSW) if active[s] is None]
                if not free or not pend:
                    break
                req = pend.pop(0)
                adm = (free[0], req, admit_gen(ms, ds, free[0], req))
            s, req, g = adm
            try:
                next(g)
            except StopIteration as e:
                active[s], adm = e.value, None
                continue
            except Exception as e:
                adm = None
                req.error = repr(e)
                srv.log("admit failed:", traceback.format_exc())
                if req.q:
                    req.q.put(None)
                req.done.set()
                continue
            if any(a is not None for a in active):
                break
        pc = getattr(S, "pcache", None)
        while pc is not None and not ILV and pend and any(a is None for a in active):
            free = [j for j in range(NSW) if active[j] is None]
            live_ = [j for j in range(NSW) if active[j] is not None]
            req = pend.pop(0)
            more = GCB_BATCH and bool(pend) and len(free) > 1
            try:
                s, plan = pc.plan(req.ids, free, live_)
                active[s] = admit(ms, ds, s, req, gcb=not more, plan=plan)
            except Exception as e:
                req.error = repr(e)
                srv.log("admit failed:", traceback.format_exc())
                if req.q:
                    req.q.put(None)
                req.done.set()
        for s in range(NSW):
            if ILV or pc is not None:
                break
            if active[s] is None and pend:
                req = pend.pop(0)
                more = GCB_BATCH and bool(pend) and any(active[j] is None for j in range(s + 1, NSW))
                try:
                    active[s] = admit(ms, ds, s, req, gcb=not more)
                except Exception as e:
                    req.error = repr(e)
                    srv.log("admit failed:", traceback.format_exc())
                    if req.q:
                        req.q.put(None)
                    req.done.set()
        if any(a is not None for a in active):
            try:
                # r64u: the engine follows the live stream count, with no operator input. live >= PLAIN_LIVE runs
                # the batched plain lane; below it, speculation. A request that is plain stays plain until it
                # finishes (a live drop does not hand it back: the reverse migration would need the drafter's
                # tap context, which the plain lane does not produce), so the lanes are never mixed in one step.
                if getattr(ms, "plain", None):
                    live = [s for s, a in enumerate(active) if a is not None]
                    # the engine of every live request is decided here, every step, from the live set alone: a request
                    # keeps the engine it was given (a plain request stays plain until it finishes), a fresh one gets
                    # the engine the live set calls for. Migrating on the per-request flag (not on the mode transition)
                    # is what makes a request admitted into an already-plain lane get its state moved at all.
                    want = len(live) >= ms.plain_live or any(
                        getattr(a, "engine", None) == "plain" for a in active if a is not None)
                    if want:
                        for s in live:
                            if getattr(active[s], "engine", None) != "plain":
                                migrate_chain_to_rows(ms, s)
                                active[s].engine = "plain"
                        if not ms.plain_mode:
                            ttnn.synchronize_device(S.mesh)
                            ms.plain_mode = True
                            srv.log(f"[ms] engine -> PLAIN ({len(live)} live, {sum(1 for s in live if getattr(active[s], 'engine', None) == 'plain')} in the lane)")
                    else:
                        for s in live:
                            if getattr(active[s], "engine", None) is None:
                                active[s].engine = "spec"
                        if ms.plain_mode:
                            ms.plain_mode = False
                            srv.log("[ms] engine -> SPEC (plain lane drained)")
                if getattr(ms, "plain_mode", False):
                    step_plain(ms, ds, active)
                else:
                    step(ms, ds, active)
            except Exception as e:
                srv.log("step failed:", traceback.format_exc())
                for s, a in enumerate(active):
                    if a is not None:
                        a.req.error = repr(e)
                        if a.req.q:
                            a.req.q.put(None)
                        a.req.done.set()
                        active[s] = None
                S.ready, S.status = False, "step failed (see log): " + repr(e)


def install_begin_direct(model, ms):
    """DF_MS_BEGIN_DIRECT: model._write_gdn_slot_from_scratch (called by prefill_paged_slots with QWEN36_GDN_SLOT_DEV
    while the B=1 scratch is bound) writes stream `slot`'s chain base (<- the recurrent state) and conv history
    (row m - 1 <- conv state m, m = 1..K-1): the bytes _begin_ops copies there from batched row `slot`, without the
    batched row. The model's slot-copy kernel (kernels/slotcopy_reader.cpp) with these targets; one program per stream,
    built by the warm-up's prefills (before any trace)."""
    import pathlib

    assert os.environ.get("QWEN36_GDN_SLOT_DEV") == "1"
    cache = {}

    def write(slot, prev):
        st = ms.streams[slot]
        jobs = []  # (mode, src, dst, n tiles, param), as model._write_gdn_slot_kernel
        for (dn, *_b), (_dn, rec_s, conv_s, *_r) in zip(prev, model._gdn_prefill_scratch):
            base, hist = st.base[id(dn)], st.hist[id(dn)]
            assert rec_s.dtype == base.dtype == ttnn.float32 and tuple(base.shape) == tuple(rec_s.shape), (base.shape, rec_s.shape)
            npg = rec_s.shape[1] * (rec_s.shape[2] // 32) * (rec_s.shape[3] // 32)
            jobs.append((0, rec_s, base, npg, 0))
            for m in range(1, dn.K):
                cs = conv_s[m]
                assert cs.dtype == hist.dtype == ttnn.bfloat16 and cs.shape[-1] == hist.shape[-1] and hist.shape[-2] == dn.K - 1
                jobs.append((1, cs, hist, cs.shape[-1] // 32, m - 1))
        key = (slot,) + tuple((j[1].buffer_address(), j[2].buffer_address()) for j in jobs)
        hit = cache.get(key)
        if hit is None:
            grid = model.mesh_device.compute_with_storage_grid_size()
            ncores = min(96, grid.x * grid.y)
            total = sum(j[3] for j in jobs)
            q = -(-total // ncores)
            segs = [[] for _ in range(ncores)]
            c, used = 0, 0
            for mode, src, dst, n, prm in jobs:
                t = 0
                while t < n:
                    take = min(n - t, q - used)
                    segs[c].append([mode, src.buffer_address(), dst.buffer_address(), t, take, prm])
                    t += take
                    used += take
                    if used == q:
                        c, used = c + 1, 0
            cores = [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(ncores)]
            rt = ttnn.RuntimeArgs()
            for k, cc in enumerate(cores):
                a = [len(segs[k])]
                for sg in segs[k]:
                    a += sg
                rt[cc.x][cc.y] = a
            crs = ttnn.num_cores_to_corerangeset(ncores, grid, True)
            kdir = pathlib.Path(sys.modules[type(model).__module__].__file__).resolve().parent.parent / "kernels"
            kd = ttnn.KernelDescriptor(
                kernel_source=str(kdir / "slotcopy_reader.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=list(ttnn.TensorAccessorArgs(jobs[0][1]).get_compile_time_args()),
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
            cbd = ttnn.CBDescriptor(total_size=4096, core_ranges=crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=4096)])
            hit = cache[key] = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cbd])
        ttnn.generic_op([j[1] for j in jobs] + [j[2] for j in jobs], hit)

    model._write_gdn_slot_from_scratch = write


# ---------------------------------------------------------------------------------------------- bring-up / warm-up
def bring_up_ms(srv, NS):
    """Called by df_server.bring_up after the model, KV pool and drafter exist."""
    global SPEC_KCAP
    S = srv.S
    import itertools

    if SPEC_KCAP:
        # r64t: stream-count-aware depth. One verifier per depth k = 1..KCAP, every one at R = NS * k <= SPEC_ROWS
        # rows (slot s owns rows s*k .. s*k+k-1); all verifiers share the streams' chain state, KV pages, drafters'
        # token buffers and context. Within a pass a non-live slot is a stale-stream row exactly as in the padded
        # 64-row pass of Stage C (bit-inert for the live rows, measured there).
        plan = []
        for it in os.environ.get("DF_SPEC_PLAN", "").split(","):
            it = it.strip()
            if ":" in it:
                a, b = it.split(":")
                plan.append((min(int(a), NS), int(b)))
        if not plan:  # default: every live count 1..NS at its scheduled depth
            plan = [(L, max(1, min(int(SPEC_KCAP), SPEC_ROWS // L))) for L in range(1, NS + 1)]
        kcap = max(b for _a, b in plan)
        for a, b in plan:
            assert a * b <= SPEC_ROWS, (a, b, SPEC_ROWS)
            assert a * b <= int(S.model.args.max_batch_size), (a, b, S.model.args.max_batch_size)
        SPEC_KCAP = kcap  # the effective depth = the chain-state capacity every stream allocates
        ms = MsVerifier(S.model, S.tap_ids, S.page_rows, log=srv.log, k=1)  # R = NS rows (the k=1 verifier's parent)
        ms.kcap = kcap
        ms.nstreams = NS
        ms.chains = None
        S.ms = ms
        S.ver = ms
        if BEGIN_DIRECT:
            install_begin_direct(S.model, ms)
        if S.drafter is not None:
            S.drafters = make_drafters(S.drafter, NS, ms.tokbufs)
            S.drafter = S.drafters[0]
        ms.plans = sorted(set(plan))
        ms.kver = {(a, b): MsVerifier(S.model, S.tap_ids, S.page_rows[:a], log=srv.log, k=b,
                                      streams=ms.streams[:a], tokbufs=ms.tokbufs[:a]) for a, b in ms.plans}
        ms.sub = {}
        ms.bdrafts = {}
        ms.bdraft, ms.bgroups = None, []
        if S.drafter is not None and NS > 1:
            # r64t: the batched drafter's all-reduce needs 8 * (streams in the graph) rows on the model's residual grid,
            # which is 64 rows (the model is built at 64). So draft in groups of at most 8 streams; a group with no
            # live stream is skipped.
            if NS <= 8:
                ms.bdraft = MsDraft(S.drafters)
                # ms4 DF_MS_BDPREFIX="2,4" (default off): also batched drafter graphs over the first n slots only
                # (R = 8 n rows instead of 8 NS = 64); a step whose live slots all lie below n uses the smallest one.
                for n_ in [int(x) for x in os.environ.get("DF_MS_BDPREFIX", "").split(",") if x.strip()]:
                    if 2 <= n_ < NS:
                        ms.bdrafts[tuple(range(n_))] = MsDraft(S.drafters[:n_])
            else:
                for g0 in range(0, NS, 8):
                    g = tuple(range(g0, min(g0 + 8, NS)))
                    ms.bgroups.append((g, MsDraft([S.drafters[s] for s in g])))
        srv.log(f"[ms] SPEC: KCAP {kcap}, {NS} slots, verifiers " +
                ", ".join(f"{a}x{k}->R={a * k}" for a, k in ms.plans))
        if PLAIN_LANE:
            build_plain(ms, srv, NS)
        return ms

    if CHAIN > 0 and NS > CHAIN and int(S.model.args.max_batch_size) < NS * T:
        # r64d Stage B: model at CHAIN*T rows, NCHAIN = NS/CHAIN chains. Every verifier is one chain's live subset
        # (<= CHAIN*T rows); the streams' per-stream GDN state / KV pages are independent, so the chains share the
        # model and never run a >32-row step. Each chain verifier has its own R-row inputs and trace.
        chains = [tuple(range(i, min(i + CHAIN, NS))) for i in range(0, NS, CHAIN)]
        head = chains[0]
        ms = MsVerifier(S.model, S.tap_ids, [S.page_rows[s] for s in head], log=srv.log)
        ms.chains = chains
        for s in range(len(head), NS):  # streams of the other chains (per-stream state, not model rows)
            ms.streams.append(StreamState(ms, s))
            ms.tokbufs.append(_dev(torch.zeros(T, 1, dtype=torch.int32), ms.mesh, ttnn.uint32))
        S.ms = ms
        S.ver = ms
        if BEGIN_DIRECT:
            install_begin_direct(S.model, ms)
        if S.drafter is not None:
            S.drafters = make_drafters(S.drafter, NS, ms.tokbufs)
            S.drafter = S.drafters[0]
        for ch in chains:
            for n in range(1, len(ch) + 1):
                for sub in itertools.combinations(ch, n):
                    ms.sub[sub] = MsVerifier(S.model, S.tap_ids, [S.page_rows[s] for s in sub], log=srv.log,
                                             streams=[ms.streams[s] for s in sub], tokbufs=[ms.tokbufs[s] for s in sub])
        ms.bdraft, ms.bdrafts = None, {}
        ms.nstreams = NS  # ms.S is one chain (the verifier); the worker admits all NS streams
        srv.log(f"[ms] CHAIN {CHAIN}: {len(chains)} chains {chains}, {len(ms.sub)} subset verifiers, "
                f"max {CHAIN * T} rows, model batch {S.model.args.max_batch_size}, {NS} stream slots")
        return ms

    ms = MsVerifier(S.model, S.tap_ids, S.page_rows, log=srv.log)
    S.ms = ms
    S.ver = ms
    if BEGIN_DIRECT:
        install_begin_direct(S.model, ms)
    if S.drafter is not None:
        S.drafters = make_drafters(S.drafter, NS, ms.tokbufs)
        S.drafter = S.drafters[0]
    mode = os.environ.get("DF_MS_SUB", "all")  # all: every proper subset of the streams; solo: 1 stream; none
    import itertools

    if CHAIN > 0 and NS > CHAIN:
        # r64d Stage B: two chains of CHAIN streams. Every verifier is a subset WITHIN one chain (<= CHAIN*8 rows);
        # the full NS-stream verifier (NS*8 rows) is never built or run - above 32 rows the model is not row-exact.
        ms.chains = [tuple(range(i, min(i + CHAIN, NS))) for i in range(0, NS, CHAIN)]
        for ch in ms.chains:
            for n in range(1, len(ch) + 1):
                for sub in itertools.combinations(ch, n):
                    ms.sub[sub] = MsVerifier(S.model, S.tap_ids, [S.page_rows[s] for s in sub], log=srv.log,
                                             streams=[ms.streams[s] for s in sub], tokbufs=[ms.tokbufs[s] for s in sub])
        srv.log(f"[ms] chains {ms.chains}: {len(ms.sub)} subset verifiers, max {CHAIN * T} rows")
    elif os.environ.get("DF_MS_SUBSETS", "").strip():
        # r64g: build exactly the listed stream groups ("0,1,2,3;4,5,6,7"), so a 64-row full verifier can coexist with
        # named narrower ones (the 64-row pass is exact, A5; a live set with no entry runs the full pass at R = NS*T,
        # its dead rows padded - rows are bit-invariant across widths with fused attention at every width).
        for grp in os.environ["DF_MS_SUBSETS"].split(";"):
            sub = tuple(int(x) for x in grp.split(",") if x.strip() != "")
            if sub:
                ms.sub[sub] = MsVerifier(S.model, S.tap_ids, [S.page_rows[s] for s in sub], log=srv.log,
                                         streams=[ms.streams[s] for s in sub], tokbufs=[ms.tokbufs[s] for s in sub])
        srv.log(f"[ms] explicit subset verifiers {[list(g) for g in ms.sub]}, full R={ms.R}")
    elif mode != "none":
        # ms1o: a step verifies only the live streams (R = 8 per live stream; 1 live = the single-stream cost), each
        # subset verifier on the streams' shared state
        sizes = range(1, NS) if mode == "all" else (1,)
        for n in sizes:
            for sub in itertools.combinations(range(NS), n):
                ms.sub[sub] = MsVerifier(S.model, S.tap_ids, [S.page_rows[s] for s in sub], log=srv.log,
                                         streams=[ms.streams[s] for s in sub], tokbufs=[ms.tokbufs[s] for s in sub])
    ms.bdraft = MsDraft(S.drafters) if (BD_MIN and S.drafter is not None) else None
    ms.bdrafts = {}
    if ms.bdraft is not None and os.environ.get("DF_MS_BDSUB", "1") == "1":
        # ms1o: a batched drafter graph per proper subset of >= BD_MIN live streams (2 live: 8 + 8 rows, not 32)
        # r64d: with chains, only subsets within a chain (a group that runs in one verify pass).
        pools = getattr(ms, "chains", [tuple(range(NS))])
        for ch in pools:
            for n in range(max(2, BD_MIN), len(ch) + 1):
                for sub in itertools.combinations(ch, n):
                    ms.bdrafts[sub] = MsDraft([S.drafters[s] for s in sub])
    srv.log(f"[ms] {len(ms.sub)} subset verifiers ({mode}), prestage {PRESTAGE}, batched drafter at >= {BD_MIN} live")
    return ms


def _fill(srv, s, L):
    """Prefill a synthetic prompt of L tokens into slot s; stream s ready for its first step. Returns (first, L)."""
    S = srv.S
    ms, ds = S.ms, S.drafters
    ids = [1000 + (i % 5000) + 7 * s for i in range(L)]
    logits, taps, tap0 = srv.prefill(ids, slot=s, page_row=S.page_rows[s])
    ms.begin_sequence(s)
    ds[s].prefill_context(taps, list(range(tap0, L)))
    return int(torch.argmax(logits)), L


def warmup_ms(srv, lens):
    """Every program shape compiled eagerly (prefill buckets + tap copies for each length, S eager drafter graphs, one
    eager verify), then the verify trace and the S drafter traces (each after a fresh prefill, as the single-stream
    warm-up does)."""
    S = srv.S
    ms, ds = S.ms, S.drafters
    NS = len(getattr(ms, "streams", [])) or ms.S  # r64d: chains extend ms.streams to every stream (ms.S = one chain)
    t0 = time.time()
    for i, L in enumerate(lens):
        if L + 2 * T + 64 >= S.max_len:
            continue
        _fill(srv, i % NS, L)
        srv.log(f"[ms] warmup prefill {L} -> slot {i % NS} ({time.time() - t0:.0f}s)")
    st = [_fill(srv, s, 64 + 40 * s) for s in range(NS)]
    for s, d in enumerate(ds):  # eager drafter graphs (each writes its token buffer)
        d._stage(st[s][0], st[s][1])
        d.outs = d._graph()
    if ms.bdraft is not None or getattr(ms, "bgroups", None):
        if ms.bdraft is not None:
            ms.bdraft.run_eager()
        for _g, bd in getattr(ms, "bgroups", []):
            for s in _g:
                ds[s]._stage(st[s][0], st[s][1])
            bd.run_eager()
        for bd in ms.bdrafts.values():
            bd.run_eager()
        ttnn.synchronize_device(S.mesh)
        srv.log(f"[ms] batched drafter eager 1 done ({time.time() - t0:.0f}s)")
    npc = lambda: S.mesh.num_program_cache_entries() if hasattr(S.mesh, "num_program_cache_entries") else -1
    _full = not (CHAIN > 0 and NS > CHAIN) and not getattr(ms, "kver", None)  # r64d / r64t: no NS*8-row pass
    if _full:
        ms.run_eager([x[1] for x in st])
    else:
        ms.__dict__.setdefault("_skipped_full", True)
    ttnn.synchronize_device(S.mesh)
    srv.log(f"[ms] eager step 1 done ({time.time() - t0:.0f}s) cache {npc()} full={_full}")
    st = [_fill(srv, s, 64 + 40 * s) for s in range(NS)]
    for s, d in enumerate(ds):
        d._stage(st[s][0], st[s][1])
        d.outs = d._graph()
    if ms.bdraft is not None:
        ms.bdraft.run_eager()
    for _g, bd in getattr(ms, "bgroups", []):
        bd.run_eager()
    for bd in ms.bdrafts.values():
        bd.run_eager()
    if _full:
        ms.run_eager([x[1] for x in st])
    ttnn.synchronize_device(S.mesh)
    srv.log(f"[ms] eager step 2 done ({time.time() - t0:.0f}s) cache {npc()}")
    kver = getattr(ms, "kver", None) or {}
    for rep in range(2) if (ms.sub or kver) else ():  # the subset verifiers, eagerly (all buffers before any capture)
        for sub, v in ms.sub.items():
            v.run_eager([st[s][1] for s in sub])
        for (a, _b), v in kver.items():  # r64t: every verifier, over its own slots
            v.run_eager([st[s][1] for s in range(a)])
        ttnn.synchronize_device(S.mesh)
        srv.log(f"[ms] subset eager pass {rep + 1} done ({time.time() - t0:.0f}s) cache {npc()}")
    st = [_fill(srv, s, 64 + 40 * s) for s in range(NS)]
    srv.log(f"[ms] cache before capture {npc()}")
    if _full:
        ms.capture_trace([x[1] for x in st])
    for sub, v in ms.sub.items():
        v.capture_trace([st[s][1] for s in sub])
    for (a, _b), v in kver.items():  # r64t
        v.capture_trace([st[s][1] for s in range(a)])
    if getattr(ms, "plain", None):
        warm_plain(ms, srv, st)  # r64u: the plain lane's traces
    if os.environ.get("DF_BEGIN_TRACE", "0") == "1" and not BEGIN_DIRECT:
        # r64d: with DF_MS_BEGIN_DIRECT the prefill writes the streams' base/hist, begin_sequence returns early and
        # these traces are never executed; they also slice the model state rows, which DF_MODEL_GDN_B shrinks.
        ms.capture_begin_traces()
    srv.log(f"[ms] cache after capture {npc()}")
    st = [_fill(srv, s, 64 + 40 * s) for s in range(NS)]
    gcb_up()  # the drafter traces are captured (and always replayed) with the prefetcher running
    for s, d in enumerate(ds):
        d.capture_trace(st[s][0], st[s][1])
    if ms.bdraft is not None or getattr(ms, "bgroups", None):
        for s, d in enumerate(ds):
            d._stage(st[s][0], st[s][1])
        if ms.bdraft is not None:
            ms.bdraft.capture_trace()
        for _g, bd in getattr(ms, "bgroups", []):
            bd.capture_trace()
        for bd in ms.bdrafts.values():
            bd.capture_trace()
        srv.log(f"[dflash-ms] batched drafter traces captured ({1 + len(ms.bgroups)} graphs)")
    ttnn.synchronize_device(S.mesh)
    # one traced step per stream count as a smoke test (the single-stream warm-up runs a traced request too)
    for slots in [list(sub) for sub in ms.sub] + ([list(range(NS))] if _full else []):
        reqs = {s: srv.Req([1000 + (i % 5000) + 11 * s for i in range(80 + 30 * s)], max_new=3 * T, min_new=0,
                           ignore_eos=True) for s in slots}
        active = [admit(ms, ds, s, reqs[s]) if s in reqs else None for s in range(NS)]
        while any(a is not None for a in active):
            step(ms, ds, active)
        srv.log(f"[ms] warmup traced slots {slots}: {[len(r.out) for r in reqs.values()]} tokens, subset steps {step.sub}, batched drafts {getattr(step, 'bd', 0)}")
    for (a, b) in sorted(kver):  # r64t: one traced step per verifier, through step() (draft, accept, commit)
        ms.force_k = b
        reqs = {s: srv.Req([1000 + (i % 5000) + 11 * s for i in range(80 + 3 * s)], max_new=3 * T, min_new=0,
                           ignore_eos=True) for s in range(a)}
        active = [admit(ms, ds, s, reqs[s]) for s in range(a)] + [None] * (NS - a)
        while any(x is not None for x in active):
            step(ms, ds, active)
        srv.log(f"[ms] warmup {a} slots at depth {b} (R={a * b}): {[len(r.out) for r in reqs.values()]} tokens")
        ttnn.synchronize_device(S.mesh)
    if getattr(ms, "plain", None):
        warm_plain_steps(ms, srv)  # r64u: the migration path + the lane, end to end before READY
    ms.force_k = None
    ttnn.synchronize_device(S.mesh)
    srv.log(f"[ms] warmup done in {time.time() - t0:.0f}s")

# ======================================================================================== r64u: the batched plain lane
# Stage T's finding 4: at k = 1 this engine is NOT the plain engine - the chain path runs one sequential GDN program per
# stream (82 ms/step at 8 live against plain decode's 30). Above the crossover (6 live) the step must hand every stream
# to the model's OWN batched decode: one fused GDN program over all rows, the per-row paged KV write, one batched paged
# SDPA. That is what this lane is. It is a separate engine on the same model, KV pages and stream slots, selected by the
# live stream count with no operator input (DF_PLAIN_LANE=1 arms it; the switch itself is automatic).


class PlainVerifier(MsVerifier):
    """r64u: the batched plain-decode lane. T = 1 (one token per stream slot), R = the bucket width (<= 32 rows).

    Unlike the speculative verifier it installs NO patches: dn.forward_decode is the model's native fused GDN decode
    (rows are independent users, state rows [0:R] of rec_state / conv_states), paged_update_cache writes one row per
    position, and paged_scaled_dot_product_attention_decode runs over R rows. Dead rows (a slot with no live stream)
    carry a scratch token at position 0: they write their own page row at position 0 and their own GDN state row, both
    of which belong to an idle slot whose next request re-prefills over them.
    """

    plain = True

    # the model's own batched GDN decode and batched SDPA (no chain, no per-stream split); only the KV write is taken
    # from the speculative verifier, whose fused masked write is exact where the native per-row one is not (see
    # MsVerifier._patched)
    patch_gdn = False
    patch_sdpa = False
    patch_kv = True

    def __init__(self, model, page_rows, streams, log=print):
        super().__init__(model, [], page_rows, log=log, streams=streams, k=1)
        self.restage_pt = True  # r64v: the lane's device page table came out as a broadcast of row 0
        self.tap_pos = []
        self.tokbufs = [_dev(torch.zeros(1, 1, dtype=torch.int32), self.mesh, ttnn.uint32) for _ in range(self.S)]

    def _forward_ms(self, cur_pos, rope, page_table):
        toks = self.tokbufs[0] if self.S == 1 else ttnn.concat(self.tokbufs, dim=0)  # [R, 1]
        ids, logits, _taps, rm = Verifier._forward(self, toks, cur_pos, rope, page_table)
        return (ids, logits, None, rm), toks, None

    def set_token(self, s, tok):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[int(tok)]], dtype=torch.int32), dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT),
            self.tokbufs[s])


def pick_plain(ms, live):
    """The smallest built bucket that covers this step's live slots (rows = slots, one row each)."""
    need = max([1] + [s + 1 for s in live])
    for R in sorted(ms.plain):
        if R >= need:
            return ms.plain[R]
    return None


def migrate_chain_to_rows(ms, s):
    """r64u: hand stream s's GDN state from its speculative chain buffers to the model's batched decode rows
    (rec_state row s / conv_states[m] row s). Pure data movement - the plain lane starts from exactly the state the
    chain held.

    Where that state IS depends on the commit convention (DF_NOCOMMIT): after a step that accepted `a` rows the
    recurrent state is in bufs[a] (base is the state at the START of that step) and the conv history is rows
    a+1 .. a+3 of the persistent window Xp (hist is only the prefill's history, the window carries it per step - see
    kernels/dfwin_reader.cpp: a = the `a` tensor, hrow = a + 1). A freshly admitted stream (pending_a None) still has
    its state in base / hist, written by the prefill. conv_states[0] (the shifted-out tap) is zeroed: the native shift
    register reads it.

    The writes go through dn._write_index (slice + concat + copy into the FULL buffer): ttnn has no in-place row write,
    so a ttnn.copy into a slice view of rec_state / conv_states is silently lost."""
    st = ms.streams[s]
    a = st.pending_a
    if WY and st.proto == "wy" and a is not None:
        wy_commit(ms, st)  # base <- S_a (the bufs are not written by the WY engine)
    wyb = WY and st.proto == "wy" and a is not None
    for dn in ms.gdn:
        if a is None:
            # a fresh admission: with DF_MS_BEGIN_DIRECT the prefill redirected the model slot write into the chain
            # buffers, so base / hist ARE the post-prompt state (rec_state rows hold warm-up leftovers - measured pcc 0.49
            # against base, and the rows are never written while BEGIN_DIRECT is on)
            src, h = st.base[id(dn)], st.hist[id(dn)]
        else:
            src = st.base[id(dn)] if wyb else st.bufs[id(dn)][int(a)]
            h = ttnn.slice(st.Xp[id(dn)], (0, int(a) + 1, 0), (1, int(a) + dn.K, dn.qkv_dim_tp))
        dn._write_index(dn.rec_state, ttnn.clone(src), s, dim=0)
        dn._write_index(dn.conv_states[0], ttnn.clone(dn._slice_along(dn._zero_conv0, 1, s, s + 1)), s, dim=1)
        for m in range(1, dn.K):
            row = ttnn.slice(h, (0, m - 1, 0), (1, m, dn.qkv_dim_tp))
            dn._write_index(dn.conv_states[m], row, s, dim=1)
        if a is not None:
            ttnn.deallocate(h)


def step_plain(ms, ds, active):
    """r64u: one batched plain-decode pass over every live stream (1 token per stream, no drafter, no verify depth)."""
    srv = srv_()
    S = srv.S
    live = [s for s, a in enumerate(active) if a is not None]
    pv = pick_plain(ms, live)
    assert pv is not None, (live, sorted(ms.plain))
    R = pv.S
    ta = time.perf_counter()
    gcb_up()
    for s in live:
        pv.set_token(s, active[s].out[-1])
    starts = [active[s].start if active[s] is not None else 0 for s in range(R)]
    if PCACHE:
        pv._live = set(live)  # ep2 dfpc: dead rows of the lane skip their KV write
    tb = time.perf_counter()
    _blocks, ids = pv.run_chained(starts)
    tc = time.perf_counter()
    host = pv.host_logits() if any(active[s].req.pen is not None for s in live) else None
    for s in live:
        a = active[s]
        req = a.req
        n_out = len(a.out)
        if host is not None:
            t = srv.host_pick(host[s], n_out, req)
        else:
            t = ids[s]
            if not req.ignore_eos and n_out < req.min_new and t in S.eos:
                lg = pv.host_logits() if host is None else host
                row = lg[s].clone()
                for e in S.eos:
                    row[e] = -float("inf")
                t = int(torch.argmax(row))
        a.out.append(t)
        a.start += 1
        a.steps += 1
        a.acc.append(1)
        if req.q:
            req.q.put([t])
        stop = (not req.ignore_eos) and t in S.eos and len(a.out) >= req.min_new
        if stop or len(a.out) >= req.max_new or a.start + 1 >= S.max_len:
            finish(a)
            active[s] = None
    step.t = (tb - ta, tc - tb, time.perf_counter() - tc)
    step.n = getattr(step, "n", 0) + 1
    step.nplain = getattr(step, "nplain", 0) + 1
    if step.n % 20 == 1 or os.environ.get("DF_MS_VERBOSE") == "1":
        srv.log(f"[ms] plain step {step.n}: token {1e3 * step.t[0]:.1f} verify {1e3 * step.t[1]:.1f} "
                f"post {1e3 * step.t[2]:.1f} ms; live {live} R {R}")


def build_plain(ms, srv, NS):
    """r64u: one PlainVerifier per bucket width (rows = stream slots, one row each, <= SPEC_ROWS = one tile row)."""
    S = srv.S
    ms.plain, ms.plain_mode, ms.plain_live = {}, False, PLAIN_LIVE
    for R in sorted(set(PLAIN_BUCKETS)):
        if R < 1 or R > min(SPEC_ROWS, NS):
            continue
        ms.plain[R] = PlainVerifier(S.model, S.page_rows[:R], ms.streams[:R], log=srv.log)
    srv.log(f"[ms] PLAIN lane: buckets {sorted(ms.plain)} rows, automatic switch at {PLAIN_LIVE} live streams")
    return ms


def warm_plain(ms, srv, st):
    """r64u: eager pass + trace per bucket (call after the per-slot fills, before any capture of another verifier)."""
    S = srv.S
    for R, pv in sorted((getattr(ms, "plain", None) or {}).items()):
        pv.run_eager([st[s][1] for s in range(R)])
        ttnn.synchronize_device(S.mesh)
        pv.capture_trace([st[s][1] for s in range(R)])
        srv.log(f"[ms] plain bucket R={R} eager + trace done")


def warm_plain_steps(ms, srv):
    """r64u: one traced plain request per bucket (migration path included), as the warm-up's verifier smoke test."""
    S = srv.S
    ds = S.drafters
    NS = getattr(ms, "nstreams", ms.S)
    for R, pv in sorted((getattr(ms, "plain", None) or {}).items()):
        reqs = {s: srv.Req([1000 + (i % 5000) + 13 * s for i in range(80 + 3 * s)], max_new=12, min_new=0,
                           ignore_eos=True) for s in range(R)}
        active = [admit(ms, ds, s, reqs[s]) if s < R else None for s in range(NS)]
        for s in range(R):
            migrate_chain_to_rows(ms, s)  # exactly what the automatic lane switch does
        ttnn.synchronize_device(S.mesh)
        while any(x is not None for x in active):
            step_plain(ms, ds, active)
        srv.log(f"[ms] warmup plain R={R}: {[len(r.out) for r in reqs.values()]} tokens")
        ttnn.synchronize_device(S.mesh)

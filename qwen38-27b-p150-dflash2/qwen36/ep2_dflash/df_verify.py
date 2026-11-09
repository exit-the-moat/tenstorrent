"""DFlash2 verify pass for Qwen3.8-27B on the qwen36 TP model code: T = 8 slots of ONE sequence in one forward.

Episode 2 night 1. Runs the model's own decode code with T "pseudo-user" rows (row i = token start+i), with three
changes made per call and undone afterwards (nothing in the model files is edited):

- GDN layers: `forward_decode` replaced by `_gdn_verify`: the conv runs over [3 history rows; T new rows] with the
  same multiply/mac order as decode, then the decode recurrence (B=1, the same function plain decode calls) once per
  row, chained. The state after row i lands in state row i (the batched GDN buffer has max_batch_size = T rows, so
  it doubles as the per-slot stash); `commit(a)` copies row a to row 0 and the conv history from the stashed window.
- full attention: `paged_update_cache` is called one row at a time (the T rows alias one page table, and a batched
  update would read-modify-write the same KV tile from several cores); SDPA either batched over the T rows with the
  aliased page table (DF_SDPA_ROWS=0) or one row at a time (DF_SDPA_ROWS=1, the B=1 program plain decode runs).
- the residual stream after the 5 tap layers is kept (all-gathered to full width) for the drafter, and the LM head's
  argmax runs on the device, so T ids cross PCIe.

Everything is fixed-shape, so the pass can be traced (DF_TRACE=1).
"""
import contextlib
import functools
import os
import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn import tp as gdn_tp
from models.demos.blackhole.qwen36.tt.gdn.tp import _silu_mul, _softplus_add
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce
from models.tt_transformers.tt.common import copy_host_to_device

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _rows_kv_cfg(hd, core=(0, 0)):
    """One KV row [1,1,32,hd] height-sharded on one core (what plain decode's B=1 KV write uses at (0,0))."""
    if core == (0, 0):
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, hd),
            core_grid=ttnn.CoreGrid(x=1, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    c = ttnn.CoreCoord(*core)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(c, c)}), (ttnn.TILE_SIZE, hd), ttnn.ShardOrientation.ROW_MAJOR),
    )


def dev_rows(t, mesh):
    """Per-device host copies of a mesh tensor via ConcatMeshToTensor(dim 0) (no get_device_tensors: on 2 cards a
    to_torch of a get_device_tensors() shard is the suspect of the 2026-11-02 03:52 segfault in from_device)."""
    nd = mesh.get_num_devices()
    if nd == 1:
        return [ttnn.to_torch(t)]
    h = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return list(torch.chunk(h, nd, dim=0))


class Verifier:
    def __init__(self, model, tap_layer_ids, page_table_row, T=8, log=print):
        self.model = model
        self.mesh = model.mesh_device
        self.T = T
        self.log = log
        self.nd = model.num_devices
        self.gdn = [l.attention for l in model.layers if not l.is_full_attention]
        self.attn = [l.attention for l in model.layers if l.is_full_attention]
        for dn in self.gdn:
            assert dn.B == T, f"GDN batch rows {dn.B} != T {T}: start the model with max_batch_size={T}"
        idx = list(model.layer_indices)
        self.tap_pos = [idx.index(t) for t in tap_layer_ids if t in idx]
        self.page_table_row = page_table_row.to(torch.int32).reshape(1, -1)
        self.sdpa_rows = os.environ.get("DF_SDPA_ROWS", "0") == "1"
        self.kv_skip = os.environ.get("DF_KV_SKIP", "1") == "1"
        self._kvm = None
        self.shard_argmax = self.nd > 1 and model._lmhead_vocab_sharded and os.environ.get("DF_SHARD_ARGMAX", "1") == "1"
        self.hist = {}
        self.X = {}
        for dn in self.gdn:
            self.hist[id(dn)] = ttnn.from_torch(
                torch.zeros(1, dn.K - 1, dn.qkv_dim_tp, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=_DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        self._alloc_fused_stash()
        self._alloc_chain()
        self.inputs = None  # persistent device inputs (tokens, cur_pos, rope, page_table)
        self._pt_host = None
        self.trace_id = None
        self.out = None
        self.timing = {}

    # ------------------------------------------------------------------ state handover
    def _alloc_chain(self):
        """DF_GDN_CHAIN (default on with the fused GDN program): per GDN layer one persistent `base` state buffer."""
        self.chain = os.environ.get("DF_GDN_CHAIN", "1") == "1" and any(self._fused(dn) for dn in self.gdn)
        # ep2 t3 DF_GDN_XROW=1: slot i = ONE program (the B = 1 fused GDN reading row i of the in-projection, state
        # bufs[i-1] -> bufs[i], gated row i of one [1, T, Nv*128] output): no row slice / tilize / untilize, no state
        # clone, no concat. The T state buffers per layer are persistent (allocated here, outside any trace).
        self.xrow = self.chain and os.environ.get("DF_GDN_XROW", "1") == "1"
        # ep2 t3 DF_GDN_XCONV=1 (with XROW): the fused program reads slot i's conv history straight from the verify
        # window X (rows i .. i+2) and never shifts the conv-state tensors: the commit keeps only the state copy and
        # the history slice (no conv_states copies)
        self.xconv = self.xrow and os.environ.get("DF_GDN_XCONV", "1") == "1"
        # ep2 t3 DF_X_PERSIST=1 (with XCONV): the verify window X lives in one persistent DRAM tensor per layer (the
        # window is built in L1, the kernel reads it there, then one copy to the persistent X): the commit traces then
        # read only persistent tensors and survive a verify re-capture (no commit re-capture per request)
        self.xpersist = self.xconv and os.environ.get("DF_X_PERSIST", "1") == "1"
        # ep2 t3 DF_GDN_CHAIN1=1 (with XCONV): the T slots as ONE program (kernels/gdnq4c_*): taps / selectors / gates
        # constants read once, the state handed from slot to slot through bufs inside the program
        self.chain1 = self.xconv and os.environ.get("DF_GDN_CHAIN1", "1") == "1"
        # ep2 t3 DF_COMMIT_K=1 (with X_PERSIST): the commit of a GDN layer = ONE data-movement program
        # (kernels/dfcommit_reader.cpp: base <- bufs[a], hist <- X rows a+1..a+3) instead of copy + slice + copy
        self.commit_k = self.xpersist and os.environ.get("DF_COMMIT_K", "1") == "1"
        # ep2 t3b DF_NOCOMMIT=1 (with DF_XWIN, traced): no commit program per step: the next verify starts slot 0 from
        # bufs[a] and takes the conv history from the window rows a + 1 .. a + 3 (a = element T*T of the KV-skip
        # position input; T + 247 = after a commit / a new sequence: base and hist)
        self.nocommit = self.commit_k and self.kv_skip and os.environ.get("DF_NOCOMMIT", "0") == "1"
        self.pending_a = None
        self._ck = {}
        self.Xp = {}
        self.bufs = {}
        self.base = {}
        if not self.chain:
            return
        for dn in self.gdn:
            z = torch.zeros(1, dn.Nv, dn.Dk, dn.Dv, dtype=torch.float32)
            mk = lambda: ttnn.from_torch(z, dtype=dn.rec_state.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                         memory_config=_DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
            self.base[id(dn)] = mk()
            if self.xrow:
                self.bufs[id(dn)] = [mk() for _ in range(self.T)]

    begin_tid = None

    def capture_begin_trace(self):
        """ep2 t3b DF_BEGIN_TRACE=1: begin_sequence's copies as one trace (all sources / targets are persistent)."""
        self.begin_sequence()  # eager first: every program compiled
        ttnn.synchronize_device(self.mesh)
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self._begin_ops()
        finally:
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        self.begin_tid = tid
        self.log("[dflash] begin_sequence trace captured")

    def begin_sequence(self):
        """After the prompt prefill (slot 0): conv history <- conv_states[1..K-1] row 0 (+ chain base <- state row 0)."""
        self.pending_a = None
        if self.begin_tid is not None:
            ttnn.execute_trace(self.mesh, self.begin_tid, cq_id=0, blocking=False)
            return
        self._begin_ops()

    def _begin_ops(self):
        for dn in self.gdn:
            if getattr(self, "chain", False):
                r0 = dn._slice_along(dn.rec_state, 0, 0, 1)
                ttnn.copy(r0, self.base[id(dn)])
                ttnn.deallocate(r0)
            rows = [dn._slice_along(dn.conv_states[m], 1, 0, 1) for m in range(1, dn.K)]  # each [1,1,D]
            h = ttnn.concat(rows, dim=1)
            ttnn.copy(h, self.hist[id(dn)])
            ttnn.deallocate(h)
            for r in rows:
                ttnn.deallocate(r)

    commit_tids = None

    def capture_commit_traces(self):
        """One trace per accepted slot a (each a is a different slice program); verify trace first (X fixed)."""
        assert self.trace_id is not None
        self.commit_tids = {}
        for a in range(self.T):
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._commit_ops(a)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.commit_tids[a] = tid
        ttnn.synchronize_device(self.mesh)
        self.log(f"[dflash] {self.T} commit traces captured")

    def release_traces(self):
        for tid in (self.commit_tids or {}).values():
            ttnn.release_trace(self.mesh, tid)
        self.commit_tids = None
        if self.begin_tid is not None:
            ttnn.release_trace(self.mesh, self.begin_tid)
            self.begin_tid = None
        self.release_trace()

    def commit(self, a):
        """Keep the state after slot a (0-based): GDN row a -> row 0, conv history = inputs a-2..a."""
        if self.nocommit and self.trace_id is not None and not self._capturing:
            self.pending_a = int(a)  # the next (traced) verify reads bufs[a] / the window rows itself
            return
        self.pending_a = None
        if self.commit_tids:
            ttnn.execute_trace(self.mesh, self.commit_tids[a], cq_id=0, blocking=False)
            return
        self._commit_ops(a)

    def _commit_ops(self, a):
        for dn in self.gdn:
            if self._fused(dn) is not None:
                self._commit_fused(dn, a)
                continue
            X = self.X[id(dn)]
            if a > 0:
                row = dn._slice_along(dn.rec_state, 0, a, a + 1)
                dn._write_recurrent_state_prefix(row, 1)
            h = ttnn.slice(X, (0, a + 1, 0), (1, a + dn.K, dn.qkv_dim_tp))
            ttnn.copy(h, self.hist[id(dn)])
            ttnn.deallocate(h)

    # ------------------------------------------------------------------ GDN verify on the fused decode kernel
    def _fused(self, dn):
        """dn's fused one-token GDN program (ep2 kernels agent, QWEN36_GDN_FUSED_DECODE=1) or None."""
        fd = getattr(dn, "_fused_decode", None)
        return fd if (fd is not None and os.environ.get("QWEN36_GDN_FUSED_DECODE") == "1") else None

    def _alloc_fused_stash(self):
        """Per GDN layer, T-1 state slots [1, Nv, Dk, Dv] (the state after slots 0..T-2; slot T-1 stays in row 0)."""
        self.stash = {}
        if os.environ.get("DF_GDN_CHAIN", "1") == "1" and os.environ.get("DF_GDN_XROW", "1") == "1":
            return  # ep2 t3: the stash is only read without the chain; XROW allocates its own T buffers per layer
        for dn in self.gdn:
            if self._fused(dn) is None:
                continue
            z = torch.zeros(1, dn.Nv, dn.Dk, dn.Dv, dtype=torch.float32)
            self.stash[id(dn)] = [
                ttnn.from_torch(z, dtype=dn.rec_state.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                memory_config=_DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
                for _ in range(self.T - 1)
            ]

    def _gdn_verify_fused(self, dn, x, fd):
        """T chained calls of the fused one-token program (B=1, row 0 in place) = exactly what plain decode runs per
        token, on rows of ONE in-projection matmul; the state after each slot is copied to a stash slot."""
        tw = dn.tw
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        T = x.shape[-2]
        D = dn.qkv_dim_tp
        qkvzab = dn._qkvzab_proj(x, T, _L1)
        W = qkvzab.shape[-1]
        xwin = self.xpersist and self.xrow and self.xconv and os.environ.get("DF_XWIN", "0") == "1"
        qkv = None if xwin else ttnn.slice(qkvzab, (0, 0, 0), (1, T, D))
        # conv inputs [hist(3); slots(T); zeros(T)] -- the zero tail lets commit copy an 8-row window straight into a
        # [1, Bmax, D] conv-state tensor (only row 0 is live in single-user serving)
        if xwin:
            # ep2 t3b DF_XWIN=1: the window built by one data-movement program (kernels/dfwin_reader.cpp), into the
            # L1 window and the persistent DRAM window at once
            shp = ttnn.Shape([1, dn.K - 1 + 2 * T, D])
            if id(dn) not in self.Xp:  # first (eager) verify, before any trace
                self.Xp[id(dn)] = ttnn.from_torch(torch.zeros(1, dn.K - 1 + 2 * T, D, dtype=torch.bfloat16),
                                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                                  memory_config=_DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
            Xl = ttnn.allocate_tensor_on_device(shp, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, _L1)
            wio = [self.hist[id(dn)], qkvzab, Xl, self.Xp[id(dn)]] + ([self._kvm] if self.nocommit else [])
            ttnn.generic_op(wio, self._win_prog(dn, qkvzab, Xl))
            X = self.X[id(dn)] = self.Xp[id(dn)]
        elif self.xpersist:
            qx = qkv if os.environ.get("DF_X_DIRECT", "1") == "1" else ttnn.to_memory_config(qkv, _DRAM)
            Xl = ttnn.concat([self.hist[id(dn)], qx, self._ztail(dn)], dim=1, memory_config=_L1)
            if qx is not qkv:
                ttnn.deallocate(qx)
            ttnn.deallocate(qkv)
            if id(dn) not in self.Xp:  # first (eager) verify, before any trace
                self.Xp[id(dn)] = ttnn.to_memory_config(Xl, _DRAM)
            else:
                ttnn.copy(Xl, self.Xp[id(dn)])
            X = self.X[id(dn)] = self.Xp[id(dn)]
        else:
            Xl = None
            X = ttnn.concat([self.hist[id(dn)], ttnn.to_memory_config(qkv, _DRAM), self._ztail(dn)], dim=1)
            ttnn.deallocate(qkv)
            old = self.X.get(id(dn))
            self.X[id(dn)] = X
            if old is not None and self.trace_id is None and not self._capturing:
                ttnn.deallocate(old)
        outs = []
        if self.xrow:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, T, dn.Nv * 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, _L1)
            bufs, src = self.bufs[id(dn)], self.base[id(dn)]
            # the kernel reads single 32-byte rows of X: from L1 (32-byte-offset reads from DRAM came back wrong)
            if self.xconv and Xl is None:
                Xl = ttnn.to_memory_config(X, _L1)
            if self.chain1:
                fd.chain_all(qkvzab, out, src, bufs, Xl, self._kvm if self.nocommit else None)
            else:
                for i in range(T):
                    fd.chain_call(qkvzab, out, i, src, bufs[i], i == T - 1, Xl)
                    src = bufs[i]
            if Xl is not None:
                ttnn.deallocate(Xl)
            ttnn.deallocate(qkvzab)
            gated = out
        elif self.chain:
            # DF_GDN_CHAIN: slot i runs the fused program in place on its own state buffer, a clone of slot i-1's
            # (slot 0 clones `base` = the committed state): one state copy per slot instead of slice + copy, and the
            # commit is one copy (bufs[a] -> base). The fused program reads/writes whatever _rec() returns.
            bufs, buf = [], self.base[id(dn)]
            try:
                for i in range(T):
                    row = ttnn.slice(qkvzab, (0, i, 0), (1, i + 1, W), memory_config=_L1)
                    nb = ttnn.clone(buf, memory_config=_DRAM)
                    fd._rec = (lambda b=nb: b)
                    outs.append(fd(row, 1))
                    ttnn.deallocate(row)
                    bufs.append(nb)
                    buf = nb
            finally:
                if "_rec" in fd.__dict__:
                    del fd._rec
            self.bufs[id(dn)] = bufs  # kept alive: trace intermediates read by the commit traces
        else:
            stash = self.stash[id(dn)]
            for i in range(T):
                row = ttnn.slice(qkvzab, (0, i, 0), (1, i + 1, W), memory_config=_L1)
                outs.append(fd(row, 1))
                ttnn.deallocate(row)
                if i < T - 1:
                    r0 = dn._slice_along(dn.rec_state, 0, 0, 1)
                    ttnn.copy(r0, stash[i])
                    ttnn.deallocate(r0)
        if not self.xrow:
            ttnn.deallocate(qkvzab)
            gated = ttnn.concat(outs, dim=1, memory_config=_L1)  # [1, T, Nv*128]
            for t in outs:
                ttnn.deallocate(t)
        partial = dn._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        from models.demos.blackhole.qwen36.tt import tp_common as _tpc

        if _tpc.ar_active():  # DF_AR: the residual stays replicated (as plain decode with QWEN36_CCL_AR)
            return _tpc.decode_all_reduce(partial, dn.tt_ccl, "gdn")
        return tt_all_reduce(
            partial, dn.mesh, dn.tt_ccl, cluster_axis=0, dim=3, topology=dn.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _ztail(self, dn):
        if not hasattr(self, "_zt"):
            self._zt = {}
        if id(dn) not in self._zt:  # allocated at the first (eager) verify, i.e. before any trace
            self._zt[id(dn)] = ttnn.from_torch(
                torch.zeros(1, self.T, dn.qkv_dim_tp, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=_DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        return self._zt[id(dn)]

    def _commit_prog(self, dn, a):
        bufa, base, Xp, hist = self.bufs[id(dn)][a], self.base[id(dn)], self.Xp[id(dn)], self.hist[id(dn)]
        key = (a,) + tuple(t.buffer_address() for t in (bufa, base, Xp, hist))
        hit = self._ck.get((id(dn), a))
        if hit is not None and hit[0] == key:
            return hit[1]
        from models.demos.blackhole.qwen36.tt.gdn.fused_decode import KDIR

        nst = dn.Nv * (dn.Dk // 32) * (dn.Dv // 32)  # fp32 state tiles of the layer
        nh = dn.qkv_dim_tp // 32
        ncores = 32
        grid = self.mesh.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(ncores)]
        crs = ttnn.num_cores_to_corerangeset(ncores, grid, True)
        ct = []
        for t in (bufa, base, Xp, hist):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        sper, hper = -(-nst // ncores), -(-nh // ncores)
        for k, c in enumerate(cores):
            s0, h0 = min(nst, k * sper), min(nh, k * hper)
            rt[c.x][c.y] = [bufa.buffer_address(), base.buffer_address(), Xp.buffer_address(), hist.buffer_address(),
                            s0, min(sper, nst - s0), h0, min(hper, nh - h0)]
        kd = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/dfcommit_reader.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=ttnn.ReaderConfigDescriptor())
        kd.defines = list(kd.defines) + [("GDNC_A", str(int(a)))]
        cbs = [
            ttnn.CBDescriptor(total_size=4 * 4096, core_ranges=crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=4096)]),
            ttnn.CBDescriptor(total_size=2 * 2048, core_ranges=crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=2048)]),
        ]
        prog = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=cbs)
        self._ck[(id(dn), a)] = (key, prog)
        return prog

    def _win_prog(self, dn, qkvzab, Xl):
        # r64t: the persistent window is per (depth, layer) - StreamState sets _xkey before the call (unset = legacy)
        hist, Xp = self.hist[id(dn)], self.Xp[getattr(self, "_xkey", id(dn))]
        ts = (hist, qkvzab, Xl, Xp) + ((self._kvm,) if self.nocommit else ())
        key = tuple(t.buffer_address() for t in ts)
        wc = self.__dict__.setdefault("_wk", {})
        hit = wc.get(id(dn))
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
                            h0, min(hper, nh - h0)] + ([self._kvm.buffer_address()] if self.nocommit else [])
        kd = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/dfwin_reader.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=ttnn.ReaderConfigDescriptor())
        kd.defines = list(kd.defines) + [("GDNW_T", str(int(self.T)))] + ([("GDNW_AIDX", "1")] if self.nocommit else [])
        cbs = [ttnn.CBDescriptor(total_size=3 * 2048, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)])]
        prog = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=cbs)
        wc[id(dn)] = (key, prog)
        return prog

    def _commit_fused(self, dn, a):
        if self.commit_k:
            prog = self._commit_prog(dn, a)
            ttnn.generic_op([self.bufs[id(dn)][a], self.base[id(dn)], self.Xp[id(dn)], self.hist[id(dn)]], prog)
            return
        X = self.X[id(dn)]
        if self.chain:
            ttnn.copy(self.bufs[id(dn)][a], self.base[id(dn)])
        if a < self.T - 1 and not self.xconv:
            if not self.chain:
                dn._write_recurrent_state_prefix(ttnn.clone(self.stash[id(dn)][a]), 1)
            B = dn.conv_states[1].shape[1]
            for m in range(1, dn.K):  # conv history rows 1..3 = inputs of slots a-2, a-1, a (row 0 of each tensor)
                src = ttnn.slice(X, (0, a + m, 0), (1, a + m + B, dn.qkv_dim_tp))
                ttnn.copy(src, dn.conv_states[m])
                ttnn.deallocate(src)
        h = ttnn.slice(X, (0, a + 1, 0), (1, a + dn.K, dn.qkv_dim_tp))
        ttnn.copy(h, self.hist[id(dn)])
        ttnn.deallocate(h)

    # ------------------------------------------------------------------ GDN verify
    def _gdn_verify(self, dn, x):
        fd = self._fused(dn)
        if fd is not None:
            return self._gdn_verify_fused(dn, x, fd)
        tw, Nk, Nv, Dk, Dv = dn.tw, dn.Nk, dn.Nv, dn.Dk, dn.Dv
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        T = x.shape[-2]
        D = dn.qkv_dim_tp
        qkv, z, a, b = dn._project_qkvzab(x, T, out_mc=_L1)
        X = ttnn.concat([self.hist[id(dn)], ttnn.to_memory_config(qkv, _DRAM)], dim=1)  # [1, K-1+T, D]
        ttnn.deallocate(qkv)
        old = self.X.get(id(dn))
        self.X[id(dn)] = X
        if old is not None and self.trace_id is None and not self._capturing:
            ttnn.deallocate(old)
        conv = None
        for j in range(dn.K):
            xs = ttnn.slice(X, (0, j, 0), (1, j + T, D), memory_config=_L1)
            if j == 0:
                conv = ttnn.multiply(xs, tw["conv_taps"][0], memory_config=_L1)
            else:
                conv = ttnn.mac(xs, tw["conv_taps"][j], conv)
            ttnn.deallocate(xs)
        conv = ttnn.silu(conv, memory_config=_L1)
        kd = dn.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, T, kd)), (T, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd)), (T, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, T, D)), (T, Nv, Dv))
        ttnn.deallocate(conv)
        rf = Nv // Nk
        q = ttnn.repeat_interleave(q, rf, dim=1)
        k = ttnn.repeat_interleave(k, rf, dim=1)
        q = ttnn.reshape(q, (T, 1, Nv, Dk), memory_config=_L1)
        k = ttnn.reshape(k, (T, 1, Nv, Dk), memory_config=_L1)
        v = ttnn.reshape(v, (T, 1, Nv, Dv), memory_config=_L1)
        beta = ttnn.reshape(ttnn.sigmoid(b, memory_config=_L1), (T, 1, Nv))
        ttnn.deallocate(b)
        g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
        ttnn.deallocate(a)
        g = ttnn.reshape(g, (T, 1, Nv))
        hp = os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"
        h0 = dn._slice_along(dn.rec_state, 0, 0, 1)  # same as plain decode's B=1 init_state
        mode = os.environ.get("DF_GDN_HOIST", "2")
        if mode == "2":
            o, states = self._recurrence_lean(q, k, v, beta, g, h0, dn.scale, dn.mesh, hp, dn.rec_state.dtype)
            outs = None
        elif mode == "1":
            outs, states = self._recurrence_hoisted(q, k, v, beta, g, h0, dn.scale, dn.mesh, hp, dn.rec_state.dtype)
        else:
            outs, states = self._recurrence_calls(q, k, v, beta, g, h0, dn.scale, dn.mesh, hp, dn.rec_state.dtype)
        for t in (q, k, v, beta, g):
            ttnn.deallocate(t)
        stash = ttnn.concat(states, dim=0, memory_config=_DRAM)  # [T, Nv, Dk, Dv]: state after slot i in row i
        for t in states:
            ttnn.deallocate(t)
        if stash.dtype != dn.rec_state.dtype:
            stash = ttnn.typecast(stash, dn.rec_state.dtype)
        ttnn.copy(stash, dn.rec_state)
        ttnn.deallocate(stash)
        if outs is not None:
            o = ttnn.concat(outs, dim=0)
            for t in outs:
                ttnn.deallocate(t)
        # ---- rest exactly as forward_decode
        out_r = ttnn.reshape(o, (T, Nv, Dv))
        out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
        ttnn.deallocate(out_r)
        out_f = ttnn.reshape(out_n, (1, T, dn.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = _silu_mul(out_f, z, _L1)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        partial = dn._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        return tt_all_reduce(
            partial, dn.mesh, dn.tt_ccl, cluster_axis=0, dim=3, topology=dn.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _recurrence_lean(self, q, k, v, beta, g, h, scale, mesh, hp, sdtype):
        """DF_GDN_HOIST=2: the decode recurrence's compute ops in the same order with the same inputs, but every
        token-independent op (typecasts, l2 norms, scale, exp, and all reshapes) runs once over the T rows, so a
        token costs 6 slices + 7 compute ops. Returns (o [T,1,H,V] DRAM, states list [T] of [1,H,K,V] L1)."""
        from models.experimental.gated_attention_gated_deltanet.tt import ttnn_delta_rule_ops as R

        T, H, K, V = q.shape[0], q.shape[2], q.shape[3], v.shape[3]
        if hp:
            q = ttnn.typecast(q, ttnn.float32)
            k = ttnn.typecast(k, ttnn.float32)
            v = ttnn.typecast(v, ttnn.float32)
            beta = ttnn.typecast(beta, ttnn.float32)
            g = ttnn.typecast(g, ttnn.float32)
        q = R.l2_norm_ttnn(q, dim=-1)
        k = R.l2_norm_ttnn(k, dim=-1)
        q = ttnn.multiply(q, scale, memory_config=_L1)
        q_row = ttnn.reshape(q, [T, H, 1, K], memory_config=_L1)
        k_row = ttnn.reshape(k, [T, H, 1, K], memory_config=_L1)
        k_col = ttnn.reshape(ttnn.reshape(k_row, [T, H, K], memory_config=_L1), [T, H, K, 1], memory_config=_L1)
        v4 = ttnn.reshape(v, [T, H, 1, V], memory_config=_L1)
        beta4 = ttnn.reshape(ttnn.reshape(beta, [T, H], memory_config=_L1), [T, H, 1, 1], memory_config=_L1)
        decay = ttnn.exp(ttnn.reshape(g, [T, H], memory_config=_L1), memory_config=_L1)
        decay4 = ttnn.reshape(decay, [T, H, 1, 1], memory_config=_L1)
        if hp and h.dtype != ttnn.float32:
            h = ttnn.typecast(h, ttnn.float32)
        h = ttnn.to_memory_config(h, _L1)
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        cfg_outer = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        try:
            pcfg = R._recurrent_read_query_program_config(mesh, K, V)
        except Exception:
            pcfg = None
        stash_l1 = os.environ.get("DF_STASH_L1", "0") == "1"  # 1: keep the 8 states in L1 (12.6 MB/device)
        outs, states = [], []
        for i in range(T):
            qr = ttnn.slice(q_row, (i, 0, 0, 0), (i + 1, H, 1, K), memory_config=_L1)
            kr = ttnn.slice(k_row, (i, 0, 0, 0), (i + 1, H, 1, K), memory_config=_L1)
            kc = ttnn.slice(k_col, (i, 0, 0, 0), (i + 1, H, K, 1), memory_config=_L1)
            vv = ttnn.slice(v4, (i, 0, 0, 0), (i + 1, H, 1, V), memory_config=_L1)
            bb = ttnn.slice(beta4, (i, 0, 0, 0), (i + 1, H, 1, 1), memory_config=_L1)
            dd = ttnn.slice(decay4, (i, 0, 0, 0), (i + 1, H, 1, 1), memory_config=_L1)
            h = ttnn.multiply(h, dd, memory_config=_L1)
            vr = ttnn.matmul(kr, h, memory_config=_L1, program_config=pcfg, compute_kernel_config=cfg)
            delta = ttnn.subtract(vv, vr, memory_config=_L1)
            outer = ttnn.matmul(kc, delta, memory_config=_L1, compute_kernel_config=cfg_outer, program_config=None)
            outer = ttnn.multiply(outer, bb, memory_config=_L1)
            h = ttnn.add(h, outer, memory_config=_L1)
            o = ttnn.matmul(qr, h, memory_config=_L1, program_config=pcfg, compute_kernel_config=cfg)
            states.append(h if stash_l1 else ttnn.to_memory_config(h, _DRAM))
            outs.append(o)
            for t in (qr, kr, kc, vv, bb, dd, vr, delta, outer):
                ttnn.deallocate(t)
        o = ttnn.concat(outs, dim=0, memory_config=_DRAM)  # [T, H, 1, V]
        for t in outs:
            ttnn.deallocate(t)
        o = ttnn.reshape(o, [T, 1, H, V])
        return o, states

    def _recurrence_calls(self, q, k, v, beta, g, h, scale, mesh, hp, sdtype):
        """T chained calls of the decode recurrence (exactly what plain decode runs per token)."""
        T, Nv, Dk, Dv = q.shape[0], q.shape[2], q.shape[3], v.shape[3]
        outs, states = [], []
        for i in range(T):
            qi = ttnn.slice(q, (i, 0, 0, 0), (i + 1, 1, Nv, Dk), memory_config=_L1)
            ki = ttnn.slice(k, (i, 0, 0, 0), (i + 1, 1, Nv, Dk), memory_config=_L1)
            vi = ttnn.slice(v, (i, 0, 0, 0), (i + 1, 1, Nv, Dv), memory_config=_L1)
            bi = ttnn.slice(beta, (i, 0, 0), (i + 1, 1, Nv), memory_config=_L1)
            gi = ttnn.slice(g, (i, 0, 0), (i + 1, 1, Nv), memory_config=_L1)
            o_i, h_new = gdn_tp.recurrent_gated_delta_rule_decode_ttnn(
                qi, ki, vi, bi, gi, scale=scale, initial_state=h, device=mesh, high_precision=hp
            )
            for t in (qi, ki, vi, bi, gi, h):
                ttnn.deallocate(t)
            st = ttnn.to_memory_config(h_new, _DRAM)
            states.append(st if st.dtype == sdtype else ttnn.typecast(st, sdtype))
            outs.append(ttnn.to_memory_config(o_i, _DRAM))
            ttnn.deallocate(o_i)
            h = h_new
        ttnn.deallocate(h)
        return outs, states

    def _recurrence_hoisted(self, q, k, v, beta, g, h, scale, mesh, hp, sdtype):
        """The same op sequence as recurrent_gated_delta_rule_decode_ttnn, with the per-row (token-independent)
        prologue -- typecasts, l2 norms, the q scale, exp(g) -- run once over all T rows instead of per token.
        Row-wise ops give the same bits for a row whatever the batch, so this should stay bit-identical
        (checked by t_exact.py: DF_GDN_HOIST=0 is the per-call fallback)."""
        from models.experimental.gated_attention_gated_deltanet.tt import ttnn_delta_rule_ops as R

        T, H, K, V = q.shape[0], q.shape[2], q.shape[3], v.shape[3]
        if hp:
            q = ttnn.typecast(q, ttnn.float32)
            k = ttnn.typecast(k, ttnn.float32)
            v = ttnn.typecast(v, ttnn.float32)
            beta = ttnn.typecast(beta, ttnn.float32)
            g = ttnn.typecast(g, ttnn.float32)
        q = R.l2_norm_ttnn(q, dim=-1)
        k = R.l2_norm_ttnn(k, dim=-1)
        q = ttnn.multiply(q, scale, memory_config=_L1)
        q_row = ttnn.reshape(q, [T, H, 1, K], memory_config=_L1)
        k_row = ttnn.reshape(k, [T, H, 1, K], memory_config=_L1)
        v_t = ttnn.reshape(v, [T, H, V], memory_config=_L1)
        beta_t = ttnn.reshape(beta, [T, H], memory_config=_L1)
        g_t = ttnn.reshape(g, [T, H], memory_config=_L1)
        decay_t = ttnn.exp(g_t, memory_config=_L1)
        if hp and h.dtype != ttnn.float32:
            h = ttnn.typecast(h, ttnn.float32)
        h = ttnn.to_memory_config(h, _L1)
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        try:
            pcfg = R._recurrent_read_query_program_config(mesh, K, V)
        except Exception:
            pcfg = None
        outs, states = [], []
        for i in range(T):
            qr = ttnn.slice(q_row, (i, 0, 0, 0), (i + 1, H, 1, K), memory_config=_L1)
            kr = ttnn.slice(k_row, (i, 0, 0, 0), (i + 1, H, 1, K), memory_config=_L1)
            vt = ttnn.slice(v_t, (i, 0, 0), (i + 1, H, V), memory_config=_L1)
            bt = ttnn.slice(beta_t, (i, 0), (i + 1, H), memory_config=_L1)
            dt = ttnn.slice(decay_t, (i, 0), (i + 1, H), memory_config=_L1)
            dec = ttnn.reshape(dt, [1, H, 1, 1], memory_config=_L1)
            h = ttnn.multiply(h, dec, memory_config=_L1)
            vr = ttnn.matmul(kr, h, memory_config=_L1, program_config=pcfg, compute_kernel_config=cfg)
            vr = ttnn.reshape(vr, [1, H, V], memory_config=_L1)
            delta = ttnn.subtract(vt, vr, memory_config=_L1)
            kt = ttnn.reshape(kr, [1, H, K], memory_config=_L1)
            h = R.fused_decay_and_write_ttnn(h=h, k_t=kt, delta=delta, decay_t=dt, beta_t=bt, device=mesh,
                                             apply_decay=False)
            o = ttnn.matmul(qr, h, memory_config=_L1, program_config=pcfg, compute_kernel_config=cfg)
            o = ttnn.reshape(o, [1, 1, H, V], memory_config=_L1)
            st = ttnn.to_memory_config(h, _DRAM)
            states.append(st if st.dtype == sdtype else ttnn.typecast(st, sdtype))
            outs.append(ttnn.to_memory_config(o, _DRAM))
            for t in (qr, kr, vt, bt, dt, vr, delta, o):
                ttnn.deallocate(t)
        ttnn.deallocate(h)
        for t in (q_row, k_row, v_t, beta_t, g_t, decay_t):
            ttnn.deallocate(t)
        return outs, states

    # ------------------------------------------------------------------ attention row-wise wrappers
    @contextlib.contextmanager
    def _patched(self):
        orig_upd = ttnn.experimental.paged_update_cache
        orig_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode
        sdpa_rows = self.sdpa_rows
        cache = {}  # per-forward row slices of cur_pos / page_table (shared by all 16 attention layers)

        def row_of(t, i):
            key = (id(t), i)
            if key not in cache:
                if t is self._kvm:  # DF_KV_SKIP: [T * T] position masks -> row i = [T] (-1 except at i)
                    cache[key] = ttnn.slice(t, (i * self.T,), ((i + 1) * self.T,))
                elif len(t.shape) == 1:
                    cache[key] = ttnn.slice(t, (i,), (i + 1,))
                else:
                    cache[key] = ttnn.slice(t, (i, 0), (i + 1, t.shape[-1]))
            return cache[key]

        fused = os.environ.get("DF_KV_FUSED", "0") == "1"
        orig_fused = ttnn.experimental.paged_fused_update_cache
        pend = []  # fused mode: the K call parks here, the V call writes both rows per op

        def rows(x_sh):
            xi = ttnn.sharded_to_interleaved(x_sh, _L1) if x_sh.is_sharded() else x_sh
            return xi, [ttnn.slice(xi, (0, i, 0, 0), (1, i + 1, xi.shape[2], xi.shape[-1]), memory_config=_L1)
                        for i in range(x_sh.shape[1])]

        kv_skip = self.kv_skip
        kvf_skip = os.environ.get("DF_KVF_SKIP", "0") == "1"

        def upd(kv_cache, x_sh, update_idxs_tensor=None, page_table=None, **kw):
            B = x_sh.shape[1]
            if page_table is None or B == 1:
                return orig_upd(kv_cache, x_sh, update_idxs_tensor=update_idxs_tensor, page_table=page_table, **kw)
            if kv_skip and kvf_skip:
                # ep2 t3b DF_KVF_SKIP=1: K and V of a row in ONE fused update program (the K call parks here)
                if not pend:
                    pend.append((kv_cache, x_sh))
                    return
                k_cache, k_sh = pend.pop()
                # the fused op wants K and V on disjoint cores: V moves one core row down (a copy)
                sp = x_sh.memory_config().shard_spec
                key = ("vcfg", tuple(sp.shape))
                if key not in cache:
                    rs = list(sp.grid.ranges())
                    lo, hi = rs[0].start, rs[-1].end
                    cr = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(lo.x, hi.y + 1),
                                                           ttnn.CoreCoord(hi.x, hi.y + 1 + (hi.y - lo.y)))})
                    cache[key] = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                                   ttnn.ShardSpec(cr, tuple(sp.shape), sp.orientation))
                v2 = ttnn.to_memory_config(x_sh, cache[key])
                for i in range(B):
                    orig_fused(k_cache, k_sh, kv_cache, v2, update_idxs_tensor=row_of(self._kvm, i),
                               page_table=page_table)
                ttnn.deallocate(v2)
                return
            if kv_skip:
                # ep2 t3 DF_KV_SKIP=1: the batched B-row update program (as multi-user decode runs it), once per row
                # with the positions of all other rows at -1 (the kernel skips those rows): rows that share a KV tile
                # are written one program after the other, no row slices or reshards
                for i in range(B):
                    orig_upd(kv_cache, x_sh, update_idxs_tensor=row_of(self._kvm, i), page_table=page_table, **kw)
                return
            hd = x_sh.shape[-1]
            if fused:
                if not pend:
                    pend.append((kv_cache, x_sh))
                    return
                k_cache, k_sh = pend.pop()
                _, krows = rows(k_sh)
                _, vrows = rows(x_sh)
                ck, cv = _rows_kv_cfg(hd), _rows_kv_cfg(hd, core=(1, 0))
                for i in range(B):
                    a = ttnn.to_memory_config(krows[i], ck)
                    b = ttnn.to_memory_config(vrows[i], cv)
                    orig_fused(k_cache, a, kv_cache, b, update_idxs_tensor=row_of(update_idxs_tensor, i),
                               page_table=row_of(page_table, i))
                    ttnn.deallocate(a)
                    ttnn.deallocate(b)
                return
            cfg = _rows_kv_cfg(hd)
            xi, rs = rows(x_sh)
            for i in range(B):
                row_sh = ttnn.to_memory_config(rs[i], cfg)
                ttnn.deallocate(rs[i])
                orig_upd(kv_cache, row_sh, update_idxs_tensor=row_of(update_idxs_tensor, i),
                         page_table=row_of(page_table, i), **kw)
                ttnn.deallocate(row_sh)
            if xi is not x_sh:
                ttnn.deallocate(xi)

        def sdpa(q, keys, values, page_table_tensor=None, cur_pos_tensor=None, **kw):
            B = q.shape[1]
            if not sdpa_rows or B == 1:
                return orig_sdpa(q, keys, values, page_table_tensor=page_table_tensor, cur_pos_tensor=cur_pos_tensor, **kw)
            outs = []
            for i in range(B):
                qi = ttnn.slice(q, (0, i, 0, 0), (1, i + 1, q.shape[2], q.shape[3]))
                o = orig_sdpa(qi, keys, values, page_table_tensor=row_of(page_table_tensor, i),
                              cur_pos_tensor=row_of(cur_pos_tensor, i), **kw)
                outs.append(ttnn.to_memory_config(o, _DRAM))
                for t in (o, qi):
                    ttnn.deallocate(t)
            out = ttnn.concat(outs, dim=1, memory_config=kw.get("memory_config", _L1))
            for t in outs:
                ttnn.deallocate(t)
            return out

        for dn in self.gdn:
            dn.forward_decode = functools.partial(self._gdn_verify, dn)
        ttnn.experimental.paged_update_cache = upd
        ttnn.transformer.paged_scaled_dot_product_attention_decode = sdpa
        try:
            yield
            assert not pend, "fused KV write: a K row write had no V partner"
        finally:
            ttnn.experimental.paged_update_cache = orig_upd
            ttnn.transformer.paged_scaled_dot_product_attention_decode = orig_sdpa
            for dn in self.gdn:
                del dn.forward_decode

    # ------------------------------------------------------------------ the forward
    def _forward(self, tokens, cur_pos, rope, page_table):
        from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope

        model = self.model
        T = self.T
        cos, sin = unpack_rope(rope)
        x = model.embd(tokens)
        if self.nd > 1 or getattr(model, "tp", False):  # df1c TP1: the TP decode input layout
            x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[-1]))
        taps = []
        pf = getattr(model, "_pf", None)  # ep2 QWEN36_PF: the verify pass streams the decode weights too
        # ep2 t3 DF_AR=1: the plain decode's QWEN36_CCL_AR path (embedding gathered once inside the PF frame, residual
        # replicated through all layers, one all-reduce per layer output instead of the fractured AG / RS pairs)
        from models.demos.blackhole.qwen36.tt import tp_common as _tpc

        _ar = getattr(model, "_ccl_ar", None) is not None and os.environ.get("DF_AR", "1") == "1"
        # ep2 t3 / t3b: with QWEN36_PF_LMHEAD=1 the frame also streams the LM head pieces (as plain decode does)
        pf_lm = pf is not None and bool(getattr(pf, "lm_entries", None)) and self.shard_argmax
        rm = None
        with self._patched(), (pf.frame() if pf is not None else contextlib.nullcontext()):
            if _ar:
                x = model._ar_gather_embedding(x)
            with _tpc.ar_decode(_ar):
                for li, layer in enumerate(model.layers):
                    if layer.is_full_attention:
                        x = layer.forward(x, cos, sin, position_tensor=cur_pos, page_table=page_table, mode="decode")
                    else:
                        x = layer.forward(x, mode="decode")
                    if li in self.tap_pos:
                        # (AR: the replicated residual may live in a cycled persistent buffer: keep a copy)
                        taps.append(ttnn.to_memory_config(x, _DRAM) if _ar else x)
            if pf_lm:
                xf = model._final_decode_any(x, _ar) if _ar else model._final_norm_decode(x)
                rm = pf.lm_head(xf, model.lm_head_weight)
                ttnn.deallocate(xf)
        if not pf_lm:
            xf = model._final_decode_any(x, _ar) if _ar else model._final_norm_decode(x)
        if self.shard_argmax and pf_lm:
            logits = rm
            idx = ttnn.argmax(rm, dim=-1)
            gi = ttnn.reshape(idx, list(idx.shape) + [1]) if len(idx.shape) < len(rm.shape) else idx
            ids = (idx, ttnn.gather(rm, -1, gi))
        elif self.shard_argmax:
            # no logits all-gather: argmax + max per vocab shard, the host picks the shard (ties -> lower index)
            nc = int(os.environ.get("DF_LM_1D", "0"))
            if nc > 0:
                # ep2 t3b DF_LM_1D=<cores>: explicit 1D mcast decode program config for the (not prefetched) LM head
                if getattr(self, "_lm_pc", None) is None:
                    from models.demos.blackhole.qwen36.tt import tp_common as tpc

                    w = model.lm_head_weight
                    self._lm_pc = tpc.create_matmul_1d_decode_progcfg(
                        ttnn.TILE_SIZE, int(w.shape[-2]), int(w.shape[-1]), num_cores=nc,
                        grid_w=self.mesh.compute_with_storage_grid_size().x)
                logits = ttnn.linear(xf, model.lm_head_weight, program_config=self._lm_pc)
            else:
                logits = ttnn.linear(xf, model.lm_head_weight)
            ttnn.deallocate(xf)
            rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)  # kept: host logits for sampling penalties
            idx = ttnn.argmax(rm, dim=-1)
            if os.environ.get("DF_MAX_GATHER", "1") == "1":
                # ep2 devloop: the shard max as the logit AT the argmax index (ttnn.gather) instead of ttnn.max over
                # the whole vocab shard: the max was one 1.57 ms program per verify step (profile 2026-11-03 03:15)
                gi = ttnn.reshape(idx, list(idx.shape) + [1]) if len(idx.shape) < len(rm.shape) else idx
                mx = ttnn.gather(rm, -1, gi)
            else:
                mx = ttnn.max(logits, dim=-1, keepdim=True)
            ids = (idx, mx)
        else:
            logits = model._lm_head(xf)
            ttnn.deallocate(xf)
            rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
            if rm.shape[-1] > model.vocab_size:
                rm2 = ttnn.slice(rm, (0, 0, 0, 0), (1, 1, rm.shape[-2], model.vocab_size))
                ttnn.deallocate(rm)
                rm = rm2
            ids = ttnn.argmax(rm, dim=-1)
        full = []
        for t in taps:
            if self.nd > 1 and t.shape[-1] < model.args.dim:  # fractured residual -> full width
                g = tt_all_gather(t, self.mesh, model.tt_ccl, cluster_axis=None, dim=3, topology=ttnn.Topology.Linear)
            else:
                g = t
            full.append(g)
        tap_cat = ttnn.concat(full, dim=3) if full else None  # [1,1,T,25600]; None = no tap layer (r64u plain)
        for t in full:
            ttnn.deallocate(t)
        return ids, logits, tap_cat, rm

    def _stage(self, tokens, start):
        T = self.T
        if tokens is None:  # ep2 t3b DF_CHAIN: the tokens are already on the card (written by the drafter trace)
            assert self.inputs is not None
            pos = torch.arange(start, start + T, dtype=torch.int32)
            if self._pt_host is None:
                self._pt_host = ttnn.from_torch(self.page_table_row.expand(T, -1).contiguous(), dtype=ttnn.int32,
                                                layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                            self.inputs[1])
            ttnn.copy_host_to_device_tensor(self.model.decode_rope_host(pos), self.inputs[2])
            if self.kv_skip:
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(self._kvm_host(pos), dtype=ttnn.int32,
                                                                layout=ttnn.ROW_MAJOR_LAYOUT), self.inputs[4])
            return
        tok = torch.tensor(tokens, dtype=torch.int32).reshape(T, 1)
        pos = torch.arange(start, start + T, dtype=torch.int32)
        pt = self.page_table_row.expand(T, -1).contiguous()
        host = self.model.prepare_decode_inputs_host(tok, pos, page_table=pt)
        if self.kv_skip:
            host = tuple(host) + (ttnn.from_torch(self._kvm_host(pos), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),)
        if self.inputs is None:
            self.inputs = copy_host_to_device(host, mesh_device=self.mesh)
        else:
            copy_host_to_device(host_tensors=host, device_tensors=self.inputs)
        self._kvm = self.inputs[4] if self.kv_skip else None

    def _kvm_host(self, pos):
        """DF_KV_SKIP position masks [T*T] (row i = -1 except pos[i] at i); with DF_NOCOMMIT 16 more entries, the first
        = the previous accepted slot a (>= T: none, base / hist hold the state)."""
        T = self.T
        m = torch.full((T, T), -1, dtype=torch.int32)
        m[torch.arange(T), torch.arange(T)] = pos
        m = m.reshape(-1)
        if self.nocommit:
            tail = torch.zeros(16, dtype=torch.int32)
            tail[0] = self.pending_a if self.pending_a is not None else T + 247
            m = torch.cat([m, tail])
        return m

    _capturing = False

    def capture_trace(self, tokens, start):
        """Capture the verify forward (all its programs must have run once eagerly before)."""
        self._stage(tokens, start)
        from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

        _wp.decode_all()  # ep2 QWEN36_PF: a decode capture: the GCB is recreated before it (begin_trace_capture)
        self._capturing = True
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.out = self._forward(*self.inputs[:4])
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self._capturing = False
        ttnn.synchronize_device(self.mesh)
        self.trace_id = tid
        self.log(f"[dflash] verify trace captured (T={self.T}, sdpa_rows={self.sdpa_rows})")

    def release_trace(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None

    def run_chained(self, start, pre=None):
        """ep2 t3b DF_CHAIN: verify the block the drafter trace wrote into the token input (no host round trip
        between draft and verify). Returns (block [T] host list, the target's greedy ids [T]). pre(): host work run
        after the replay is queued, before the read back (DF_PRESTAGE)."""
        assert self.trace_id is not None
        self._pre = pre
        try:
            post = self.run(None, start)
        finally:
            self._pre = None
        blk = dev_rows(self.inputs[0], self.mesh)[0].reshape(-1)[: self.T].to(torch.int64) & 0xFFFFFFFF
        return [int(v) for v in blk.tolist()], post

    def run(self, tokens, start):
        """Verify `tokens` (T ids; tokens[0] = the last committed token) at positions start..start+T-1.
        Returns the target's greedy ids [T] (host list); taps / logits stay on device in self.out."""
        t0 = time.perf_counter()
        from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

        _wp.decode_all()  # ep2 QWEN36_PF: captures from here on are decode captures
        if self.trace_id is not None and _wp.recreate_pending() and os.environ.get("DF_NO_RECAP", "0") == "1":
            # ep2 t3b DF_NO_RECAP=1 (experiment): recreate the GCB (same L1 address, checked by the placeholder) and
            # replay the verify trace captured before the prefill instead of re-capturing it
            _tq = time.perf_counter()
            _wp.recreated()
            self.timing["recreate_ms"] = round(1000 * (time.perf_counter() - _tq), 1)
            self.recaps_skipped = getattr(self, "recaps_skipped", 0) + 1
        if self.trace_id is not None and _wp.recreate_pending():
            # the GCB was freed for a prefill: a trace captured with the old one reads wrong data -> re-capture
            # ep2 devloop fix (DF_PF_RECOMMIT=1, default): the commit traces read the verify trace's intermediates
            # (self.X conv windows, self.bufs chained states) at the addresses baked when they were captured. A
            # re-captured verify trace allocates NEW X / bufs, so the old commit traces committed stale states (from the
            # warm-up verify): drafts never accepted (acceptance 1.0), output wrong from token ~2. Re-capture the commit
            # traces too, and free the old intermediates (they leaked before: ~8 state copies per GDN layer per request).
            recommit = os.environ.get("DF_PF_RECOMMIT", "1") == "1" and bool(self.commit_tids) and not self.xpersist
            old_x, old_bufs = dict(self.X), {k: list(v) for k, v in self.bufs.items()}
            if recommit:
                for tid in self.commit_tids.values():
                    ttnn.release_trace(self.mesh, tid)
                self.commit_tids = None
            self.release_trace()
            _wp.recreated()
            self.capture_trace(tokens, start)
            if recommit:
                self.capture_commit_traces()
                for k, t in old_x.items():
                    if self.X.get(k) is not t:
                        ttnn.deallocate(t)
                for k, bl in old_bufs.items():
                    keep = {id(b) for b in self.bufs.get(k, [])}
                    for b in bl:
                        if id(b) not in keep:
                            ttnn.deallocate(b)
        _ts0 = time.perf_counter()
        self._stage(tokens, start)
        _ts1 = time.perf_counter()
        if self.trace_id is not None:
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
            _ts2 = time.perf_counter()
            st = self.__dict__.setdefault("step_t", [0, 0.0, 0.0, 0.0])
            st[0] += 1; st[1] += _ts1 - _ts0; st[2] += _ts2 - _ts1
            if getattr(self, "_pre", None) is not None:
                self._pre()
        else:
            if self.out is not None:
                for t in self.out:
                    for u in (t if isinstance(t, tuple) else (t,)):
                        (ttnn.deallocate(u) if u is not None else None)  # df1c
            self.out = self._forward(*self.inputs[:4])
        ids_dev = self.out[0]
        if self.shard_argmax:
            idx_t, mx_t = ids_dev
            per = []
            ii_all, vv_all = dev_rows(idx_t, self.mesh), dev_rows(mx_t, self.mesh)
            for d in range(self.nd):
                ii = ii_all[d].reshape(-1)[: self.T]
                vv = vv_all[d].float().reshape(-1)[: self.T]
                per.append((ii.to(torch.int64) & 0xFFFFFFFF, vv))
            shard = self.model.vocab_size // self.nd  # vocab-sharded LM head: device d holds [d*shard, (d+1)*shard)
            ids = per[0][0].clone()
            best = per[0][1].clone()
            for d in range(1, self.nd):
                better = per[d][1] > best  # strict: a tie keeps the lower shard (first max, like host argmax)
                ids = torch.where(better, per[d][0] + d * shard, ids)
                best = torch.where(better, per[d][1], best)
        elif self.nd > 1:
            ids = dev_rows(ids_dev, self.mesh)[0]
        else:
            ids = ttnn.to_torch(ids_dev)
        self.timing["verify_ms"] = 1000 * (time.perf_counter() - t0)
        return [int(v) for v in ids.reshape(-1)[: self.T].tolist()]

    def host_logits(self):
        """All T rows' logits on the host [T, vocab] fp32 (row-major output: no tile padding crosses PCIe)."""
        rm = self.out[3]
        if self.nd > 1 and self.shard_argmax:
            parts = [t.reshape(-1, t.shape[-1])[: self.T] for t in dev_rows(rm, self.mesh)]
            h = torch.cat(parts, dim=-1)
        elif self.nd > 1:
            h = dev_rows(rm, self.mesh)[0].reshape(-1, rm.shape[-1])[: self.T]
        else:
            h = ttnn.to_torch(rm).reshape(-1, rm.shape[-1])[: self.T]
        return h[:, : self.model.vocab_size].float()

    def row_logits(self, i):
        """Host logits of verify row i (a whole-tensor read; no device slice program)."""
        return self.host_logits()[i]

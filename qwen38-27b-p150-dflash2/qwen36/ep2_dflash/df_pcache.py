"""ep2 dfpc DF_PREFIX_CACHE=1: prefix caching for the multi-stream DFlash2 server.

Off by default (DF_PREFIX_CACHE unset: this module is never imported and no code path changes).

What a resumed admission needs, and where it comes from:
- GDN state at a 2048 boundary b: the pcx snapshot store in the model (QWEN36_PCX_STORE=1, set by df_server before the
  model import; saves at the hit boundary and the last two trace boundaries, QWEN36_PCX_SAVE_PREV=1).
- KV pages [0, b) of the target slot holding the FULL-CHUNK prefill bytes of ids[0:b): this module's per-slot index
  (`valid`), kept in place when the slot is reused by the same prefix (affinity), else copied byte for byte from the
  slot that holds them (one fixed device program, kernels/kvcopy_reader.cpp; parameters in a device int32 tensor, so
  nothing compiles or allocates at request time).
- Drafter taps: only from chunks that run, so b <= n - (PF_ROWS - 7) (the pc1 rule).

Placement keeps df_ms's lowest-free-slot rule (the spec verifiers' row layout is by slot index). A cached prefix in the
target slot that is held nowhere else is first moved ("evacuated") to the least valuable other free slot.
"""
import os
import pathlib
import sys
import time

import torch

import ttnn

CHUNK = 2048
LOGV = os.environ.get("DF_PREFIX_CACHE_VERBOSE", "1") == "1"
# DF_PREFIX_VERIFY=1: after every page copy, an on-card byte compare of the copied pages against the source pages
# (kernels/kvcmp_reader.cpp; mismatching 32-bit words counted on both cards). Check runs only: it doubles the reads.
VERIFY = os.environ.get("DF_PREFIX_VERIFY", "0") == "1"


def _reg():
    from models.demos.blackhole.qwen36.tt import pcx_registry

    return pcx_registry


class PrefixCache:
    def __init__(self, srv):
        import df_drafter

        S = srv.S
        self.srv, self.S, self.model = srv, S, S.model
        self.NS = len(S.page_rows)
        self.nbs = int(S.page_rows[0].shape[1])
        self.block = int(S.block)
        assert CHUNK % self.block == 0
        self.bpc = CHUNK // self.block  # KV blocks per 2048 chunk
        self.tap_need = df_drafter.PF_ROWS - 7
        self.valid = [(0, {}) for _ in range(self.NS)]  # slot -> (K, {k: key}) : pages [0, 2048 K) exact
        self.last_used = [0.0] * self.NS
        self.stats = dict(admits=0, hits=0, misses=0, tokens_skipped=0, copies=0, copy_blocks=0, evacuations=0,
                          copy_s=0.0, drops=0, drafter_capped=0, evictions=0, evicted_chunks=0, verified=0,
                          verify_bad_words=0, store_evictions=0)
        self.copy_log = []  # (chunks, ms) per copy
        self._build_copy()

    # ------------------------------------------------------------------ device page copy
    def _build_copy(self):
        S, mesh = self.S, self.S.mesh
        kv = []
        for pair in S.kv:
            for t in pair:
                kv.append(t)
        t0 = kv[0]
        for t in kv:
            assert t.layout == ttnn.TILE_LAYOUT and tuple(t.shape) == tuple(t0.shape) and t.dtype == t0.dtype, \
                (t.shape, t.dtype, t.layout)
            assert t.memory_config().buffer_type == ttnn.BufferType.DRAM and not t.is_sharded()
        nb, nkv, blk, hd = [int(x) for x in t0.shape]
        assert blk == self.block, (blk, self.block)
        self.tpb = nkv * (blk // 32) * (hd // 32)  # tiles per block
        pb = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.float32: 4096}[t0.dtype]
        self.kv = kv
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.p_dev = ttnn.from_torch(torch.zeros(16, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                     device=mesh, mesh_mapper=rep)
        grid = mesh.compute_with_storage_grid_size()
        ncores = min(int(os.environ.get("DF_PREFIX_COPY_CORES", "64")), grid.x * grid.y)
        cores = [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(ncores)]
        crs = ttnn.num_cores_to_corerangeset(ncores, grid, True)
        rt = ttnn.RuntimeArgs()
        for k, c in enumerate(cores):
            rt[c.x][c.y] = [k, ncores, pb, self.p_dev.buffer_address(), len(kv)] + [t.buffer_address() for t in kv]
        ct = list(ttnn.TensorAccessorArgs(t0).get_compile_time_args()) + \
            list(ttnn.TensorAccessorArgs(self.p_dev).get_compile_time_args())
        kdir = pathlib.Path(sys.modules[type(self.model).__module__].__file__).resolve().parent.parent / "kernels"
        batch = 8
        kd = ttnn.KernelDescriptor(kernel_source=str(kdir / "kvcopy_reader.cpp"),
                                   source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
                                   compile_time_args=ct, runtime_args=rt, config=ttnn.ReaderConfigDescriptor())
        kd.defines = list(kd.defines) + [("KVC_BATCH", str(batch))]
        cbd = ttnn.CBDescriptor(total_size=batch * pb + 64, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=batch * pb + 64)])
        self.prog = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cbd])
        # the byte-compare program (same P; output: one 64-byte row per core)
        self.ncores = ncores
        self.o_dev = ttnn.from_torch(torch.zeros(16 * ncores, dtype=torch.int32), dtype=ttnn.int32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        rt2 = ttnn.RuntimeArgs()
        for k, c in enumerate(cores):
            rt2[c.x][c.y] = [k, ncores, pb, self.p_dev.buffer_address(), self.o_dev.buffer_address(), len(kv)] + \
                [t.buffer_address() for t in kv]
        ct2 = ct + list(ttnn.TensorAccessorArgs(self.o_dev).get_compile_time_args())
        kd2 = ttnn.KernelDescriptor(kernel_source=str(kdir / "kvcmp_reader.cpp"),
                                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
                                    compile_time_args=ct2, runtime_args=rt2, config=ttnn.ReaderConfigDescriptor())
        cb2 = ttnn.CBDescriptor(total_size=2 * pb + 128, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2 * pb + 128)])
        self.prog_cmp = ttnn.ProgramDescriptor(kernels=[kd2], semaphores=[], cbs=[cb2])
        self.pb = pb
        self.dev_bytes = 64 + 64 * ncores  # P + O, per card (the programs' only buffers, DRAM)
        self.compiled = False

    def compile(self):
        """Both programs compiled by empty runs. Called right after the first warm-up prefill (df_server.prefill):
        before any trace, and with the prefetcher's L1 (its GCB / pre-start placeholder) already released by that
        prefill, so the programs' L1 CBs never meet it and the bring-up L1 layout is untouched."""
        self._copy_blocks(0, 0, 0)
        self._cmp_blocks(0, 0, 0)
        ttnn.synchronize_device(self.S.mesh)
        self.compiled = True
        self.t_compiled = time.time()
        self.srv.log(f"[pcache] copy + compare programs compiled (first warm-up prefill, before any trace capture): "
                     f"{len(self.kv)} KV tensors {list(self.kv[0].shape)} {self.kv[0].dtype}, {self.tpb} tiles "
                     f"({self.tpb * self.pb} B) per block, {self.ncores} cores; {self.NS} slots x {self.nbs} blocks")

    def _copy_blocks(self, src_blk, dst_blk, nblk):
        from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

        _wp.release_all()  # as every prefill: with QWEN36_PF the prefetcher's global CB holds L1 between steps
        p = torch.zeros(16, dtype=torch.int32)
        p[0], p[1], p[2] = src_blk * self.tpb, dst_blk * self.tpb, nblk * self.tpb
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(p, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.S.mesh)),
                                        self.p_dev)
        ttnn.generic_op(self.kv + [self.p_dev], self.prog)

    def _cmp_blocks(self, src_blk, dst_blk, nblk):
        """-> (mismatching 32-bit words, pages compared), summed over both cards."""
        from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

        _wp.release_all()
        p = torch.zeros(16, dtype=torch.int32)
        p[0], p[1], p[2] = src_blk * self.tpb, dst_blk * self.tpb, nblk * self.tpb
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(p, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.S.mesh)),
                                        self.p_dev)
        ttnn.generic_op(self.kv + [self.p_dev, self.o_dev], self.prog_cmp)
        o = ttnn.to_torch(self.o_dev, mesh_composer=ttnn.ConcatMeshToTensor(self.S.mesh, dim=0)).reshape(-1, 16)
        return int(o[:, 0].sum()), int(o[:, 1].sum())

    def compare_prefix(self, src, dst, k):
        """Byte compare KV pages [0, 2048 k) of slot src vs slot dst, all KV tensors, both cards."""
        nblk = k * self.bpc
        bad, pages = self._cmp_blocks(int(self.S.page_rows[src][0, 0]), int(self.S.page_rows[dst][0, 0]), nblk)
        exp = nblk * self.tpb * len(self.kv) * self.S.mesh.get_num_devices()
        return dict(bad_words=bad, pages=pages, expected_pages=exp, bytes=pages * self.pb)

    def copy_prefix(self, src, dst, k):
        """KV pages [0, 2048 k) of slot src -> slot dst (same in-slot block offsets). Queued on cq 0."""
        if k <= 0 or src == dst:
            return
        t0 = time.perf_counter()
        nblk = k * self.bpc
        self._copy_blocks(int(self.S.page_rows[src][0, 0]), int(self.S.page_rows[dst][0, 0]), nblk)
        if os.environ.get("DF_PREFIX_COPY_SYNC", "1") == "1":
            ttnn.synchronize_device(self.S.mesh)  # (for the copy-cost number; the queue orders it anyway)
        self.stats["copies"] += 1
        self.stats["copy_blocks"] += nblk
        ms = 1000 * (time.perf_counter() - t0)
        self.stats["copy_s"] += ms / 1000
        self.copy_log = (self.copy_log + [(k, round(ms, 2))])[-200:]
        mib = nblk * self.tpb * self.pb * len(self.kv) / 2**20
        msg = f"[pcache] COPY slot {src} -> {dst}: {k} chunks ({nblk} blocks, {mib:.0f} MiB/card) {ms:.2f} ms"
        if VERIFY:
            r = self.compare_prefix(src, dst, k)
            self.stats["verified"] += 1
            self.stats["verify_bad_words"] += r["bad_words"]
            ok = r["bad_words"] == 0 and r["pages"] == r["expected_pages"]
            msg += f"; VERIFY {'OK' if ok else 'MISMATCH'} {r}"
            assert ok, msg
        self.srv.log(msg)
        K, keys = self.valid[src]
        self.valid[dst] = (k, {j: keys[j] for j in range(1, k + 1)})

    # ------------------------------------------------------------------ index
    def keys(self, ids, nf):
        return _reg().boundary_keys(torch.tensor(ids[: nf * CHUNK], dtype=torch.int32).numpy(), nf, CHUNK) if nf else {}

    def kv_k(self, s, keys):
        K, ks = self.valid[s]
        for k in range(min(K, len(keys)), 0, -1):
            if ks.get(k) == keys[k]:
                return k
        return 0

    def _held(self, key, k, skip=None):
        return any(s != skip and self.valid[s][0] >= k and self.valid[s][1].get(k) == key for s in range(self.NS))

    def _set_valid(self, s, K, keys, future=None):
        """Slot s now exact for [0, 2048 K) with these keys; snapshots of the keys it lost and nobody holds go
        (future: the keys the coming prefill of s recreates byte for byte - not lost)."""
        oldK, old = self.valid[s]
        self.valid[s] = (K, {j: keys[j] for j in range(1, K + 1)})
        store = getattr(self.model, "_pcx_index", None)
        fut = future or {}
        for j, key in old.items():
            if j <= K and keys.get(j) == key:
                continue
            if fut.get(j) == key:
                continue
            if store is not None and key in store and not self._held(key, j):
                if self.model._pcx_drop(key):
                    self.stats["drops"] += 1

    def plan(self, ids, free, live):
        """-> (slot, plan) for the head request. Slot = the lowest free slot (df_ms's rule)."""
        n = len(ids)
        nf, tail = n // CHUNK, n % CHUNK
        limit = nf if tail else nf - 1
        kd = (n - min(n, self.tap_need)) // CHUNK
        keys = self.keys(ids, nf)
        kvk = [self.kv_k(s, keys) for s in range(self.NS)]
        hit = min(max(kvk), kd)  # the prefix whose KV some slot holds (the pcx "vLLM hit")
        kmax = max(0, min(limit, hit))
        store = getattr(self.model, "_pcx_index", None) or {}
        k0 = 0
        for k in range(kmax, 0, -1):
            if keys[k] in store:
                k0 = k
                break
        if max(kvk) > kd and k0 < min(limit, max(kvk)):
            self.stats["drafter_capped"] += 1
        target = min(free)
        src = None
        if k0:
            if kvk[target] >= k0:
                src = target
            else:  # a free holder first (nothing to disturb), else a live one (its pages below its prompt are final)
                cands = [s for s in range(self.NS) if kvk[s] >= k0]
                src = min(cands, key=lambda s: (s not in free, s))
        return target, dict(n=n, nf=nf, keys=keys, k0=k0, src=src, hit=hit, kvk=kvk, free=list(free))

    def before_prefill(self, s, p):
        """Evacuate slot s's own cache if valuable and held nowhere else, bring in the prefix, truncate. Returns the
        cached_len to hand the model (its pcx plan then picks k0 and the saves)."""
        self.stats["admits"] += 1
        self.model._pcx_last_k0 = 0
        K, ks = self.valid[s]
        k0, src, fut = p["k0"], p["src"], p["keys"]
        keep = k0 if src == s else 0  # what of slot s's current content survives this admission
        # boundaries of slot s that this admission destroys (not kept, not recreated identically, held nowhere else)
        lost = [j for j in range(keep + 1, K + 1) if fut.get(j) != ks[j] and not self._held(ks[j], j, skip=s)]
        evacuated = False
        if lost:
            # the victim: another free slot, not the prefix's source; empty first, else the least recently used
            vic = [v for v in p["free"] if v != s and v != src]
            if vic:
                v = min(vic, key=lambda v: (self.valid[v][0] > 0, self.last_used[v], v))
                if self.valid[v][0] == 0 or self.last_used[v] <= self.last_used[s]:
                    self.stats["evacuations"] += 1
                    self._evict(v, f"victim of the evacuation of slot {s}")
                    self.copy_prefix(s, v, K)
                    self.last_used[v] = self.last_used[s]
                    evacuated = True
                    self.srv.log(f"[pcache] EVACUATE slot {s} -> {v} ({K} chunks)")
            if not evacuated:
                self._evict(s, "overwritten by the new request (no less valuable free slot)", lost=lost)
        if k0 and src != s:
            self._set_valid(s, 0, {}, future=fut)
            self.copy_prefix(src, s, k0)
        self._set_valid(s, k0, fut, future=fut)
        return CHUNK * p["hit"]

    def _evict(self, s, why, lost=None):
        """Slot s's cached pages are given up (logged); its keys no slot holds lose their snapshots."""
        K, ks = self.valid[s]
        gone = [j for j in (lost if lost is not None else range(1, K + 1)) if not self._held(ks[j], j, skip=s)]
        if gone:
            self.stats["evictions"] += 1
            self.stats["evicted_chunks"] += len(gone)
            self.srv.log(f"[pcache] EVICT slot {s}: {K} cached chunks, {len(gone)} held nowhere else are lost ({why}); "
                         f"last used {time.time() - self.last_used[s]:.0f} s ago")
        if lost is None:
            self._set_valid(s, 0, {})

    def after_prefill(self, s, p, ok=True):
        if not ok:
            self._set_valid(s, 0, {})
            return
        got = int(getattr(self.model, "_pcx_last_k0", 0) or 0)
        assert got == p["k0"], f"pcache: model resumed at chunk {got}, plan said {p['k0']}"
        self.model._pcx_last_k0 = 0
        ev = int((getattr(self.model, "_pcx_stats", None) or {}).get("evictions", 0))
        if ev > self.stats["store_evictions"]:
            self.srv.log(f"[pcache] snapshot store full: {ev - self.stats['store_evictions']} LRU snapshot(s) evicted")
            self.stats["store_evictions"] = ev
        if p["k0"]:
            self.stats["hits"] += 1
            self.stats["tokens_skipped"] += CHUNK * p["k0"]
        else:
            self.stats["misses"] += 1
        self._set_valid(s, p["nf"], p["keys"])
        if LOGV:
            st = getattr(self.model, "_pcx_index", {})
            self.srv.log(f"[pcache] slot {s}: n={p['n']} resume_at={CHUNK * p['k0']} src={p['src']} hit={CHUNK * p['hit']} "
                         f"valid={p['nf']} store={len(st)} stats={self.stats}")

    def finished(self, s):
        self.last_used[s] = time.time()

    def store_bytes(self):
        """Snapshot store, tile-padded bytes per card (all slots)."""
        tot = 0
        for rec, conv, carry in getattr(self.model, "_pcx_slots", None) or []:
            for t in [*rec, *carry] + [c for cs in conv for c in cs]:
                v = 1
                for d in list(t.padded_shape):
                    v *= int(d)
                tot += v * (4 if t.dtype == ttnn.float32 else 2)
        return tot

    def info(self):
        return dict(stats=self.stats, valid=[v[0] for v in self.valid], store=len(getattr(self.model, "_pcx_index", {})),
                    store_slots=len(getattr(self.model, "_pcx_slots", []) or []),
                    store_gib_per_card=round(self.store_bytes() / 2**30, 3), copy_prog_bytes=self.dev_bytes,
                    copy_log=self.copy_log[-20:])

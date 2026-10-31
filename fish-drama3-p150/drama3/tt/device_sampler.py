# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fish's codebook-head sampler on the card (Day 4b), so the tower's 9 segments chain without host round trips.

Host version (host/tower.py ``sample_top_k_top_p``, codebook heads: keep_crossing=False, top-k 30, top-p 0.9):
sort the logits, drop ranks whose cumulative softmax (over ALL logits) exceeds top_p and ranks >= top_k (rank 0
always kept), softmax over the kept ones, draw argmax(probs / noise) with exponential noise indexed by rank.
Here: fp32 top-32 (ttnn.topk, sorted), the full softmax normaliser from all 1024 logits, cumsum over the 32,
the same masks, then argmax(probs / noise[:32]) through a max + one-hot. The host draws the noise from the same
generator in the same order (4096 values per head, of which ranks 0..31 can matter), so for equal logits the
draw only differs where fp32 rounding on the card flips a comparison (cum vs top_p, near-equal scores).
"""

import os

import torch

import ttnn

K = 32  # top-k width on the card (>= Fish's codebook top-k of 30)


class DeviceSampler:
    def __init__(self, mesh, n_codes=1024, top_k=30, top_p=0.9, temperature=1.0):
        self.mesh, self.top_k, self.top_p, self.temperature = mesh, top_k, top_p, temperature

        def const(t, dtype=ttnn.float32):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh)

        self.rank = const(torch.arange(K, dtype=torch.float32).reshape(1, 1, 1, K))
        self.rank_ok = const((torch.arange(K) < top_k).float().reshape(1, 1, 1, K))  # rank < top_k
        self.first = const((torch.arange(K) == 0).float().reshape(1, 1, 1, K))
        self.code_ids = const(torch.arange(n_codes, dtype=torch.float32).reshape(1, 1, 1, n_codes))
        # DRAMA3_SAMPLER_LEAN=1 (port7, opt-in): the same draw in fewer programs. The keep mask in one compare
        # (cum + c <= top_p, c = -1e4 at rank 0 so it is always kept, +1e4 from rank top_k on: c = 0 leaves cum
        # exact), and the code read as sum(ids * (score == best)) without the lowest-rank tie break (two kept
        # ranks with bit-equal scores e / noise; the noise is continuous)
        self.lean = os.environ.get("DRAMA3_SAMPLER_LEAN", "0") == "1"
        c = torch.zeros(K)
        c[0], c[top_k:] = -1e4, 1e4
        self.keep_bias = const(c.reshape(1, 1, 1, K))
        # DRAMA3_SAMPLER_FUSED=1 (port10 bb12, opt-in): one row's draw as ONE program (FusedDraw) with the lean math
        self.fused = None
        if os.environ.get("DRAMA3_SAMPLER_FUSED", "0") == "1":
            nc = int(os.environ.get("DRAMA3_SAMPLER_FUSED_CORES", "4"))  # 1: the one-core kernel
            self.fused = FusedDraw(mesh, n_codes, top_k, top_p) if nc == 1 else FusedDrawMC(mesh, n_codes, top_k, top_p, nc)

    def __call__(self, logits, noise):
        """logits [1,1,1,n] fp32 (device), noise [1,1,1,K] fp32 (device) -> (code [1,1,1,1] fp32, one-hot [1,1,1,n] bf16).
        Batch mode: B rows on dim 2 ([1,1,B,n], [1,1,B,K] -> [1,1,B,1], [1,1,B,n])."""
        x = logits if self.temperature == 1.0 else ttnn.multiply(logits, 1.0 / self.temperature)
        if self.fused is not None and x.shape[2] == 1 and x.shape[-1] == self.fused.n:
            return self.fused(x, noise)
        vals, idx = ttnn.topk(x, K, dim=-1, largest=True, sorted=True)
        m = ttnn.slice(vals, [0, 0, 0, 0], [1, 1, vals.shape[2], 1])  # the max (per row: batch mode has B rows)
        z = ttnn.sum(ttnn.exp(ttnn.subtract(x, m)), dim=-1, keepdim=True)  # softmax normaliser over all logits
        e = ttnn.exp(ttnn.subtract(vals, m))
        cum = ttnn.cumsum(ttnn.divide(e, z), dim=-1)
        if self.lean:
            keep = ttnn.le(ttnn.add(cum, self.keep_bias), self.top_p)
            score = ttnn.divide(ttnn.multiply(e, keep), noise)
            hit = ttnn.eq(score, ttnn.max(score, dim=-1, keepdim=True))
            code = ttnn.sum(ttnn.multiply(ttnn.typecast(idx, ttnn.float32), hit), dim=-1, keepdim=True)
            return code, ttnn.typecast(ttnn.eq(self.code_ids, code), ttnn.bfloat16)
        keep = ttnn.multiply(ttnn.le(cum, self.top_p), self.rank_ok)  # drop = cum > top_p | rank >= top_k
        keep = ttnn.maximum(keep, self.first)  # rank 0 always kept
        # probs = softmax(kept); argmax(probs / noise) = argmax(e * keep / noise) (the common 1/sum cancels)
        score = ttnn.divide(ttnn.multiply(e, keep), noise)
        best = ttnn.max(score, dim=-1, keepdim=True)
        hit = ttnn.eq(score, best)
        rank = ttnn.min(ttnn.add(ttnn.multiply(self.rank, hit), ttnn.multiply(ttnn.rsub(hit, 1.0), 1e4)), dim=-1, keepdim=True)
        onehot_rank = ttnn.eq(self.rank, rank)  # first winning rank (ties: lowest rank, as torch.argmax)
        code = ttnn.sum(ttnn.multiply(ttnn.typecast(idx, ttnn.float32), onehot_rank), dim=-1, keepdim=True)
        onehot = ttnn.typecast(ttnn.eq(self.code_ids, code), ttnn.bfloat16)
        return code, onehot


KDIR = "models/demos/blackhole/drama3/kernels"


class FusedDraw:
    """DRAMA3_SAMPLER_FUSED=1 (port10 bb12): the lean draw of one row in one program on one core
    (kernels/sampler_fused_reader.cpp + sampler_fused_compute.cpp): top-32 on the RISC (ties: lower code), E = exp(x - m)
    over all logits on the SFPU (the ops ttnn.subtract / ttnn.exp run), z = sum(E) as a fixed-point sum rounded once,
    p = e / z and e / noise on the SFPU (div_binary_tile, as ttnn.divide), the cum as ttnn.cumsum's Kahan recurrence,
    then keep / argmax in rank order. The codes match the lean sampler's except
    where fp32 rounding of z, p or cum differs at a threshold (tests/check_sampler_fused.py counts them).
    Core: the last grid column, row 0 (outside the prefetcher's receiver cores (0..7, 0..7))."""

    def __init__(self, mesh, n_codes, top_k, top_p):
        import struct

        assert n_codes == 1024 and top_k <= K
        self.mesh, self.n, self.top_k = mesh, n_codes, top_k
        self.top_p_bits = struct.unpack("<I", struct.pack("<f", float(top_p)))[0]
        g = mesh.compute_with_storage_grid_size()
        c = ttnn.CoreCoord(int(os.environ.get("DRAMA3_SAMPLER_FUSED_CORE_X", g.x - 1)), 0)
        self.core, self.crs = c, ttnn.CoreRangeSet([ttnn.CoreRange(c, c)])
        self.ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in (0, 1, 6, 7, 8):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        self.ccfg.unpack_to_dest_mode = modes
        # DRAMA3_SAMPLER_FUSED_PROFILE=1 (checks): wall-clock stamps in row 0 of the code tile (the code still at (0, 0))
        self.defs = [("SAMPLER_PROFILE", "1")] if os.environ.get("DRAMA3_SAMPLER_FUSED_PROFILE", "0") == "1" else []

    def __call__(self, logits, noise):
        """logits [1, 1, 1, n] fp32 TILE, noise [1, 1, 1, K] fp32 TILE -> (code [1, 1, 1, 1] fp32, one-hot [1, 1, 1, n] bf16)."""
        code = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32] if self.defs else [1, 1, 1, 1]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        oh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        ct = [self.n, self.top_k, self.top_p_bits]
        for t in (logits, noise, code, oh):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        rt[self.core.x][self.core.y] = [logits.buffer_address(), noise.buffer_address(), code.buffer_address(), oh.buffer_address()]
        kd = lambda src, ct_, rt_, cfg, defines=(): ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct_, runtime_args=rt_, config=cfg, defines=list(defines),
        )
        kernels = [  # the reader source runs on both data-movement RISCs (SAMPLER_B: the BRISC half)
            kd("sampler_fused_reader.cpp", ct, rt, ttnn.ReaderConfigDescriptor(), self.defs),
            kd("sampler_fused_reader.cpp", ct, rt, ttnn.WriterConfigDescriptor(), [("SAMPLER_B", "1")] + self.defs),
            kd("sampler_fused_compute.cpp", [], [], self.ccfg),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.crs, initial_value=0) for i in range(4)]
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(  # noqa: E731
            total_size=page * n, core_ranges=self.crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )
        f32, bf = ttnn.float32, ttnn.bfloat16
        cbs = [cb(0, f32, 4096, 1), cb(1, f32, 4096, 1), cb(2, f32, 4096, 1), cb(3, f32, 4096, 1), cb(4, bf, 2048, 2), cb(5, f32, 4096, 1)] + [cb(i, f32, 4096, 1) for i in range(6, 11)]
        ttnn.generic_op([logits, noise, code, oh], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return code, oh


class FusedDrawMC(FusedDraw):
    """DRAMA3_SAMPLER_FUSED_CORES=NC (2, 4 or 8; with DRAMA3_SAMPLER_FUSED=1): the fused draw over NC cores of the last grid
    column (rows 0 ..; kernels/sampler_mc_*.cpp): core j takes the top-32 and the exp sum of logits [j N / NC, ..),
    core 0 merges and draws. z = sum_j exp(m_j - m) * sum_j (instead of one sum over all exp(x - m)): the codes can
    differ from the one-core kernel's where fp32 rounding of z flips a threshold."""

    def __init__(self, mesh, n_codes, top_k, top_p, nc):
        super().__init__(mesh, n_codes, top_k, top_p)
        assert nc in (2, 4, 8)
        self.nc = nc
        g = mesh.compute_with_storage_grid_size()
        x = self.core.x
        self.cores = [ttnn.CoreCoord(x, r) for r in range(nc)]
        assert nc <= g.y
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(self.cores[0], self.cores[-1])])
        cn = mesh.worker_core_from_logical_core(self.cores[0])
        self.cnoc = (cn.x, cn.y)
        modes = list(self.ccfg.unpack_to_dest_mode)
        for i in (0, 1, 6, 7, 8, 11, 12):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        self.ccfg.unpack_to_dest_mode = modes

    def __call__(self, logits, noise):
        code = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, 1]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        oh = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        ct = [self.n, self.top_k, self.top_p_bits, self.nc]
        for t in (logits, noise, code, oh):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        addrs = [logits.buffer_address(), noise.buffer_address(), code.buffer_address(), oh.buffer_address()]
        for j, c in enumerate(self.cores):
            rt[c.x][c.y] = addrs + [j, self.cnoc[0], self.cnoc[1]]
            rc[c.x][c.y] = [j]
        kd = lambda src, ct_, rt_, cfg, defines=(): ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{KDIR}/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
            compile_time_args=ct_, runtime_args=rt_, config=cfg, defines=list(defines),
        )
        kernels = [
            kd("sampler_mc_reader.cpp", ct, rt, ttnn.ReaderConfigDescriptor()),
            kd("sampler_mc_reader.cpp", ct, rt, ttnn.WriterConfigDescriptor(), [("SAMPLER_B", "1")]),
            kd("sampler_mc_compute.cpp", [], rc, self.ccfg),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.crs, initial_value=0) for i in range(6)]
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(  # noqa: E731
            total_size=page * n, core_ranges=self.crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )
        f32, bf = ttnn.float32, ttnn.bfloat16
        cbs = [cb(i, f32, 4096, 1) for i in (0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12)] + [cb(4, bf, 2048, 2), cb(5, f32, 8192, 1)]
        ttnn.generic_op([logits, noise, code, oh], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return code, oh

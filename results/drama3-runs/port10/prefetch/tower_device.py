# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drama 3's codebook tower on the P150 (Day 4), next to the backbone.

Per layer (Fish's DecoderLayer): ``out = norm_f(mlp(norm_p(h1)) + h1)``, ``h1 = mixed + attn(norm_i(mixed))``,
``mixed = eh_proj([norm_e(e), norm_h(h)])``; Gemma (1 + w) norms, gated attention, SwiGLU MLP.
- Attention is qwen36's Qwen36GatedAttention (same geometry as the backbone: 16 / 2 heads, head_dim 256, gated,
  q/k norms, RoPE on 64 dims with base 1e7) on its own paged KV cache. Fish's tower rotates INTERLEAVED pairs;
  permuting the q/k projection rows (and q/k norm weights) inside the 64 rotary dims to [even | odd] turns that
  into the backbone's neox rotation exactly (q.k is invariant under a shared permutation).
- eh_proj is split into its e / h halves (two matmuls, no concat). Dense weights BF16, HiFi4 + fp32 accumulation.
- Codebook heads keep only the codec's 1024 legal codes (identical to masking the rest).
- Token / codebook embeddings are looked up on the host (one 2048-wide row per layer and frame).
Sampling stays on the host and uses CodebookTower's functions in the same RNG order, so for equal logits the
sampled codes are identical to the host tower's.
"""

import os

import torch

import ttnn
from models.demos.blackhole.drama3.host.tower import (
    CODEBOOK_TOP_K,
    MIN_TEMPERATURE,
    RAS_TEMPERATURE,
    RAS_TOP_K,
    RAS_TOP_P,
    RESIDUAL_CODEBOOK_SIZE,
    sample_top_k_top_p,
)
from models.demos.blackhole.drama3.tt import decode_norm
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention

BLOCK = 64


def _rotary_perm(head_dim, rotary_dim):
    """Index map new <- old within one head: rotary dims [0,2,..,rd-2, 1,3,..,rd-1], then the pass dims."""
    return torch.cat([torch.arange(0, rotary_dim, 2), torch.arange(1, rotary_dim, 2), torch.arange(rotary_dim, head_dim)])


def sample_with_noise(logits, temperature, top_k, top_p, noise, keep_crossing=False, m=64):
    """host/tower.py sample_top_k_top_p with its exponential noise passed in (drawn earlier from the same
    generator, same size). Softmax / cumsum / masks / argmax(probs / noise) over the top m ranks only, which hold
    every kept rank whenever the top-p crossing or the top-k cut falls inside them; else the full sort."""
    scaled = logits.float() / max(temperature, MIN_TEMPERATURE)
    if m < scaled.numel():
        vals, ids = torch.topk(scaled, m)
        cum = torch.cumsum(torch.exp(vals - torch.logsumexp(scaled, -1)), -1)
        if cum[-2] > top_p or top_k <= m:
            drop = cum > top_p
            if keep_crossing:
                drop[1:] = drop[:-1].clone()
            drop |= torch.arange(m) >= top_k
            drop[0] = False
            probs = torch.softmax(vals.masked_fill(drop, float("-inf")), -1)
            return int(ids[torch.argmax(probs / noise[:m])])
    sorted_logits, ids = torch.sort(scaled, descending=True)
    cum = torch.cumsum(torch.softmax(sorted_logits, -1), -1)
    drop = cum > top_p
    if keep_crossing:
        drop[1:] = drop[:-1].clone()
    drop |= torch.arange(sorted_logits.numel()) >= top_k
    drop[0] = False
    probs = torch.softmax(sorted_logits.masked_fill(drop, float("-inf")), -1)
    return int(ids[torch.argmax(probs / noise)])


def _batch_main():
    """True when the running program is host/batch.py or host/cbatch.py (python -m ...)."""
    import sys

    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return (getattr(spec, "name", "") or "").rsplit(".", 1)[-1] in ("batch", "cbatch")


class TowerLayerDevice:
    def __init__(self, mesh, host_layer, cfg, max_seq, page_table, ckc, mlp_dtype):
        nh, nkv, hd, rd = cfg["heads"], cfg["kv_heads"], cfg["head_dim"], cfg["rotary_dim"]
        H = host_layer.eh.shape[0]
        self.mesh, self.ckc, self.eps = mesh, ckc, cfg["eps"]
        self.host = host_layer  # its embedding table is used on the host

        def w(t, dtype=ttnn.bfloat16):  # [out, in] -> device [in, out]
            return ttnn.from_torch(t.T.contiguous().to(torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh)

        def g(t):  # Gemma norm weight -> (1 + w)
            return ttnn.from_torch((1.0 + t).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh)

        L = host_layer
        self.n_e, self.n_h = g(L.enorm), g(L.hnorm)
        from models.demos.blackhole.drama3.tt import decode_norm

        self.row_norm = decode_norm.ShardedRowNorm(mesh, H) if decode_norm.ENABLED else None
        self.rm = {}  # tile weight -> ROW_MAJOR [1, 1, H / 32, 32] copy for the one-token norm
        if self.row_norm is not None:
            for t in (L.enorm, L.hnorm, L.in_norm, L.post_norm, L.final_norm):
                self.rm[id(t)] = ttnn.from_torch(
                    (1.0 + t).to(torch.bfloat16).reshape(1, 1, H // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
                )
        self.w_e, self.w_h = w(L.eh[:, :H]), w(L.eh[:, H:])
        self.n_in, self.n_post, self.n_final = g(L.in_norm), g(L.post_norm), g(L.final_norm)
        if self.row_norm is not None:  # device tile weight -> its ROW_MAJOR twin
            src = (L.enorm, L.hnorm, L.in_norm, L.post_norm, L.final_norm)
            dev = (self.n_e, self.n_h, self.n_in, self.n_post, self.n_final)
            self.rm = {id(d): self.rm[id(t)] for d, t in zip(dev, src)}
        self.w1, self.w3, self.w2 = w(L.w1, mlp_dtype), w(L.w3, mlp_dtype), w(L.w2, mlp_dtype)
        # DRAMA3_ADDNORM_FUSED=1 (port8, opt-in): every one-token residual add + norm as one program (tt/addnorm_fused.py)
        from models.demos.blackhole.drama3.tt import addnorm_fused

        self.an = addnorm_fused.get(mesh, H) if addnorm_fused.ENABLED and self.row_norm is not None else None
        # DRAMA3_TOWER_DS=1 (port7 precision candidate): the one-token eh_h / w1 / w3 / w2 matmuls DRAM-sharded at
        # HiFi2 (tests/bench_dram_mm.py: 410-440 GB/s vs ~340 at HiFi4); the normed row stays width-sharded on the
        # norm's 8 cores, which are the matmuls' 8 cores. Second copies of the weights (prefill keeps the others).
        self.ds = os.environ.get("DRAMA3_TOWER_DS", "0") == "1" and self.row_norm is not None
        if self.ds:
            nb = mesh.dram_grid_size().x
            crs = ttnn.num_cores_to_corerangeset(8, mesh.compute_with_storage_grid_size(), row_wise=True)

            def wds(t, dtype):
                K, N = t.shape[1], t.shape[0]
                mc = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))}), (K, N // nb), ttnn.ShardOrientation.ROW_MAJOR),
                )
                return ttnn.from_torch(t.T.contiguous().to(torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc)

            def omc(n):
                return ttnn.create_sharded_memory_config((32, n // 8), crs, ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)

            def pc(K, N, ibw, act=None):
                return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=ibw, per_core_M=1, per_core_N=N // 32 // 8, fused_activation=act)

            F = int(L.w1.shape[0])
            self.ds_w = dict(h=wds(L.eh[:, H:], ttnn.bfloat16), w1=wds(L.w1, mlp_dtype), w3=wds(L.w3, mlp_dtype), w2=wds(L.w2, mlp_dtype))
            self.ds_mc = dict(H=omc(H), F=omc(F))
            self.ds_pc = dict(h=pc(H, H, 8), w1=pc(H, F, 4, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)), w3=pc(H, F, 4), w2=pc(F, H, 8))
            self.ds_ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True)

        # attention weights in qwen36's layout: q_proj rows per head [q_h | gate_h], q/k rotary dims de-interleaved
        perm = _rotary_perm(hd, rd)
        qw, kvw = nh * hd, nkv * hd
        q, gate, k, v = L.wqkv.split([qw, qw, kvw, kvw], 0)
        q = q.view(nh, hd, H)[:, perm]
        q_proj = torch.cat([q, gate.view(nh, hd, H)], 1).reshape(2 * qw, H)
        k_proj = k.view(nkv, hd, H)[:, perm].reshape(kvw, H)
        sd = {
            "q_proj.weight": q_proj,
            "k_proj.weight": k_proj,
            "v_proj.weight": v,
            "o_proj.weight": L.wo,
            "q_norm.weight": L.qn[perm],  # Qwen36 load_norm adds the +1
            "k_norm.weight": L.kn[perm],
        }
        from models.demos.blackhole.qwen36.tt.attention import weights as attn_weights

        # DRAMA3_TOWER_ATTN_BFP4=1 (port8 precision candidate): the tower attention projections in BFP4
        bfp4 = os.environ.get("DRAMA3_TOWER_ATTN_BFP4", "0") == "1"
        if bfp4:
            attn_weights.PROJ_DTYPE = ttnn.bfloat4_b
        self.attn = Qwen36GatedAttention(
            mesh,
            AttentionConfig(num_heads=nh, num_kv_heads=nkv, head_dim=hd, norm_eps=cfg["eps"], max_seq_len=max_seq),
            {k_: v_.to(torch.bfloat16) for k_, v_ in sd.items()},
            None,
        )
        attn_weights.PROJ_DTYPE = ttnn.bfloat8_b
        from models.demos.blackhole.drama3.tt.backbone_paged import kv_pool_blocks

        shape = [kv_pool_blocks(max_seq // BLOCK), nkv, BLOCK, hd]  # batch mode: a pool (port10: compact option)
        self.k_cache = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        self.v_cache = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        self.attn.set_paged_kv_cache(self.k_cache, self.v_cache)
        self.page_table = page_table

    def _ds(self, x, w, n, keep=False):
        """DRAM-sharded one-token matmul of a width-sharded row; keep=True: the output stays sharded, else
        interleaved (and x is freed)."""
        y = ttnn.linear(x, self.ds_w[w], program_config=self.ds_pc[w], memory_config=self.ds_mc[n], compute_kernel_config=self.ds_ckc)
        if keep:
            return y
        ttnn.deallocate(x)
        out = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(y)
        return out

    def _lin(self, x, w):
        return ttnn.linear(x, w, compute_kernel_config=self.ckc)

    def _norm(self, x, w):
        if self.row_norm is not None and x.shape[-2] <= decode_norm.ROWS:  # ROWS > 1 only in batch mode
            return self.row_norm(x, self.rm[id(w)], self.eps, self.ckc)
        return ttnn.rms_norm(x, weight=w, epsilon=self.eps, compute_kernel_config=self.ckc)

    def __call__(self, h, e, cos, sin, pos=None, chunk_pt=None, csi=None, pre_a=False):
        """h, e: [1, N, H] device. Decode (N == 1): pos = cur_pos tensor; prefill: chunk_pt / csi.
        pre_a (port7 DRAMA3_TOWER_ETABLE): e already is the e half of eh_proj, norm_e(e) @ W_e, from a table."""
        a = e if pre_a else self._lin(self._norm(e, self.n_e), self.w_e)
        ds = self.ds and h.shape[-2] == 1
        if self.an is not None and h.shape[-2] == 1 and not ds:
            return self._call_addnorm(h, e, cos, sin, pos, pre_a)
        if ds:
            b = self._ds(self.row_norm.sharded(h, self.rm[id(self.n_h)], self.eps, self.ckc), "h", "H")
        else:
            b = self._lin(self._norm(h, self.n_h), self.w_h)
        mixed = ttnn.add(a, b)
        if not pre_a:
            ttnn.deallocate(a)
        ttnn.deallocate(b)
        xi = self._norm(mixed, self.n_in)
        if chunk_pt is None:
            att = self.attn.forward(xi, cos, sin, position_tensor=pos, page_table=self.page_table)
        else:
            att = self.attn.forward(xi, cos, sin, page_table=self.page_table, chunk_page_table=chunk_pt, chunk_start_idx_tensor=csi)
        ttnn.deallocate(xi)
        if len(att.shape) == 4:
            att = ttnn.reshape(att, (1, att.shape[-2], att.shape[-1]))
        h1 = ttnn.add(mixed, att)
        ttnn.deallocate(mixed)
        ttnn.deallocate(att)
        if ds:
            ffs = self.row_norm.sharded(h1, self.rm[id(self.n_post)], self.eps, self.ckc)
            u, v = self._ds(ffs, "w1", "F", keep=True), self._ds(ffs, "w3", "F", keep=True)
            ttnn.deallocate(ffs)
            m = ttnn.mul(u, v, memory_config=self.ds_mc["F"])
            ttnn.deallocate(u)
            ttnn.deallocate(v)
            d = self._ds(m, "w2", "H")
            s = ttnn.add(d, h1)
            ttnn.deallocate(d)
            ttnn.deallocate(h1)
            out = self._norm(s, self.n_final)
            ttnn.deallocate(s)
            return out
        ff = self._norm(h1, self.n_post)
        u = ttnn.linear(ff, self.w1, activation="silu", compute_kernel_config=self.ckc)
        v = self._lin(ff, self.w3)
        ttnn.deallocate(ff)
        m = ttnn.mul(u, v)
        ttnn.deallocate(u)
        ttnn.deallocate(v)
        d = self._lin(m, self.w2)
        ttnn.deallocate(m)
        s = ttnn.add(d, h1)
        ttnn.deallocate(d)
        ttnn.deallocate(h1)
        out = self._norm(s, self.n_final)
        ttnn.deallocate(s)
        return out

    def _call_addnorm(self, h, e, cos, sin, pos, pre_a):
        """One-token decode with the fused add + norm programs (port8): same math as __call__, 3 adds folded."""
        an, rm, eps = self.an, self.rm, self.eps
        if pre_a:
            a = e
        else:
            _, en = an(e, None, rm[id(self.n_e)], eps)
            a = self._lin(en, self.w_e)
            ttnn.deallocate(en)
        _, hn = an(h, None, rm[id(self.n_h)], eps)
        b = self._lin(hn, self.w_h)
        ttnn.deallocate(hn)
        mixed, xi = an(a, b, rm[id(self.n_in)], eps)
        if not pre_a:
            ttnn.deallocate(a)
        ttnn.deallocate(b)
        att = self.attn.forward(xi, cos, sin, position_tensor=pos, page_table=self.page_table)
        ttnn.deallocate(xi)
        h1, ff = an(mixed, att, rm[id(self.n_post)], eps)
        ttnn.deallocate(mixed)
        ttnn.deallocate(att)
        u = ttnn.linear(ff, self.w1, activation="silu", compute_kernel_config=self.ckc)
        v = self._lin(ff, self.w3)
        ttnn.deallocate(ff)
        m = ttnn.mul(u, v)
        ttnn.deallocate(u)
        ttnn.deallocate(v)
        d = self._lin(m, self.w2)
        ttnn.deallocate(m)
        _, out = an(d, h1, rm[id(self.n_final)], eps, want_h=False)
        ttnn.deallocate(d)
        ttnn.deallocate(h1)
        return out


class TowerDevice:
    def __init__(
        self, mesh, host_tower, rope, max_seq=2048, mlp_dtype=ttnn.bfloat16, fidelity=ttnn.MathFidelity.HiFi4, cos_sin_host=None
    ):
        self.mesh, self.host, self.rope = mesh, host_tower, rope
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=fidelity, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.page_table_host = torch.arange(max_seq // BLOCK, dtype=torch.int32).unsqueeze(0)
        self.page_table = ttnn.from_torch(self.page_table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        cfg = host_tower.cfg
        mk = lambda L: TowerLayerDevice(mesh, L, cfg, max_seq, self.page_table, self.ckc, mlp_dtype)  # noqa: E731
        self.shared = [mk(L) for L in host_tower.shared]
        self.residual = [mk(L) for L in host_tower.residual]
        self.heads = [
            ttnn.from_torch(h[:RESIDUAL_CODEBOOK_SIZE].T.contiguous().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh)
            for h in host_tower.heads
        ]
        self.H = host_tower.heads[0].shape[1]
        # DRAMA3_TOWER_ETABLE=1 (port7, opt-in): the e half of eh_proj depends only on the code, so for every layer
        # but shared 0 (whose e also carries the shared input) it is a table norm_e(emb[c]) @ W_e (fp32 on the
        # host, rounded to bf16): shared 1-3 + residual 0 looked up on the host, residual 1-8 by the one-hot
        # matmul on the card (instead of the embedding row + one-token norm + matmul)
        self.etable = os.environ.get("DRAMA3_TOWER_ETABLE", "0") == "1"
        if self.etable:
            from models.demos.blackhole.drama3.host.tower import gemma_norm

            def table(L, n):
                with torch.no_grad():
                    return (gemma_norm(L.emb[:n], L.enorm, cfg["eps"]) @ L.eh[:, : self.H].T).to(torch.bfloat16)

            layers = list(host_tower.shared) + list(host_tower.residual)
            ns = len(host_tower.shared)
            # host tables of every layer but shared 0 (the fused trace looks up shared 1-3 + residual 0, step() all);
            # device tables (1024 legal codes) of residual 1-8 for the one-hot matmul
            self.a_host = [None] + [table(L, L.emb.shape[0] if i <= ns else RESIDUAL_CODEBOOK_SIZE) for i, L in enumerate(layers) if i > 0]
            self.a_dev = [None] + [ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=mesh) for t in self.a_host[ns + 1 :]]
        self.pos_buf = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        self.cos_sin_host = cos_sin_host or rope.get_cos_sin_host  # the backbone's (full width with DRAMA3_ROPE_MATMUL)
        c, s = self.cos_sin_host(0)
        self.cos_buf, self.sin_buf = ttnn.to_device(c, mesh), ttnn.to_device(s, mesh)
        # DRAMA3_ATTN_FUSED_PRE=1 (port5, opt-in): the one-token attention pre-SDPA step as one kernel
        from models.demos.blackhole.drama3.tt import attn_fused_decode

        attn_fused_decode.install([L.attn for L in self.shared + self.residual], mesh)
        # DRAMA3_TOWER_PREFETCH=1 (port10, opt-in): the one-row decode (fused frame trace, step()) streams its 6 x 13
        # matmul weights through the DRAM-core tensor prefetcher (tt/tower_prefetch.py); after the merged qkv weight
        # exists, before any trace capture. A no-op with DRAMA3_TOWER_DS and in batch mode (host/batch, host/cbatch).
        from models.demos.blackhole.drama3.tt import tower_prefetch

        self.pf = None
        if tower_prefetch.ENABLED and not any(L.ds for L in self.shared + self.residual) and not _batch_main():
            shared = os.environ.get("DRAMA3_BB_PREFETCH", "0") == "1" or os.environ.get("DRAMA3_TOWER_PREFETCH_SHARED", "0") == "1"
            cls = tower_prefetch.SharedTower if shared else tower_prefetch.TowerPrefetch  # shared: one GCB with the backbone
            self.pf = cls(mesh, self.shared + self.residual)
        self._pf_capture = False

    def _pf_frame(self):
        import contextlib

        return self.pf.frame(capture=self._pf_capture) if self.pf is not None else contextlib.nullcontext()

    # host-side embedding rows ---------------------------------------------------------------------
    def _emb_rows(self, layer, code_ids, label_is_semantic, shared_input=None):
        e = torch.where(label_is_semantic[:, None], layer.host.emb[code_ids.clamp(0, layer.host.emb.shape[0] - 1)], 0.0)
        return e if shared_input is None else e + shared_input

    def _up(self, x):
        return ttnn.from_torch(x.reshape(1, -1, self.H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)

    # prefill -------------------------------------------------------------------------------------
    def prefill(self, h_dev, rows, T, collect=False, start=0, last=True):
        """Fill the tower caches from the backbone's final-norm prefill output h_dev [1, bucket, H] (device) for the
        prompt rows start..T-1 (rows: the whole prompt). Row p is labelled with frame p + 1; with last=True the
        last real row gets an eos placeholder that the first step rewrites, with last=False (a cached prefix) it is
        labelled with rows[T], which must exist. start: multiple of 128, as the backbone's prefill."""
        bucket = h_dev.shape[-2]
        n = T - start
        nxt = torch.zeros(bucket, rows.shape[1], dtype=torch.long)
        if last:
            nxt[: n - 1] = rows[start + 1 : T]
            nxt[n - 1, 0] = self.host.eos
        else:
            nxt[:n] = rows[start + 1 : T + 1]
        inp = torch.zeros(bucket, rows.shape[1], dtype=torch.long)
        inp[:n] = rows[start:T]
        lab = self.host.is_semantic(nxt[:, 0])
        si = self.host.shared_input(inp)
        cos, sin = self.rope.get_prefill_rot_mats(start, bucket)
        # fixed per bucket (no request-time compile); padded rows sit at positions >= T
        chunk_pt = ttnn.from_torch(
            self.page_table_host[:, start // BLOCK : (start + bucket) // BLOCK].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
        )
        csi = ttnn.from_torch(torch.tensor([start], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh)
        h = h_dev
        outs = []
        for i, layer in enumerate(self.shared + self.residual):
            code = nxt[:, 1] if i < len(self.shared) else nxt[:, 1 + i - len(self.shared)]
            e = self._up(self._emb_rows(layer, code, lab, si if i == 0 else None))
            y = layer(h, e, cos, sin, chunk_pt=chunk_pt, csi=csi)
            ttnn.deallocate(e)
            if h is not h_dev:
                ttnn.deallocate(h)
            h = y
            if collect:
                outs.append(ttnn.to_torch(h).float().reshape(-1, self.H)[:n])
        ttnn.deallocate(h)
        return outs

    # traced frame -------------------------------------------------------------------------------
    def _seg(self, i, h, pre_a=False):
        """Segment i: 0 = shared 0..3 + residual 0 + head 0; r = residual r + head r. -> (h, logits).
        pre_a: the e buffers of every layer but shared 0 hold table rows (DRAMA3_TOWER_ETABLE)."""
        layers = [*self.shared, self.residual[0]] if i == 0 else [self.residual[i]]
        bufs = self.e_bufs[: len(self.shared) + 1] if i == 0 else [self.e_bufs[len(self.shared) + i]]
        h0 = h  # the segment input (h_in or the previous segment's output) is never freed here
        for j, (layer, e) in enumerate(zip(layers, bufs)):
            y = layer(h, e, self.cos_buf, self.sin_buf, pos=self.pos_buf, pre_a=pre_a and (i > 0 or j > 0))
            if h is not h0:
                ttnn.deallocate(h)
            h = y
        lg = ttnn.linear(h, self.heads[i], dtype=ttnn.float32, compute_kernel_config=self.ckc)
        return h, lg

    def capture(self, h_in):
        """Capture the 9 frame segments as traces. h_in: the persistent backbone final-norm row [1, 1, H] (the
        backbone decode trace's output), read in place. Call after every program is compiled (step() once);
        the captures write garbage into the tower caches at the staged position, which the next frame overwrites."""
        self.h_in = h_in
        zeros = torch.zeros(1, 1, self.H)
        self.e_bufs = [
            ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)
            for _ in self.shared + self.residual
        ]
        # compile run of the exact traced bodies (same shapes as step(), but on persistent inputs)
        h = h_in
        for i in range(len(self.residual)):
            h_new, lg = self._seg(i, h)
            ttnn.deallocate(lg)
            if h is not h_in:
                ttnn.deallocate(h)
            h = h_new
        ttnn.deallocate(h)
        ttnn.synchronize_device(self.mesh)
        self.traces, h = [], h_in
        for i in range(len(self.residual)):
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            h, lg = self._seg(i, h)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces.append((tid, h, lg))
        ttnn.synchronize_device(self.mesh)

    def _put(self, buf, row):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(row.reshape(1, 1, -1).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT), buf)

    def step_traced(self, cand_logits, pos, input_frame, window, gen, temperature=1.0, top_p=0.9, top_k=None):
        """step() on the captured segments: per frame 9 replays, 9 head-logit readbacks, 13 embedding-row copies."""
        t = self.host
        top_k = top_k or len(t.candidates)
        first = int(t.candidates[sample_top_k_top_p(cand_logits, temperature, top_k, top_p, gen, keep_crossing=True)])
        ras = int(t.candidates[sample_top_k_top_p(cand_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, gen, keep_crossing=True)])
        main = ras if (first in window and t.sem0 <= first < t.sem0 + t.cb_size) else first
        code0 = min(max(main - t.sem0, 0), t.cb_size - 1)
        lab = t.is_semantic(torch.tensor([main]))
        si = t.shared_input(input_frame.reshape(1, -1))
        self._stage_pos(pos)
        c0 = torch.tensor([code0])
        ns = len(self.shared)
        for i, layer in enumerate(self.shared):
            self._put(self.e_bufs[i], self._emb_rows(layer, c0, lab, si if i == 0 else None))
        self._put(self.e_bufs[ns], self._emb_rows(self.residual[0], c0, lab))
        codes = [code0]
        k = min(top_k, CODEBOOK_TOP_K)
        pad = torch.full((t.cb_size - RESIDUAL_CODEBOOK_SIZE,), float("-inf"))
        for r, (tid, _, lg) in enumerate(self.traces):
            if r > 0:
                self._put(self.e_bufs[ns + r], self._emb_rows(self.residual[r], torch.tensor([codes[-1]]), lab))
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
            logits = ttnn.to_torch(lg).float().reshape(-1)[:RESIDUAL_CODEBOOK_SIZE]
            codes.append(sample_top_k_top_p(torch.cat([logits, pad]), temperature, k, top_p, gen))
        return torch.tensor([main, *codes], dtype=torch.long), first != main

    # one trace per frame, codes sampled on the card ---------------------------------------------
    def _fused_body(self):
        with self._pf_frame():
            return self._fused_body_inner()

    def _fused_body_inner(self):
        """All 9 segments in one body: head r's code is drawn on the card (DeviceSampler, noise row r of
        noise_buf) and embedded for residual layer r + 1 by a one-hot matmul. -> codes [1,1,1,9] fp32."""
        h, codes, self._lgs = self.h_in, [], []
        for i in range(self.n_host_rows):
            ttnn.copy(ttnn.slice(self.e_host_buf, [0, i, 0], [1, i + 1, self.H]), self.e_bufs[i])
        for i in range(len(self.residual)):
            h_new, lg = self._seg(i, h, pre_a=self.etable)
            self._lgs.append(lg)
            if h is not self.h_in:
                ttnn.deallocate(h)
            h = h_new
            noise = ttnn.slice(self.noise_buf, [0, 0, i, 0], [1, 1, i + 1, self.noise_buf.shape[-1]])
            code, onehot = self.sampler(ttnn.reshape(lg, (1, 1, 1, lg.shape[-1])), noise)
            codes.append(code)
            if i + 1 < len(self.residual):
                tab = self.a_dev[i + 1] if self.etable else self.res_emb[i + 1]
                e = ttnn.matmul(onehot, tab, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
                ttnn.copy(ttnn.reshape(e, (1, 1, self.H)), self.e_bufs[len(self.shared) + i + 1])
        ttnn.deallocate(h)
        return ttnn.concat(codes, dim=-1)

    def prepare_fused(self, top_p=0.9, temperature=1.0):
        """Persistent inputs of the fused frame trace. Call BEFORE the backbone's decode trace is captured: a
        buffer allocated after a capture may overlap that trace's intermediates and be overwritten by its replays."""
        from models.demos.blackhole.drama3.tt.device_sampler import K, DeviceSampler

        zeros = torch.zeros(1, 1, self.H)
        self.e_bufs = [ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh) for _ in self.shared + self.residual]
        self.noise_buf = ttnn.from_torch(torch.ones(1, 1, len(self.residual), K), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh)
        self.res_emb = [
            ttnn.from_torch(L.host.emb[:RESIDUAL_CODEBOOK_SIZE].to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.mesh)
            for L in self.residual
        ]
        self.sampler = DeviceSampler(self.mesh, RESIDUAL_CODEBOOK_SIZE, CODEBOOK_TOP_K, top_p, temperature)
        self.fused_params = (top_p, temperature)
        # the host rows of the shared layers + residual 0 in ONE buffer (one upload per frame instead of 5)
        self.n_host_rows = len(self.shared) + 1
        self.e_host_buf = ttnn.from_torch(
            torch.zeros(1, self.n_host_rows, self.H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh
        )

    def capture_fused(self, h_in):
        """Device-side sampling (DRAMA3_TOWER_DEVICE_SAMPLING=1): capture the whole frame as ONE trace (after
        prepare_fused). Per frame the host samples only the main token, uploads 5 embedding rows + the 9 heads'
        noise, replays, reads 9 codes."""
        self.h_in = h_in
        ttnn.deallocate(self._fused_body())  # compile run
        ttnn.synchronize_device(self.mesh)
        self.fused_tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._pf_capture = True  # the prefetch request is captured too: every replay re-sends it
        try:
            self.fused_codes = self._fused_body()
        finally:
            self._pf_capture = False
        self.fused_lgs = self._lgs  # the heads' logits stay readable after a replay (checks)
        ttnn.end_trace_capture(self.mesh, self.fused_tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)

    def prepare_frame(self, input_frame, gen):
        """The host work of a frame that does not need the backbone logits, in the RNG order of step_fused
        (main draw, RAS draw, then the 9 heads): call it while the backbone trace runs."""
        from models.demos.blackhole.drama3.tt.device_sampler import K

        t = self.host
        n = len(t.candidates)
        n1 = torch.empty(n).exponential_(1, generator=gen)
        n2 = torch.empty(n).exponential_(1, generator=gen)
        heads = torch.stack([torch.empty(t.cb_size).exponential_(1, generator=gen) for _ in self.residual])
        noise_host = ttnn.from_torch(heads[None, None, :, :K].contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        # input_frame None (port7 frame pipe): the noise only, the caller sets "si" once the frame's codes are back
        si = None if input_frame is None else t.shared_input(input_frame.reshape(1, -1))
        return dict(n1=n1, n2=n2, heads=heads, noise_host=noise_host, si=si)

    def step_fused(self, cand_logits, pos, input_frame, window, gen, temperature=1.0, top_p=0.9, top_k=None, debug=False, pre=None):
        """pre (prepare_frame's result): the noise and the shared input come from it, gen is not touched."""
        if pre is not None:
            return self._step_fused_pre(cand_logits, pos, window, temperature, top_p, top_k, debug, pre)
        return self._step_fused(cand_logits, pos, input_frame, window, gen, temperature, top_p, top_k, debug)

    def _host_rows(self, c0, lab, si):
        """The fused trace's host rows: shared 0 (embedding + shared input), shared 1-3, residual 0 (embedding rows,
        or with DRAMA3_TOWER_ETABLE their projected table rows)."""
        layers = [*self.shared, self.residual[0]]
        if not self.etable:
            return [self._emb_rows(layer, c0, lab, si if i == 0 else None) for i, layer in enumerate(layers)]
        return [self._emb_rows(layers[0], c0, lab, si)] + [self._a_rows(i, c0, lab) for i in range(1, len(layers))]

    def _a_rows(self, i, code_ids, lab):
        t = self.a_host[i]
        return torch.where(lab[:, None], t[code_ids.clamp(0, t.shape[0] - 1)].float(), 0.0)

    def _upload_host_rows(self, rows):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.cat(rows).reshape(1, self.n_host_rows, -1).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT),
            self.e_host_buf,
        )

    def _step_fused_pre(self, cand_logits, pos, window, temperature, top_p, top_k, debug, pre):
        main, code0, redrawn = self.enqueue_fused_pre(cand_logits, pos, window, temperature, top_p, top_k, pre)
        frame = torch.cat([torch.tensor([main, code0]), self.read_fused()])
        return (frame, redrawn, pre["heads"]) if debug else (frame, redrawn)

    def read_fused(self):
        """The 9 head codes of the last fused replay (blocking read)."""
        return ttnn.to_torch(self.fused_codes).float().reshape(-1)[: len(self.residual)].round().long()

    def enqueue_fused_pre(self, cand_logits, pos, window, temperature, top_p, top_k, pre):
        """The host draws of the main token, the uploads and the fused replay, without the readback (port7 frame pipe:
        the next backbone step is enqueued behind it). -> (main token, code 0, RAS redraw?)"""
        assert (top_p, temperature) == self.fused_params and (top_k or 1 << 30) >= CODEBOOK_TOP_K
        t = self.host
        top_k = top_k or len(t.candidates)
        first = int(t.candidates[sample_with_noise(cand_logits, temperature, top_k, top_p, pre["n1"], keep_crossing=True)])
        ras = int(t.candidates[sample_with_noise(cand_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, pre["n2"], keep_crossing=True)])
        main = ras if (first in window and t.sem0 <= first < t.sem0 + t.cb_size) else first
        code0 = min(max(main - t.sem0, 0), t.cb_size - 1)
        lab = t.is_semantic(torch.tensor([main]))
        c0 = torch.tensor([code0])
        rows = self._host_rows(c0, lab, pre["si"])
        if not pre.get("noise_up"):
            ttnn.copy_host_to_device_tensor(pre["noise_host"], self.noise_buf)
        self._stage_pos(pos)
        self._upload_host_rows(rows)
        ttnn.execute_trace(self.mesh, self.fused_tid, cq_id=0, blocking=False)
        return main, code0, first != main

    def upload_noise(self, pre):
        """Queue the heads' noise of a prepare_frame result now (port7: behind the running backbone step)."""
        ttnn.copy_host_to_device_tensor(pre["noise_host"], self.noise_buf)
        pre["noise_up"] = True

    def _step_fused(self, cand_logits, pos, input_frame, window, gen, temperature=1.0, top_p=0.9, top_k=None, debug=False):
        """step_traced with the codebook draws on the card; same RNG order (the main token's two draws, then
        4096 exponential values per head, of which ranks 0..31 go to the card). debug: also return the heads'
        noise rows (for a host re-draw)."""
        from models.demos.blackhole.drama3.tt.device_sampler import K

        assert (top_p, temperature) == self.fused_params and (top_k or 1 << 30) >= CODEBOOK_TOP_K
        t = self.host
        top_k = top_k or len(t.candidates)
        first = int(t.candidates[sample_top_k_top_p(cand_logits, temperature, top_k, top_p, gen, keep_crossing=True)])
        ras = int(t.candidates[sample_top_k_top_p(cand_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, gen, keep_crossing=True)])
        main = ras if (first in window and t.sem0 <= first < t.sem0 + t.cb_size) else first
        code0 = min(max(main - t.sem0, 0), t.cb_size - 1)
        lab = t.is_semantic(torch.tensor([main]))
        si = t.shared_input(input_frame.reshape(1, -1))
        noise = torch.stack([torch.empty(t.cb_size).exponential_(1, generator=gen) for _ in self.residual])
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(noise[None, None, :, :K].contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), self.noise_buf
        )
        self._stage_pos(pos)
        c0 = torch.tensor([code0])
        rows = self._host_rows(c0, lab, si)
        self._upload_host_rows(rows)
        ttnn.execute_trace(self.mesh, self.fused_tid, cq_id=0, blocking=False)
        codes = ttnn.to_torch(self.fused_codes).float().reshape(-1)[: len(self.residual)].round().long()
        frame = torch.cat([torch.tensor([main, code0]), codes])
        return (frame, first != main, noise) if debug else (frame, first != main)

    # one frame -----------------------------------------------------------------------------------
    def share_pos(self, bb):
        """Use the backbone's position / cos / sin buffers (call before any trace capture)."""
        self.pos_buf, self.cos_buf, self.sin_buf = bb.pos_buf, bb.cos_buf, bb.sin_buf
        self.pos_owner = bb

    def _stage_pos(self, pos):
        if getattr(self, "pos_owner", None) is not None:
            return self.pos_owner.stage_pos(pos)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), self.pos_buf
        )
        c, s = self.cos_sin_host(pos)
        ttnn.copy_host_to_device_tensor(c, self.cos_buf)
        ttnn.copy_host_to_device_tensor(s, self.sin_buf)

    def step(self, *a, **k):
        with self._pf_frame():
            return self._step(*a, **k)

    def _step(
        self, cand_logits, h_dev, pos, input_frame, window, gen, temperature=1.0, top_p=0.9, top_k=None, collect=False,
        force_codes=None,
    ):
        """Same contract and RNG order as CodebookTower.step; h_dev = backbone final-norm row [1, 1, H] (device).
        force_codes (check only): embed these codes 1..9 in the residual layers instead of the sampled ones, so a
        comparison with the host tower stays on the same path; the sampled codes are still returned."""
        t = self.host
        top_k = top_k or len(t.candidates)
        first = int(t.candidates[sample_top_k_top_p(cand_logits, temperature, top_k, top_p, gen, keep_crossing=True)])
        ras = int(t.candidates[sample_top_k_top_p(cand_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, gen, keep_crossing=True)])
        main = ras if (first in window and t.sem0 <= first < t.sem0 + t.cb_size) else first
        code0 = min(max(main - t.sem0, 0), t.cb_size - 1)
        lab = t.is_semantic(torch.tensor([main]))
        si = t.shared_input(input_frame.reshape(1, -1))
        self._stage_pos(pos)
        c0 = torch.tensor([code0])
        h = h_dev
        outs, head_logits = [], []
        for i, layer in enumerate(self.shared):
            ta = self.etable and i > 0
            e = self._up(self._a_rows(i, c0, lab) if ta else self._emb_rows(layer, c0, lab, si if i == 0 else None))
            y = layer(h, e, self.cos_buf, self.sin_buf, pos=self.pos_buf, pre_a=ta)
            ttnn.deallocate(e)
            if h is not h_dev:
                ttnn.deallocate(h)
            h = y
            if collect:
                outs.append(ttnn.to_torch(h).float().reshape(-1)[: self.H])
        codes = [code0]
        k = min(top_k, CODEBOOK_TOP_K)
        for r, layer in enumerate(self.residual):
            prev = codes[-1] if (force_codes is None or r == 0) else int(force_codes[r])
            ci = torch.tensor([prev])
            e = self._up(self._a_rows(len(self.shared) + r, ci, lab) if self.etable else self._emb_rows(layer, ci, lab))
            y = layer(h, e, self.cos_buf, self.sin_buf, pos=self.pos_buf, pre_a=self.etable)
            ttnn.deallocate(e)
            ttnn.deallocate(h)
            h = y
            lg = ttnn.linear(h, self.heads[r], dtype=ttnn.float32, compute_kernel_config=self.ckc)
            logits = ttnn.to_torch(lg).float().reshape(-1)[:RESIDUAL_CODEBOOK_SIZE]
            ttnn.deallocate(lg)
            if collect:
                outs.append(ttnn.to_torch(h).float().reshape(-1)[: self.H])
                head_logits.append(logits.clone())
            # pad back to the head's 4096 outputs (-inf, as the host masks them): same distribution AND the
            # same RNG consumption as CodebookTower.step (its exponential noise is drawn over all 4096)
            full = torch.cat([logits, torch.full((t.cb_size - RESIDUAL_CODEBOOK_SIZE,), float("-inf"))])
            codes.append(sample_top_k_top_p(full, temperature, k, top_p, gen))
        ttnn.deallocate(h)
        frame = torch.tensor([main, *codes], dtype=torch.long)
        return (frame, first != main, outs, head_logits) if collect else (frame, first != main)

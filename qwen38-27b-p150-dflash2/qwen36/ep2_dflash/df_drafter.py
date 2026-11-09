"""DFlash2 drafter (z-lab/Qwen3.8-27B-DFlash2) on a Tenstorrent mesh, replicated on every device (no CCL).

Port of z-lab/dflash `dflash/model.py` (DFlash2DraftModel + CandidateSelector), Episode 2 night 1.
The drafter's numerics only change the acceptance rate, never the output: the target verifies every token.

Layout decisions (see PLAN.md):
- weights BFP8 (env DF_DRAFT_MLP=bfp4 puts MLP gate/up in BFP4), activations bf16, rows padded to tiles by ttnn.
- context K/V ring per layer: [64, 1, S_LANE, 128] = (8 kv heads x 8 lanes) "batch rows". A verify step's
  context rows j = 0..7 go to lane j at ring row `self.row` (one row per step), invalid rows to the trash row
  S_LANE-1; the block's own K/V (8 rows) go to lanes 0..7 at row S_LANE-2 every step. The prompt fills lanes by
  position (lane = p % 8, row = p // 8). The host keeps the position of every (lane, row) and builds the
  sliding-window mask (|q - k| < 2048, bidirectional) per step. SDPA keys = the ring viewed as [1, 8, 8*S_LANE, 128]
  (a pure view: no concat per step).
- LM head: the target's vocab-sharded head; top-16 on the host, selector on the host (codebooks in host RAM).
"""
import json
import math
import os
import time

import torch
from safetensors import safe_open

import ttnn

H = 5120
NH = 32
NKV = 8
HD = 128
NL = 5
GROUPS = 320
GS = 16
BLK = 8  # block = anchor + 7 drafts
NCTX = 8  # context rows per step (<= block)
# ep2 ccl DF_RING_POS=1: context K/V placed by position (lane p % 8, row (p // 8) % CTX_ROWS), so the ring only needs
# the 2,048-token window (+ slack): 288 rows per lane = 2,304 SDPA keys instead of 8,192 (same visible key set)
RING_POS = os.environ.get("DF_RING_POS", "0") == "1"
SHIFT_MM = os.environ.get("DF_SHIFT_MM", "0") == "1"  # ep2 ccl: row shift as one exact one-hot matmul (4 ops -> 1)
RMS_W = os.environ.get("DF_RMS_W", "0") == "1"  # ep2 ccl: weight inside rms_norm (2 ops -> 1)
CONV_BIAS = os.environ.get("DF_CONV_BIAS", "0") == "1"  # ep2 ccl: conv base added as the expansion matmul's bias
CONV_ADDCMUL = os.environ.get("DF_CONV_ADDCMUL", "0") == "1"  # ep2 ccl: x*c0 + prev*c1 as multiply + addcmul
CONV_BF8 = os.environ.get("DF_CONV_BF8", "0") == "1"  # ep2 ccl: conv kernel_projection weights BFP8 (was bf16)
# ep2 ccl DF_RMS_SH=1: the [8, 5120] RMSNorms (hidden_norm, input/post norms, final norm) on the target's width-sharded
# 32-core decode norm config instead of ttnn's one-core interleaved norm (~84 us each, 12 per step: the profile's gaps)
RMS_SH = os.environ.get("DF_RMS_SH", "0") == "1"
BF4_ALL = os.environ.get("DF_DRAFT_BF4ALL", "0") == "1"  # ep2 ccl: q / kv / o / down / fc in BFP4 too (bytes)
# ep2 ccl DF_DRAFT_TP=1 (2 cards): the drafter split across the cards instead of replicated: q / kv heads and the MLP
# columns per card (Megatron), fc output columns + one all-gather, o / down row-parallel + one all-reduce each (the
# target's QWEN36_CCL_AR all-reduce when present). Each card reads half the weight tiles (the matmuls are tile-rate bound).
TP = os.environ.get("DF_DRAFT_TP", "0") == "1"
ARGMAX_TILE = os.environ.get("DF_ARGMAX_TILE", "0") == "1"
MAX_GATHER = os.environ.get("DF_DRAFT_MAX_GATHER", "0") == "1"
TPN = 2 if TP else 1
NH_L = NH // TPN
CONV_TP = TP and os.environ.get("DF_CONV_TP", "0") == "1"  # ep2 ccl: conv kernel_projection column-split + all-gather
CONV_HIFI2 = os.environ.get("DF_CONV_HIFI2", "0") == "1"  # ep2 ccl: conv kernel_projection matmul at HiFi2 (was HiFi4)
# ep2 ccl DF_STAGE_PACK=1: the per-step bf16 inputs (mask, 4 rope tables, context permutation) in ONE host->device copy
# (a [64, KLEN] blob sliced on the device inside the trace) instead of six
STAGE_PACK = os.environ.get("DF_STAGE_PACK", "0") == "1"
NKV_L = NKV // TPN
S_LANE = int(os.environ.get("DF_S_LANE", "288" if RING_POS else "1024"))  # ring rows per lane; row S_LANE-1 = trash, row S_LANE-2 = this step's block
TRASH = S_LANE - 1
BROW = S_LANE - 2
CTX_ROWS = S_LANE - 2  # ring rows usable for context (wrap modulo)
RING = 8 * S_LANE  # 8192 keys into SDPA
KLEN = RING
QPAD = 32
WINDOW = 2048
PF_ROWS = 2080  # prompt context rows processed at prefill (>= window + block + 7 lead), multiple of 32
NEG = -1e9


def _rep(mesh):
    return ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None


def _up(t, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    kw = {}
    if _rep(mesh) is not None:
        kw["mesh_mapper"] = _rep(mesh)
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=mem, **kw)


def _shard(t, mesh, dtype, dim):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim))


def _dev0(t):
    """Host copy of device 0's replica."""
    try:
        return dev_rows(t, t.device())[0]
    except Exception:
        return ttnn.to_torch(t)


def dev_rows(t, mesh):
    """Per-device host copies of a mesh tensor via ConcatMeshToTensor(dim 0) (no get_device_tensors: on 2 cards a
    to_torch of a get_device_tensors() shard is the suspect of the 2026-11-02 03:52 segfault in from_device)."""
    nd = mesh.get_num_devices()
    if nd == 1:
        return [ttnn.to_torch(t)]
    h = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return list(torch.chunk(h, nd, dim=0))


def rope_cos_sin(positions, theta=1e7, dim=HD):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    f = torch.outer(torch.as_tensor(positions, dtype=torch.float64), inv)
    emb = torch.cat([f, f], dim=-1)
    return emb.cos().float(), emb.sin().float()


class HostSelector:
    """CandidateSelector.select for greedy decoding (temperature 0), on the host in fp32."""

    def __init__(self, ckpt):
        with safe_open(os.path.join(ckpt, "model.safetensors"), framework="pt") as f:
            self.pred = f.get_tensor("candidate_selector.predecessor_codebook").float()  # [V, 256]
            self.succ = f.get_tensor("candidate_selector.successor_codebook").float()
            self.hproj = f.get_tensor("candidate_selector.hidden_projection.weight").float()  # [256, 5120]

    def select(self, hidden, top_vals, top_ids, anchor):
        """hidden [n, 5120] (after the drafter's final norm), top_vals/top_ids [n, k]; returns n token ids."""
        hp = hidden.float() @ self.hproj.T  # [n, 256]
        pred = int(anchor)
        out = []
        for p in range(hidden.shape[0]):
            cand = top_ids[p]
            s = top_vals[p].float() + (self.succ[cand] * (self.pred[pred] * hp[p])[None, :]).sum(-1)
            pred = int(cand[int(torch.argmax(s))])
            out.append(pred)
        return out


class TtDrafter:
    def __init__(self, mesh, ckpt, target, log=print):
        self.mesh = mesh
        self.target = target  # Qwen36Model (embedding + vocab-sharded LM head)
        self.log = log
        cfg = json.load(open(os.path.join(ckpt, "config.json")))
        dc = cfg["dflash_config"]
        self.tap_ids = list(dc["target_layer_ids"])
        self.mask_id = int(dc["mask_token_id"])
        self.block = int(dc["block_size"])
        assert self.block == BLK, self.block
        assert cfg["num_hidden_layers"] == NL and cfg["hidden_size"] == H
        self.eps = float(cfg["rms_norm_eps"])
        self.theta = float(cfg["rope_parameters"]["rope_theta"])
        self.window = int(cfg["sliding_window"])
        assert not cfg.get("is_causal", True), "DFlash2 checkpoint is expected to be bidirectional"
        self.selector = HostSelector(ckpt)
        self.top_k = int(dc["selector_top_k"])
        self.hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        t0 = time.time()
        self._load(ckpt)
        self.log(f"[dflash] drafter weights on device in {time.time() - t0:.1f}s")
        self.lm_w = self._lm_weight()
        self._alloc_state()

    # ---------------------------------------------------------------- weights
    def _lm_weight(self):
        """xfer DF_DRAFT_LM_COLS=N (default 0 = all): the drafter's LM head keeps only the first N columns of each
        card's vocab shard (ids [0, N) and [V/2, V/2 + N)); DF_DRAFT_LM_BF4=1: its own weight copy in BFP4. The drafter
        LM head is the largest single drafter program (1.3 ms of 7.4); drafts only change the acceptance, never the
        output. Default = the target's own weight (no extra copy)."""
        w = self.target.lm_head_weight
        nd = self.mesh.get_num_devices()
        self.lm_shard = self.target.vocab_size // nd  # ids of device d's LM head columns: d * lm_shard + column
        cols = int(os.environ.get("DF_DRAFT_LM_COLS", "0"))
        bf4 = os.environ.get("DF_DRAFT_LM_BF4", "0") == "1"
        if os.environ.get("DF_DRAFT_LM_HALF", "0") == "1" and nd == 2:
            # xfer: the two cards split the FIRST HALF of the vocab between them (dev 0 ids [0, V/4), dev 1 [V/4, V/2)):
            # the drafter LM head is width-bound (half the columns per card: 1.32 -> ~0.7 ms), ids >= V/2 are never
            # drafted (0.6 % of prose tokens, none in code); drafts only change the acceptance. Same tiles as the
            # target's columns (a tile-aligned split, BFP4 re-packed unchanged).
            t = ttnn.to_torch(w, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))
            t = t.reshape(int(w.shape[-2]), -1)[:, : self.target.vocab_size // 2]
            new = ttnn.from_torch(t.reshape(1, 1, *t.shape).to(torch.bfloat16), dtype=w.dtype, layout=ttnn.TILE_LAYOUT,
                                  device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                  mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=-1))
            ttnn.synchronize_device(self.mesh)
            self.lm_shard = self.target.vocab_size // 4
            self.log(f"[dflash] drafter LM head: first half of the vocab split over the cards, shape {list(new.shape)} dtype {new.dtype} shard {self.lm_shard}")
            return new
        if not cols and not bf4:
            return w
        if cols:
            assert cols % 32 == 0 and 0 < cols <= int(w.shape[-1]), cols
            w = ttnn.slice(w, (0, 0), (int(w.shape[-2]), cols), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if bf4:
            w = ttnn.typecast(w, ttnn.bfloat4_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(self.mesh)
        self.log(f"[dflash] drafter LM head weight: shape {list(w.shape)} dtype {w.dtype} (DF_DRAFT_LM_COLS={cols} BF4={bf4})")
        return w

    def _load(self, ckpt):
        wdt = ttnn.bfloat4_b if BF4_ALL else ttnn.bfloat8_b
        mlp_dt = ttnn.bfloat4_b if os.environ.get("DF_DRAFT_MLP", "bfp8") == "bfp4" else ttnn.bfloat8_b
        m = self.mesh
        f = safe_open(os.path.join(ckpt, "model.safetensors"), framework="pt")
        g = f.get_tensor

        def lin(name, dtype=wdt, shard=None):  # HF [out, in] -> ttnn [in, out]; shard = mesh dim under DF_DRAFT_TP
            t = g(name).T.contiguous()
            return _shard(t, m, dtype, shard) if (TP and shard is not None) else _up(t, m, dtype=dtype)

        if TP:
            assert m.get_num_devices() == TPN, "DF_DRAFT_TP needs 2 devices"

        def vec(name, shape=None):
            t = g(name).to(torch.bfloat16)
            return _up(t.reshape(shape or (1, 1, 1, t.numel())), m)

        self.fc = lin("fc.weight", shard=-1)  # [25600, 5120] (TP: output columns per card)
        self.hidden_norm = vec("hidden_norm.weight")
        self.norm = vec("norm.weight")
        E = torch.zeros(GROUPS, H, dtype=torch.bfloat16)
        for gi in range(GROUPS):
            E[gi, gi * GS : (gi + 1) * GS] = 1
        self.E = _up(E, m)  # one-hot group expansion, exact under fp32 accumulation
        self.layers = []
        for i in range(NL):
            p = f"layers.{i}."
            L = {}
            L["in_norm"] = vec(p + "input_layernorm.weight")
            L["post_norm"] = vec(p + "post_attention_layernorm.weight")
            L["q"] = lin(p + "self_attn.q_proj.weight", shard=-1)  # TP: q heads 0-15 | 16-31
            kv = torch.cat([g(p + "self_attn.k_proj.weight"), g(p + "self_attn.v_proj.weight")], dim=0)
            if TP:  # per card [k heads | v heads] of its NKV_L kv heads (GQA: q heads 16c..16c+15 use kv 4c..4c+3)
                kT = kv[: NKV * HD].T.reshape(H, NKV, HD)
                vT = kv[NKV * HD :].T.reshape(H, NKV, HD)
                parts = [torch.cat([kT[:, c * NKV_L : (c + 1) * NKV_L].reshape(H, -1), vT[:, c * NKV_L : (c + 1) * NKV_L].reshape(H, -1)], dim=-1) for c in range(TPN)]
                L["kv"] = _shard(torch.cat(parts, dim=-1).contiguous(), m, wdt, -1)
            else:
                L["kv"] = _up(kv.T.contiguous(), m, dtype=wdt)  # [5120, 2048] = [k | v]
            L["o"] = lin(p + "self_attn.o_proj.weight", shard=-2)  # TP: rows of the card's heads
            L["q_norm"] = vec(p + "self_attn.q_norm.weight")
            L["k_norm"] = vec(p + "self_attn.k_norm.weight")
            L["gate"] = lin(p + "mlp.gate_proj.weight", mlp_dt, shard=-1)
            L["up"] = lin(p + "mlp.up_proj.weight", mlp_dt, shard=-1)
            L["down"] = lin(p + "mlp.down_proj.weight", shard=-2)
            for c in ("attention_conv", "mlp_conv"):
                base = g(p + c + ".base_kernel").to(torch.bfloat16)  # [2 (prepare/finish), 2 (offset), 5120]
                L[c + ".base"] = [[_up(base[a, o].reshape(1, 1, 1, H), m) for o in range(2)] for a in range(2)]
                L[c + ".proj"] = lin(p + c + ".kernel_projection.weight", ttnn.bfloat8_b if CONV_BF8 else ttnn.bfloat16,
                                     shard=-1 if CONV_TP else None)  # [5120, 1280]
            self.layers.append(L)

    def _alloc_state(self):
        m = self.mesh
        z = torch.zeros(NKV_L * 8, 1, S_LANE, HD, dtype=torch.bfloat16)
        self.ring_k = [_up(z, m) for _ in range(NL)]
        self.ring_v = [_up(z, m) for _ in range(NL)]
        self.zrow = _up(torch.zeros(1, 1, 1, H, dtype=torch.bfloat16), m, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.zero_taps = _up(torch.zeros(1, 1, NCTX, len(self.tap_ids) * H, dtype=torch.bfloat16), m)
        self._alloc_inputs()
        self.kv_upd_cfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, HD),
            core_grid=ttnn.CoreGrid(x=8, y=NKV_L),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        grid = m.compute_with_storage_grid_size()
        self.sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            q_chunk_size=32,
            k_chunk_size=int(os.environ.get("DF_DRAFT_KCHUNK", "256")),
            exp_approx_mode=False,
        )
        self.reset()

    def reset(self):
        """New request: forget the ring (host bookkeeping only; stale rows are masked)."""
        self.pos = torch.full((8, S_LANE), -1, dtype=torch.long)  # position held by (lane, row)
        self.row = 0  # next ring row for per-step context
        self.pending_ctx = None  # device taps [1,1,8,25600] + n valid + positions, consumed by the next step

    # ---------------------------------------------------------------- small ops
    def _rms(self, x, w):
        if RMS_SH and x.shape[-1] == H and x.shape[-2] <= 32:
            # r64g: the framework's DECODE attn norm config is sized from max_batch_size, so on the 64-row
            # multi-stream model it is two tile rows tall and interleaved_to_sharded refuses it ("Shard height 64
            # must match physical height 32"). The drafter is always <= 32 rows: take the width-correct config
            # (decode_norm_cfg at one tile row is the framework's own config, byte-identical to the 32-row build).
            _rows = int(x.shape[-2])
            _ncs = self.__dict__.setdefault("_rms_ncs", {})
            nc = _ncs.get(_rows)
            if nc is None:
                nc = _ncs[_rows] = self.target.args.decode_norm_cfg(_rows, "attn")
            xs = ttnn.to_memory_config(x, nc["sharded_output_config"])
            y = ttnn.rms_norm(
                xs,
                epsilon=self.eps,
                program_config=nc["sharded_program_config"],
                memory_config=nc["sharded_output_config"],
                compute_kernel_config=self.hifi4,
            )
            ttnn.deallocate(xs)
            yi = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(y)
            out = ttnn.multiply(yi, w)
            ttnn.deallocate(yi)
            return out
        if RMS_W:
            return ttnn.rms_norm(x, epsilon=self.eps, weight=w, compute_kernel_config=self.hifi4)
        return ttnn.multiply(ttnn.rms_norm(x, epsilon=self.eps, compute_kernel_config=self.hifi4), w)

    def _pc1d(self, x, w, act=None, key=None):
        """ep2 t3b DF_DRAFT_1D=<cores>[,name=cores..]: explicit 1D mcast decode program config for a drafter matmul
        (interleaved weight, M = 1 tile); None = ttnn's own choice."""
        spec = os.environ.get("DF_DRAFT_1D", "")
        if not spec or x.is_sharded() or int(x.padded_shape[-2]) > ttnn.TILE_SIZE:
            return None
        parts = spec.split(",")
        nc = int(parts[0]) if parts[0].isdigit() else 0
        for kv in parts:
            if "=" in kv and key is not None and kv.split("=")[0] == key:
                nc = int(kv.split("=")[1])
        if nc <= 0:
            return None
        K, N = int(w.shape[-2]), int(w.shape[-1])
        ck = (K, N, nc, act)
        pcs = self.__dict__.setdefault("_pcs", {})
        if ck not in pcs:
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            gw = self.mesh.compute_with_storage_grid_size().x
            pcs[ck] = tpc.create_matmul_1d_decode_progcfg(ttnn.TILE_SIZE, K, N, num_cores=nc, grid_w=gw,
                                                         **({"fused_activation": act} if act is not None else {}))
        return pcs[ck]

    def _lin(self, x, w, key=None):
        pc = self._pc1d(x, w, key=key)
        if pc is not None:
            return ttnn.linear(x, w, compute_kernel_config=self.hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               program_config=pc)
        return ttnn.linear(x, w, compute_kernel_config=self.hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _shift(self, x):
        """Rows t -> t+1 (row 0 = 0) along dim 2 of [1,1,8,H]."""
        if SHIFT_MM:
            return ttnn.matmul(self.shift_m, x, compute_kernel_config=self.hifi4)
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        body = ttnn.slice(rm, (0, 0, 0, 0), (1, 1, BLK - 1, H))
        cat = ttnn.concat([self.zrow, body], dim=2)
        out = ttnn.to_layout(cat, ttnn.TILE_LAYOUT)
        for t in (rm, body, cat):
            ttnn.deallocate(t)
        return out

    def _conv_dyn(self, L, c, x):
        """kernel_projection(x) -> 4 expanded [1,1,8,H] coefficient maps: d[a][o]."""
        _pc = self._pc1d(x, L[c + ".proj"], key="conv")
        proj = ttnn.linear(x, L[c + ".proj"], compute_kernel_config=self.hifi2 if CONV_HIFI2 else self.hifi4,
                           **({"program_config": _pc} if _pc is not None else {}))  # [1,1,8,1280]
        if CONV_TP:
            proj = self._gather(proj)
        d = [[None, None], [None, None]]
        for a in range(2):
            for o in range(2):
                lo = a * 640 + o * 320
                part = ttnn.slice(proj, (0, 0, 0, lo), (1, 1, BLK, lo + GROUPS))
                if CONV_BIAS:
                    d[a][o] = ttnn.linear(part, self.E, bias=L[c + ".base"][a][o], compute_kernel_config=self.hifi4)
                    ttnn.deallocate(part)
                    continue
                e = ttnn.matmul(part, self.E, compute_kernel_config=self.hifi4)
                d[a][o] = ttnn.add(e, L[c + ".base"][a][o])
                ttnn.deallocate(part)
                ttnn.deallocate(e)
        ttnn.deallocate(proj)
        return d

    def _conv_apply(self, x, coef):
        prev = self._shift(x)
        if CONV_ADDCMUL:
            xm = ttnn.multiply(x, coef[0])
            out = ttnn.addcmul(xm, prev, coef[1], value=1.0)
            ttnn.deallocate(xm)
            ttnn.deallocate(prev)
            return out
        out = ttnn.add(ttnn.multiply(x, coef[0]), ttnn.multiply(prev, coef[1]))
        ttnn.deallocate(prev)
        return out

    def _heads(self, x, n_rows, n_heads):
        """[1,1,n,heads*128] -> [1,heads,n,128]."""
        r = ttnn.reshape(x, (1, n_rows, n_heads, HD))
        return ttnn.transpose(r, 1, 2)

    def _rope(self, x, cos, sin):
        return ttnn.experimental.rotary_embedding_hf(x, cos, sin, is_decode_mode=False)

    # ---------------------------------------------------------------- context
    def _gather(self, x):
        """DF_DRAFT_TP: column-parallel output [1,1,n,N/2] per card -> [1,1,n,N] on both."""
        from models.tt_transformers.tt.ccl import tt_all_gather

        return tt_all_gather(x, self.mesh, self.target.tt_ccl, cluster_axis=None, dim=3, topology=ttnn.Topology.Linear)

    def _allreduce(self, x):
        """DF_DRAFT_TP: row-parallel partial [1,1,8,5120] -> the sum on both cards (DRAM interleaved)."""
        ar = getattr(self.target, "_ccl_ar", None)
        if ar is not None:  # the target's one-shot all_reduce_async (QWEN36_CCL_AR; never runs concurrently with it)
            y = ar(x, "draft")
            out = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(y)
            return out
        ccl = self.target.tt_ccl
        shape = list(x.shape)
        g = ttnn.experimental.all_gather_async(
            x, persistent_output_buffer=None, dim=1, multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=ccl.get_num_links(1), topology=ttnn.Topology.Linear, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(), chunks_per_sync=10, num_workers_per_link=2,
            num_buffers_per_channel=2)
        ttnn.deallocate(x)
        B, N = shape[-2], shape[-1]
        g0 = ttnn.slice(g, [0, 0, 0, 0], [1, 1, B, N])
        g1 = ttnn.slice(g, [0, 1, 0, 0], [1, 2, B, N])
        ttnn.deallocate(g)
        out = ttnn.add(g0, g1)
        ttnn.deallocate(g0)
        ttnn.deallocate(g1)
        return out

    def ctx_hidden(self, taps):
        """taps: device [1,1,n,25600] replicated -> hidden_norm(fc(taps)) [1,1,n,5120]."""
        h = self._lin(taps, self.fc, key="fc")
        if TP:
            h = self._gather(h)
        out = self._rms(h, self.hidden_norm)
        ttnn.deallocate(h)
        return out

    def _ctx_kv(self, L, ctx_h, n, cos, sin):
        kv = self._lin(ctx_h, L["kv"], key="kv")  # [1,1,n,2048]
        k = ttnn.slice(kv, (0, 0, 0, 0), (1, 1, n, NKV_L * HD))
        v = ttnn.slice(kv, (0, 0, 0, NKV_L * HD), (1, 1, n, 2 * NKV_L * HD))
        ttnn.deallocate(kv)
        kh = self._heads(k, n, NKV_L)
        k = self._rms(kh, L["k_norm"])
        k = self._rope(k, cos, sin)
        vh = self._heads(v, n, NKV_L)
        return k, vh  # [1,8,n,128] each

    def prefill_context(self, taps_host, positions):
        """Prompt context: taps_host [n, 25600] (host bf16) for consecutive positions[0..n), n <= PF_ROWS.

        Fills every layer's ring by position (lane = p % 8). Called once per request after the prompt."""
        _tc = time.perf_counter()
        self.reset()
        prebuilt = isinstance(taps_host, tuple)  # ("full", [PF_ROWS, 25600] buffer, n): df_server DF_TAPS_FULL
        on_dev = prebuilt and taps_host[0] == "dev"  # ("dev", [1, 1, PF_ROWS, 25600] device tensor, n): DF_TAPS_DEV
        n = taps_host[2] if prebuilt else taps_host.shape[0]
        assert n <= PF_ROWS and n == len(positions)
        p0 = int(positions[0]) - int(positions[0]) % 8  # align the first row to lane 0
        lead = int(positions[0]) - p0
        rows = PF_ROWS
        if prebuilt:
            full = taps_host[1]
        else:
            full = torch.zeros(rows, taps_host.shape[1], dtype=torch.bfloat16)
            full[lead : lead + n] = taps_host
        pos = torch.arange(p0, p0 + rows)
        _t0 = time.perf_counter()
        taps = full if on_dev else _up(full.reshape(1, 1, rows, -1), self.mesh)
        ttnn.synchronize_device(self.mesh) if os.environ.get("DF_CTX_TIMING") == "1" else None
        _t1 = time.perf_counter()
        ctx_h = self.ctx_hidden(taps)
        ttnn.deallocate(taps)
        ttnn.synchronize_device(self.mesh) if os.environ.get("DF_CTX_TIMING") == "1" else None
        _t2 = time.perf_counter()
        c, s = rope_cos_sin(pos, self.theta)
        cos = _up(c.to(torch.bfloat16).reshape(1, 1, rows, HD), self.mesh)
        sin = _up(s.to(torch.bfloat16).reshape(1, 1, rows, HD), self.mesh)
        nr = rows // 8  # ring rows used (260 < CTX_ROWS)
        assert nr <= CTX_ROWS
        rowidx = (torch.arange(nr) + p0 // 8) % CTX_ROWS if RING_POS else torch.arange(nr)
        ctx_dev = os.environ.get("DF_CTX_DEV", "0") == "1"
        if ctx_dev:
            # ep2 t3b DF_CTX_DEV=1: the ring rows built on the card (no host round trip per layer): ring row
            # (h * 8 + lane, rowidx[r]) = source row h * rows + r * 8 + lane of the [NKV_L * rows, 128] row-major view of
            # the context K / V, every other ring row = the zero row appended at index NKV_L * rows. One exact row gather
            # (ttnn.embedding) per tensor, fixed shapes (the indices are data).
            zr = NKV_L * rows
            idx = torch.full((NKV_L, 8, S_LANE), zr, dtype=torch.int64)
            r = torch.arange(nr)
            for h in range(NKV_L):
                for lane in range(8):
                    idx[h, lane, rowidx] = h * rows + r * 8 + lane
            self._h2d_ctx_idx(idx.reshape(1, -1))
        for li, L in enumerate(self.layers):
            k, v = self._ctx_kv(L, ctx_h, rows, cos, sin)
            for src, ring in ((k, self.ring_k[li]), (v, self.ring_v[li])):
                if ctx_dev:
                    w = ttnn.to_layout(src, ttnn.ROW_MAJOR_LAYOUT)
                    w = ttnn.reshape(w, (NKV_L * rows, HD))
                    wz = ttnn.concat([w, self.ctx_zrow], dim=0)
                    ttnn.deallocate(w)
                    g = ttnn.embedding(self.ctx_idx, wz, layout=ttnn.TILE_LAYOUT)
                    ttnn.deallocate(wz)
                    g = ttnn.reshape(g, (NKV_L * 8, 1, S_LANE, HD))
                    ttnn.copy(g, ring)
                    ttnn.deallocate(g)
                    ttnn.deallocate(src)
                    continue
                bufs = []
                for hd in (dev_rows(src, self.mesh) if TP else [_dev0(src)]):
                    hst = hd.to(torch.bfloat16).reshape(NKV_L, nr, 8, HD)  # [h, r, lane, d]
                    lanes = hst.permute(0, 2, 1, 3).contiguous()  # [h, lane, r, d]
                    buf = torch.zeros(NKV_L, 8, S_LANE, HD, dtype=torch.bfloat16)
                    buf[:, :, rowidx] = lanes
                    bufs.append(buf.reshape(NKV_L * 8, 1, S_LANE, HD))
                if TP:
                    t = _shard(torch.cat(bufs, dim=0), self.mesh, ttnn.bfloat16, 0)
                else:
                    t = _up(bufs[0], self.mesh)
                ttnn.copy(t, ring)
                ttnn.deallocate(t)
                ttnn.deallocate(src)
        for t in (ctx_h, cos, sin):
            ttnn.deallocate(t)
        _t3 = time.perf_counter()
        self.ctx_timing = (round(1e3 * (_t1 - _t0), 1), round(1e3 * (_t2 - _t1), 1), round(1e3 * (_t3 - _t2), 1),
                           round(1e3 * (_t0 - _tc), 1))
        valid = torch.zeros(rows, dtype=torch.bool)
        valid[lead : lead + n] = True
        pv = torch.where(valid, pos, torch.full_like(pos, -1))
        self.pos[:, rowidx] = pv.reshape(nr, 8).T
        self.row = nr

    def _h2d_ctx_idx(self, idx):
        """DF_CTX_DEV: the persistent gather indices [1, NKV_L * 8 * S_LANE] (uint32) and the zero row (once)."""
        h = ttnn.from_torch(idx.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        if getattr(self, "ctx_idx", None) is None:
            self.ctx_idx = _up(idx.to(torch.int32), self.mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            self.ctx_zrow = _up(torch.zeros(1, HD, dtype=torch.bfloat16), self.mesh, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            ttnn.copy_host_to_device_tensor(h, self.ctx_idx)

    def set_step_context(self, taps_dev, n, first_pos):
        """Context of the next step: device taps [1,1,8,25600] (rows 0..n-1 valid), positions first_pos..+n."""
        self.pending_ctx = (taps_dev, int(n), int(first_pos))

    # ---------------------------------------------------------------- one draft step
    def _alloc_inputs(self):
        """Persistent device inputs of a draft step (allocated before any trace; refreshed by copies)."""
        m = self.mesh
        self.in_ctx = _up(torch.zeros(1, 1, NCTX, len(self.tap_ids) * H, dtype=torch.bfloat16), m)
        self.in_tok = _up(torch.zeros(BLK, dtype=torch.int32), m, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.in_idx = _up(torch.zeros(NKV_L * 8, dtype=torch.int32), m, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.blk_idx = _up(torch.full((NKV_L * 8,), BROW, dtype=torch.int32), m, dtype=ttnn.int32,
                           layout=ttnn.ROW_MAJOR_LAYOUT)
        self.in_mask = _up(torch.zeros(1, 1, QPAD, KLEN, dtype=torch.bfloat16), m)
        self.in_rope = [_up(torch.zeros(1, 1, BLK, HD, dtype=torch.bfloat16), m) for _ in range(4)]  # cb sb cc sc
        # DF_RING_POS: context row j -> lane (p_ctx + j) % 8 as a one-hot permutation (exact matmul on the context hidden)
        self.in_perm = _up(torch.eye(NCTX, dtype=torch.bfloat16).reshape(1, 1, NCTX, NCTX), m)
        self.in_blob = _up(torch.zeros(1, 1, 2 * QPAD, KLEN, dtype=torch.bfloat16), m) if STAGE_PACK else None
        # t3c DF_PREUP=1: device buffers for the 8 prestaged (idx, blob) variants (DF_PRESTAGE), uploaded while the verify
        # runs; after the read back the chosen one is a device copy
        self.preup = os.environ.get("DF_PREUP", "0") == "1" and STAGE_PACK
        if self.preup:
            self.pre_idx = [_up(torch.zeros(NKV_L * 8, dtype=torch.int32), m, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                            for _ in range(BLK)]
            self.pre_blob = [_up(torch.zeros(1, 1, 2 * QPAD, KLEN, dtype=torch.bfloat16), m) for _ in range(BLK)]
        sh = torch.zeros(BLK, BLK, dtype=torch.bfloat16)
        for t in range(1, BLK):
            sh[t, t - 1] = 1
        self.shift_m = _up(sh.reshape(1, 1, BLK, BLK), m)  # DF_SHIFT_MM: prev = S @ x
        self.trace_id = None
        self.outs = None
        self.topk_dev = os.environ.get("DF_DRAFT_TOPK", "device") == "device"
        self.greedy = os.environ.get("DF_DRAFT_GREEDY", "1") == "1"  # argmax drafts (v2): faster than top-k + selector

    @staticmethod
    def _h2d(t, dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dtype, layout=layout), dev)

    def _stage(self, anchor, start):
        """Host bookkeeping for one step + copies into the persistent inputs."""
        if self.pending_ctx is None:  # first step after the prompt: its context is already in the ring
            taps, n, p_ctx = self.zero_taps, 0, max(0, start - NCTX)
        else:
            taps, n, p_ctx = self.pending_ctx
        self.pending_ctx = None
        ttnn.copy(taps, self.in_ctx)
        # ring bookkeeping: this step's context rows go to lane j, ring row `wr`; the block to row BROW
        idx = torch.full((NKV_L * 8,), TRASH, dtype=torch.int32)
        if RING_POS:
            # lane l takes the context row j with (p_ctx + j) % 8 == l, at ring row (p // 8) % CTX_ROWS
            perm = torch.zeros(NCTX, NCTX, dtype=torch.bfloat16)
            cpos = []
            for lane in range(NCTX):
                j = (lane - p_ctx) % 8
                pp = p_ctx + j
                perm[lane, j] = 1
                cpos.append(pp)
                if j < n:
                    r = (pp // 8) % CTX_ROWS
                    self.pos[lane, r] = pp
                    idx[torch.arange(NKV_L) * 8 + lane] = r
            if not STAGE_PACK:
                self._h2d(perm.reshape(1, 1, NCTX, NCTX), self.in_perm)
        else:
            perm = torch.eye(NCTX, dtype=torch.bfloat16)
            wr = self.row % CTX_ROWS
            self.row += 1
            self.pos[:, wr] = -1
            for j in range(n):
                self.pos[j, wr] = p_ctx + j
                idx[torch.arange(NKV_L) * 8 + j] = wr
            cpos = list(range(p_ctx, p_ctx + NCTX))
        self.pos[:, BROW] = torch.arange(start, start + BLK)
        self._h2d(idx, self.in_idx, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        # mask: slots visible if within the window of each query (block slots always are)
        qpos = torch.arange(start, start + BLK)
        kpos = self.pos.reshape(-1)  # lane-major = SDPA key order
        vis = (kpos[None, :] >= 0) & ((qpos[:, None] - kpos[None, :]).abs() < self.window)
        mask = torch.full((QPAD, KLEN), NEG, dtype=torch.float32)
        mask[:BLK][vis] = 0.0
        mask[BLK:, BROW] = 0.0  # padding queries see block row 0 (no all-masked rows)
        cb, sb = rope_cos_sin(qpos, self.theta)
        cc, sc = rope_cos_sin(torch.tensor(cpos), self.theta)
        if STAGE_PACK:
            blob = torch.zeros(2 * QPAD, KLEN, dtype=torch.float32)
            blob[:QPAD] = mask
            for i, t in enumerate((cb, sb, cc, sc)):
                blob[QPAD : QPAD + BLK, i * HD : (i + 1) * HD] = t
            blob[QPAD : QPAD + NCTX, 4 * HD : 4 * HD + NCTX] = perm.float()
            self._h2d(blob.to(torch.bfloat16).reshape(1, 1, 2 * QPAD, KLEN), self.in_blob)
        else:
            self._h2d(mask.to(torch.bfloat16).reshape(1, 1, QPAD, KLEN), self.in_mask)
            for t, dev in zip((cb, sb, cc, sc), self.in_rope):
                self._h2d(t.to(torch.bfloat16).reshape(1, 1, BLK, HD), dev)
        self._h2d(torch.tensor([int(anchor)] + [self.mask_id] * (BLK - 1), dtype=torch.int32), self.in_tok,
                  ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    # ---------------------------------------------------------------- t3b DF_PRESTAGE: host staging during the verify
    def _parts(self, start, n, p_ctx, pos):
        """The host inputs of a draft step at `start` whose context rows are n rows from p_ctx (RING_POS +
        STAGE_PACK only); updates `pos` (a copy of self.pos) like _stage. Returns host ttnn tensors (idx, blob)."""
        idx = torch.full((NKV_L * 8,), TRASH, dtype=torch.int32)
        perm = torch.zeros(NCTX, NCTX, dtype=torch.bfloat16)
        cpos = []
        for lane in range(NCTX):
            j = (lane - p_ctx) % 8
            pp = p_ctx + j
            perm[lane, j] = 1
            cpos.append(pp)
            if j < n:
                r = (pp // 8) % CTX_ROWS
                pos[lane, r] = pp
                idx[torch.arange(NKV_L) * 8 + lane] = r
        pos[:, BROW] = torch.arange(start, start + BLK)
        qpos = torch.arange(start, start + BLK)
        kpos = pos.reshape(-1)
        vis = (kpos[None, :] >= 0) & ((qpos[:, None] - kpos[None, :]).abs() < self.window)
        mask = torch.full((QPAD, KLEN), NEG, dtype=torch.float32)
        mask[:BLK][vis] = 0.0
        mask[BLK:, BROW] = 0.0
        cb, sb = rope_cos_sin(qpos, self.theta)
        cc, sc = rope_cos_sin(torch.tensor(cpos), self.theta)
        blob = torch.zeros(2 * QPAD, KLEN, dtype=torch.float32)
        blob[:QPAD] = mask
        for i, t in enumerate((cb, sb, cc, sc)):
            blob[QPAD : QPAD + BLK, i * HD : (i + 1) * HD] = t
        blob[QPAD : QPAD + NCTX, 4 * HD : 4 * HD + NCTX] = perm.float()
        return (ttnn.from_torch(idx, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                ttnn.from_torch(blob.to(torch.bfloat16).reshape(1, 1, 2 * QPAD, KLEN), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT))

    def prepare_all(self, start, taps):
        """While the verify of the block at `start` runs: the next step's host inputs for every accepted count a
        (next start = start + a + 1, context = the verify taps, a + 1 rows from start)."""
        if not (RING_POS and STAGE_PACK):
            return
        self.pre = {}
        for a in range(BLK):
            pos = self.pos.clone()
            self.pre[a] = (taps, pos, self._parts(start + a + 1, a + 1, start, pos))
            if getattr(self, "preup", False):
                ttnn.copy_host_to_device_tensor(self.pre[a][2][0], self.pre_idx[a])
                ttnn.copy_host_to_device_tensor(self.pre[a][2][1], self.pre_blob[a])

    def draft_async_pre(self, a, anchor):
        """DF_PRESTAGE: the next draft with the inputs prepared for accepted slot a (only device copies left)."""
        taps, pos, (idx_h, blob_h) = self.pre.pop(a)
        self.pre = None
        self.pending_ctx = None
        self.pos = pos
        ttnn.copy(taps, self.in_ctx)
        if getattr(self, "preup", False):
            ttnn.copy(self.pre_idx[a], self.in_idx)
            ttnn.copy(self.pre_blob[a], self.in_blob)
        else:
            ttnn.copy_host_to_device_tensor(idx_h, self.in_idx)
            ttnn.copy_host_to_device_tensor(blob_h, self.in_blob)
        self._h2d(torch.tensor([int(anchor)] + [self.mask_id] * (BLK - 1), dtype=torch.int32), self.in_tok,
                  ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)

    def _graph(self):
        """The device part of a draft step: persistent inputs -> (final hidden, top-k values/ids or RM logits)."""
        cos_b, sin_b, cos_c, sin_c = self.in_rope
        mask_t, perm_t = self.in_mask, self.in_perm
        if STAGE_PACK:
            b = self.in_blob
            mask_t = ttnn.slice(b, (0, 0, 0, 0), (1, 1, QPAD, KLEN))
            cos_b, sin_b, cos_c, sin_c = [ttnn.slice(b, (0, 0, QPAD, i * HD), (1, 1, QPAD + BLK, (i + 1) * HD)) for i in range(4)]
            perm_t = ttnn.slice(b, (0, 0, QPAD, 4 * HD), (1, 1, QPAD + NCTX, 4 * HD + NCTX))
        x = self._embed_dev(self.in_tok)
        ctx_h = self.ctx_hidden(self.in_ctx)
        if RING_POS:
            ctx_p = ttnn.matmul(perm_t, ctx_h, compute_kernel_config=self.hifi4)
            ttnn.deallocate(ctx_h)
            ctx_h = ctx_p
        for li, L in enumerate(self.layers):
            x = self._layer(li, L, x, ctx_h, self.in_idx, mask_t, cos_b, sin_b, cos_c, sin_c)
        ttnn.deallocate(ctx_h)
        if STAGE_PACK:
            for t in (mask_t, cos_b, sin_b, cos_c, sin_c, perm_t):
                ttnn.deallocate(t)
        hn = self._rms(x, self.norm)
        ttnn.deallocate(x)
        if getattr(self, "greedy", False):
            # DF_DRAFT_GREEDY: plain argmax drafts (no candidate selector): argmax + max per vocab shard, the host
            # picks the shard -- no top-k (the top-k over two 62,080-wide halves costs ~20 ms per step)
            _pc = self._pc1d(hn, self.lm_w, key="lm")
            logits = ttnn.linear(hn, self.lm_w, compute_kernel_config=self.hifi2,
                                 **({"program_config": _pc} if _pc is not None else {}))
            if ARGMAX_TILE:  # ep2 ccl DF_ARGMAX_TILE=1: argmax straight on the tile layout
                idx = ttnn.argmax(logits, dim=-1)
            else:
                rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
                idx = ttnn.argmax(rm, dim=-1)
            if MAX_GATHER and not ARGMAX_TILE:
                # ep2 ccl DF_DRAFT_MAX_GATHER=1: the shard max = the logit AT the argmax index (the devloop verify trick):
                # ttnn.max over the 124,160-wide shard was 1.62 ms per step, the gather is tiny (bit-identical value)
                gi = ttnn.reshape(idx, list(idx.shape) + [1]) if len(idx.shape) < len(rm.shape) else idx
                mx = ttnn.gather(rm, -1, gi)
            else:
                mx = ttnn.max(logits, dim=-1, keepdim=True)
            if getattr(self, "chain_dst", None) is not None:
                # ep2 t3b DF_CHAIN: [anchor, drafts] combined on the card straight into the verify's token input
                import df_chain

                blk = df_chain.combine_dev(self.mesh, self.target.tt_ccl, idx, mx, self.in_tok,
                                           self.lm_shard, self.chain_row0)
                ttnn.copy(blk, self.chain_dst)
                ttnn.deallocate(blk)
            return hn, idx, mx
        if not self.topk_dev:
            logits = ttnn.linear(hn, self.target.lm_head_weight, compute_kernel_config=self.hifi2)
            rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(logits)
            return hn, rm
        # device top-k: rows padded to 32 (topk needs N*C*H % 32 == 0), vocab shard split in halves < 65536
        hp = ttnn.pad(hn, [1, 1, QPAD, H], [0, 0, 0, 0], 0.0)
        logits = ttnn.linear(hp, self.target.lm_head_weight, compute_kernel_config=self.hifi2)
        # no deallocate(hp): ttnn.pad to the tile-padded height can return hn itself (03:52 + 04:05 segfaults: hn freed)
        half = logits.shape[-1] // 2
        res = []
        for lo in (0, half):
            part = ttnn.slice(logits, (0, 0, 0, lo), (1, 1, QPAD, lo + half))
            v, i = ttnn.topk(part, self.top_k, dim=-1, largest=True, sorted=True)
            res += [v, i]
            ttnn.deallocate(part)
        ttnn.deallocate(logits)
        return (hn, *res)

    def capture_trace(self, anchor, start):
        """Trace the device part (every program must have run eagerly first)."""
        self._stage(anchor, start)
        # ep2 ccl: some switches (DF_RMS_SH, the TP all-reduce) put drafter intermediates in L1. If the QWEN36_PF global
        # CB is freed right now (decode mode, after a prefill), those trace intermediates could land in its range and
        # the replays (between verify frames, GCB present) would overwrite it. Hold the GCB range (+64 KB for the
        # prefetcher own L1 just below it) with a placeholder during the capture, so they land below it.
        hold = None
        try:
            from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

            for pf in _wp._by_mesh.values():
                if pf.started and pf.gcb is None and not pf.broken:
                    hold = pf._hold_alloc(64 * 1024)
        except Exception as e:  # noqa: BLE001
            self.log(f"[dflash] drafter capture: no GCB placeholder ({e!r})")
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self.outs = self._graph()
        finally:
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        if hold is not None:
            ttnn.deallocate(hold)
        self.trace_id = tid
        self.log("[dflash] drafter trace captured")

    def release_trace(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None

    def _rows(self, t, d):
        nd = self.mesh.get_num_devices()
        return dev_rows(t, self.mesh)[d]

    def link_verify(self, dst):
        """ep2 t3b DF_CHAIN: every later drafter graph writes [anchor, drafts] into dst (the verify token input)."""
        self.chain_dst = dst
        self.chain_row0 = _up(torch.tensor([1.0] + [0.0] * (BLK - 1)).reshape(1, 1, BLK, 1), self.mesh, dtype=ttnn.float32)

    def draft_async(self, anchor, start):
        """ep2 t3b DF_CHAIN: stage + replay the drafter trace, no read back (the verify reads the block)."""
        assert self.trace_id is not None and getattr(self, "chain_dst", None) is not None
        self._stage(anchor, start)
        ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)

    def draft(self, anchor, start):
        """Propose 7 tokens after `anchor` (the last committed token, at position `start`)."""
        m = self.mesh
        nd = m.get_num_devices()
        _ts0 = time.perf_counter()
        self._stage(anchor, start)
        _ts1 = time.perf_counter()
        if self.trace_id is not None:
            ttnn.execute_trace(m, self.trace_id, cq_id=0, blocking=False)
            outs = self.outs
        else:
            if self.outs is not None:
                for t in self.outs:
                    ttnn.deallocate(t)
            outs = self.outs = self._graph()
        V = self.target.vocab_size
        if getattr(self, "greedy", False):
            nd = self.mesh.get_num_devices()
            ii = dev_rows(outs[1], self.mesh)
            vv = dev_rows(outs[2], self.mesh)
            shard = self.lm_shard
            ids = ii[0].reshape(-1)[:BLK].to(torch.int64) & 0xFFFFFFFF
            best = vv[0].float().reshape(-1)[:BLK]
            for d in range(1, nd):
                cand = vv[d].float().reshape(-1)[:BLK]
                better = cand > best
                ids = torch.where(better, (ii[d].reshape(-1)[:BLK].to(torch.int64) & 0xFFFFFFFF) + d * shard, ids)
                best = torch.where(better, cand, best)
            _ts2 = time.perf_counter()
            st = self.__dict__.setdefault("step_t", [0, 0.0, 0.0, 0.0])
            st[0] += 1; st[1] += _ts1 - _ts0; st[2] += _ts2 - _ts1
            return [int(t) for t in ids[1:BLK].tolist()]
        hid = self._rows(outs[0], 0).reshape(-1, H)[1:BLK].float()
        if not self.topk_dev:
            rm = outs[1]
            lg = ttnn.to_torch(rm, mesh_composer=ttnn.ConcatMeshToTensor(m, dim=-1)) if nd > 1 else ttnn.to_torch(rm)
            lg = lg.reshape(BLK, -1)[1:, :V].float()
            vals, ids = torch.topk(lg, self.top_k, dim=-1, sorted=False)
        else:
            shard = V // nd
            cv, ci = [], []
            for hi in range(2):
                for d in range(nd):
                    vv = self._rows(outs[1 + 2 * hi], d)
                    ii = self._rows(outs[2 + 2 * hi], d)
                    off = d * shard + hi * (shard // 2)
                    cv.append(vv.reshape(QPAD, -1)[1:BLK].float())
                    ii = ii.reshape(QPAD, -1)[1:BLK].to(torch.int64) & 0xFFFF  # UINT16 indices (shard half < 65536)
                    ci.append(ii + off)
            cv, ci = torch.cat(cv, dim=1), torch.cat(ci, dim=1)
            top = torch.topk(cv, self.top_k, dim=-1)
            vals, ids = top.values, torch.gather(ci, 1, top.indices)
        self.last_hidden, self.last_top = hid, (vals, ids)
        return self.selector.select(hid, vals, ids, anchor)

    def _embed_dev(self, tok):
        m = self.mesh
        x = self.target.embd(tok)
        x = ttnn.reshape(x, (1, 1, BLK, x.shape[-1]))
        if m.get_num_devices() > 1 and x.shape[-1] < H:  # TP embedding is fractured on the hidden dim
            from models.tt_transformers.tt.ccl import tt_all_gather

            x = tt_all_gather(x, m, self.target.tt_ccl, cluster_axis=None, dim=3, topology=ttnn.Topology.Linear)
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _layer(self, li, L, x, ctx_h, upd_idx, mask_t, cos_b, sin_b, cos_c, sin_c):
        # ---- attention block
        res = x
        hn = self._rms(x, L["in_norm"])
        da = self._conv_dyn(L, "attention_conv", hn)
        hc = self._conv_apply(hn, da[0])
        ttnn.deallocate(hn)
        q = self._lin(hc, L["q"], key="q")  # [1,1,8,4096]
        qh = self._heads(q, BLK, NH_L)
        ttnn.deallocate(q)
        q = self._rope(self._rms(qh, L["q_norm"]), cos_b, sin_b)  # [1,32,8,128]
        kvb = self._lin(hc, L["kv"], key="kv")
        ttnn.deallocate(hc)
        kb = ttnn.slice(kvb, (0, 0, 0, 0), (1, 1, BLK, NKV_L * HD))
        vb = ttnn.slice(kvb, (0, 0, 0, NKV_L * HD), (1, 1, BLK, 2 * NKV_L * HD))
        ttnn.deallocate(kvb)
        kb = self._rope(self._rms(self._heads(kb, BLK, NKV_L), L["k_norm"]), cos_b, sin_b)  # [1,8,8,128]
        vb = self._heads(vb, BLK, NKV_L)
        # context rows -> ring (one row per lane, invalid rows to the trash row); block rows -> row BROW
        kc, vc = self._ctx_kv(L, ctx_h, NCTX, cos_c, sin_c)  # [1,8,8,128]
        for src, ring, ix in ((kc, self.ring_k[li], upd_idx), (vc, self.ring_v[li], upd_idx),
                              (kb, self.ring_k[li], self.blk_idx), (vb, self.ring_v[li], self.blk_idx)):
            r = ttnn.reshape(src, (1, NKV_L * 8, 1, HD))
            r = ttnn.pad(r, [1, NKV_L * 8, 32, HD], [0, 0, 0, 0], 0.0)
            r = ttnn.to_memory_config(r, self.kv_upd_cfg)
            ttnn.experimental.paged_update_cache(ring, r, update_idxs_tensor=ix)
            ttnn.deallocate(r)
            ttnn.deallocate(src)
        kfull = ttnn.reshape(self.ring_k[li], (1, NKV_L, RING, HD))
        vfull = ttnn.reshape(self.ring_v[li], (1, NKV_L, RING, HD))
        qp = ttnn.pad(q, [1, NH_L, QPAD, HD], [0, 0, 0, 0], 0.0)  # may alias q (padded height already 32): no free of q
        attn = ttnn.transformer.scaled_dot_product_attention(
            qp, kfull, vfull, attn_mask=mask_t, is_causal=False, scale=HD**-0.5,
            program_config=self.sdpa_cfg, compute_kernel_config=self.hifi2,
        )
        ttnn.deallocate(qp)
        a = ttnn.slice(attn, (0, 0, 0, 0), (1, NH_L, BLK, HD))
        ttnn.deallocate(attn)
        a = ttnn.reshape(ttnn.transpose(a, 1, 2), (1, 1, BLK, NH_L * HD))
        o = self._lin(a, L["o"], key="o")
        ttnn.deallocate(a)
        if TP:
            o = self._allreduce(o)
        o2 = self._conv_apply(o, da[1])
        ttnn.deallocate(o)
        x = ttnn.add(res, o2)
        ttnn.deallocate(o2)
        ttnn.deallocate(res)
        for row in da:
            for t in row:
                ttnn.deallocate(t)
        # ---- MLP block
        res = x
        hn = self._rms(x, L["post_norm"])
        dm = self._conv_dyn(L, "mlp_conv", hn)
        hc = self._conv_apply(hn, dm[0])
        ttnn.deallocate(hn)
        _pc = self._pc1d(hc, L["gate"], act=ttnn.UnaryOpType.SILU, key="gate")
        if _pc is not None:
            gt = ttnn.linear(hc, L["gate"], compute_kernel_config=self.hifi2, program_config=_pc)
        else:
            gt = ttnn.linear(hc, L["gate"], compute_kernel_config=self.hifi2, activation="silu")
        up = self._lin(hc, L["up"], key="up")
        ttnn.deallocate(hc)
        act = ttnn.multiply(gt, up)
        ttnn.deallocate(gt)
        ttnn.deallocate(up)
        dn = self._lin(act, L["down"], key="down")
        ttnn.deallocate(act)
        if TP:
            dn = self._allreduce(dn)
        m2 = self._conv_apply(dn, dm[1])
        ttnn.deallocate(dn)
        x = ttnn.add(res, m2)
        ttnn.deallocate(m2)
        ttnn.deallocate(res)
        for row in dm:
            for t in row:
                ttnn.deallocate(t)
        return x

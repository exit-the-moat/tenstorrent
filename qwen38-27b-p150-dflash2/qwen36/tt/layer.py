# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid TransformerBlock for Qwen3.5-9B.

Dispatches to either Gated DeltaNet (linear attention) or Gated Full Attention
based on the layer index. Both share the same RMSNorm + residual pattern and MLP.
"""

import os

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
from models.demos.blackhole.qwen36.tt.gdn import GDNConfig, Qwen36GatedDeltaNet
from models.demos.blackhole.qwen36.tt import decode_mode as _dm
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.utils.substate import substate
from models.tt_transformers.tt.common import Mode


def _same_device_buffer(a, b):
    """True only when both tensors are allocated device tensors at the same buffer address.

    A re-layout that is a no-op can come back as a NEW tensor handle on the SAME device buffer (ring-only decode,
    ep2 r64x); deallocating such a handle frees the caller's tensor. HOST tensors do not support buffer_address
    (TT_FATAL), so anything that is not a devic tensor - or any error - answers False (keep the old free).
    """
    try:
        return bool(a.is_allocated() and b.is_allocated() and a.buffer_address() == b.buffer_address())
    except Exception:  # noqa: BLE001
        return False


class Qwen36DecoderLayer:
    """Single transformer layer with hybrid attention dispatch.

    Pattern: x → attention_norm → attention → residual → ff_norm → MLP → residual
    Attention is either GatedAttention (full, with RoPE) or GatedDeltaNet (linear).
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, tensor_cache_path=None, tt_ccl=None):
        self.layer_num = layer_num
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1)
        self.tp = self.num_devices > 1 or os.environ.get("QWEN36_TP1") == "1"  # df1c QWEN36_TP1
        self.is_full_attention = args.is_full_attention_layer(layer_num)

        prefix = f"layers.{layer_num}"

        # Zero-centered RMSNorm (Qwen3.5): output = x_normed * (1 + weight). The
        # framework RMSNorm applies the +1 internally via add_unit_offset=True and
        # is mesh-aware (replicates the weight across a MeshDevice).
        #
        # Single device: plain RMSNorm on the full hidden state (validated path).
        # TP (27B on a (1,4) mesh): the residual stream is fractured along the
        # hidden dim, so each norm is wrapped in the framework DistributedNorm,
        # which all-gathers (PREFILL: distributed rmsnorm + gather; DECODE:
        # gather-then-norm) to hand the modules a replicated full-dim input —
        # exactly as models/demos/qwen35_27b does via the framework decoder.
        # Prefill fuses the norm all-gather into the in-proj matmul (all_gather_minimal_matmul_async):
        # GDN qkvzab and full-attn QKV. attention_norm then skips its post-norm AG (prefill only;
        # decode gathers pre-norm). Gates must match the module-side _fuse_agmm gates.
        self._fuse_norm_agmm = self.num_devices > 1 and (
            (not self.is_full_attention and getattr(args, "gdn_qkvz_weight_memcfg", None) is not None)
            or (self.is_full_attention and getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None)
        )
        self.attention_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "input_layernorm",
            tensor_cache_path,
            tt_ccl,
            "attention_norm",
            enable_all_gather=not self._fuse_norm_agmm,
        )
        # Prefill: ff_norm skips AG (fused into gate/up AGMM); decode gathers pre-norm so this is a no-op there.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)
        self.ffn_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "post_attention_layernorm",
            tensor_cache_path,
            tt_ccl,
            "ff_norm",
            enable_all_gather=not self._fuse_ff_agmm,
        )

        if self.tp:
            # Tensor-parallel modules (sharded weights from the raw substate).
            # Cache the sharded mesh weights to disk so re-runs skip the (slow,
            # single-threaded) reorder+shard of the full 27B.
            tp_cache = (tensor_cache_path / f"layers.{layer_num}" / "tp") if tensor_cache_path else None
            if self.is_full_attention:
                from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp

                tw = load_attention_weights_tp(
                    mesh_device, substate(state_dict, f"layers.{layer_num}.self_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPAttention(mesh_device, args, tw, tt_ccl)
            else:
                from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp

                tw = load_gdn_weights_tp(
                    mesh_device, substate(state_dict, f"layers.{layer_num}.linear_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
        elif self.is_full_attention:
            attn_state = substate(state_dict, f"layers.{layer_num}.self_attn")
            attn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen36GatedAttention(mesh_device, AttentionConfig.from_args(args), attn_state, attn_cache)
        else:
            gdn_state = substate(state_dict, f"layers.{layer_num}.linear_attn")
            gdn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen36GatedDeltaNet(mesh_device, GDNConfig.from_args(args), gdn_state, gdn_cache)

        # ep2: QWEN36_ADDNORM_FUSED=1 -> single-device decode norms (+ the residual add before ff_norm) as one
        # multi-core program each (tt/addnorm_fused.py); weights packed now, before any trace capture
        self._addnorm = None
        if (not self.tp):
            from models.demos.blackhole.qwen36.tt import addnorm_fused as _an

            if _an.enabled():
                self._addnorm = _an.get(mesh_device, args.dim)
                self._addnorm.packed_weight(self.attention_norm.weight)
                self._addnorm.packed_weight(self.ffn_norm.weight)

        # ep2 QWEN36_ADDNORM_TP=1 (needs QWEN36_CCL_AR=1): TP decode residual add + RMSNorm as one program
        # (tt/addnorm_tp.py); norm weights packed now, before any trace capture
        self._antp = None
        if self.tp:
            from models.demos.blackhole.qwen36.tt import addnorm_tp as _antp_mod
            from models.demos.blackhole.qwen36.tt import tp_common as _tpc0

            if _antp_mod.enabled() and _tpc0.ccl_ar_enabled():
                _res_mc = getattr(args, "_ep2_res_mc", None) or args.get_norm_config("attn", Mode.DECODE)[
                    "sharded_output_config"
                ]
                self._antp = _antp_mod.get(mesh_device, _res_mc, args.dim)
                self._antp.packed_weight(self.attention_norm.norm.weight)
                self._antp.packed_weight(self.ffn_norm.norm.weight)

        mlp_state = substate(state_dict, f"layers.{layer_num}.mlp")
        mlp_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        self.feed_forward = Qwen36MLP(mesh_device, mlp_state, mlp_cache, args=args, tt_ccl=tt_ccl)

    def _make_norm(
        self,
        mesh_device,
        args,
        state_dict,
        layer_num,
        weight_key,
        tensor_cache_path,
        tt_ccl,
        ag_key,
        enable_all_gather=True,
    ):
        """Build the per-layer RMSNorm; wrap in DistributedNorm when TP>1.

        On a single device this returns the same plain RMSNorm the validated 9B
        path used. The DistributedNorm wrapper (TP>1) mirrors tt_transformers
        decoder.py and handles the fractured->replicated transition.
        """
        norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key=weight_key,
            state_dict_prefix=f"layers.{layer_num}.",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
            **(
                dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=tt_ccl)
                if self.tp
                else {}
            ),
        )
        if self.tp:
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            return DistributedNorm(
                norm, args, tt_ccl=tt_ccl, TG=args.is_galaxy, ag_config_key=ag_key, enable_all_gather=enable_all_gather
            )
        return norm

    def _ring_norm_config(self, base):
        """QWEN36_AR_RING: the decode norm config with the PF ring residual layout (args._ep2_res_mc)."""
        nc = getattr(self, "_ring_nc", None)
        if nc is None:
            mc = self.args._ep2_res_mc
            bb = mc.shard_spec.grid.bounding_box()
            s, e = (getattr(bb, "start", None) or bb.start_coord), (getattr(bb, "end", None) or bb.end_coord)
            bw = mc.shard_spec.shape[1] // 32
            nc = self._ring_nc = dict(base)
            nc["sharded_program_config"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(e.x - s.x + 1, e.y - s.y + 1), subblock_w=bw, block_h=1, block_w=bw,
                inplace=False,
            )
            nc["sharded_output_config"] = mc
        return nc

    def _sd_norm_sharded(self, norm, x):
        """Single-device decode RMSNorm on the framework's sharded attention-input grid; interleaved L1 out."""
        nc = getattr(self, "_sd_norm_cfg", None)
        if nc is None:
            nc = self._sd_norm_cfg = self.args.get_norm_config("attn", Mode.DECODE)
        x_sh = ttnn.to_memory_config(x, nc["sharded_output_config"])
        y = ttnn.rms_norm(
            x_sh,
            epsilon=norm.eps,
            weight=norm.weight,
            program_config=nc["sharded_program_config"],
            memory_config=nc["sharded_output_config"],
            compute_kernel_config=norm.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(x_sh)
        out = ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(y)
        return out

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        mode="decode",
        chunk_size=128,  # = GDN long_prefill_chunk_size; the only size the chunk-seq prefill kernel supports
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        valid_len=None,
        gdn_collect=False,
    ):
        """One layer forward. mode == "decode" marks ONE DECODE STEP: the decode/prefill branches inside the
        layer and the MLP are then chosen by that flag, not by the activation height (tt/decode_mode.py), so a
        decode step of several tile rows (a DFlash2 verify at 8+ streams) still takes the decode branches.
        args.max_decode_rows is the widest decode step those branches must serve."""
        _rest = (cos, sin, mode, chunk_size, position_tensor, page_table, chunk_page_table, chunk_start_idx,
                 chunk_start_idx_tensor, valid_len, gdn_collect)
        if mode != "decode":
            return self._forward(x, *_rest)
        with _dm.step(getattr(self.args, "max_decode_rows", None)):
            return self._forward(x, *_rest)

    def _forward(
        self,
        x,
        cos=None,
        sin=None,
        mode="decode",
        chunk_size=128,  # = GDN long_prefill_chunk_size; the only size the chunk-seq prefill kernel supports
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        valid_len=None,
        gdn_collect=False,
    ):
        _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        if self.tp:
            # TP: DistributedNorm uses the framework's per-norm memory configs.
            _attn_norm_config = self.args.get_norm_config("attn", _norm_mode)
            if _norm_mode == Mode.DECODE:
                _attn_norm_config = self.args.decode_norm_cfg(x.shape[-2])
            # PREFILL: distributed rmsnorm outputs in L1 so the fused in-proj AGMM gathers from L1, not DRAM.
            if _norm_mode == Mode.PREFILL:
                _attn_norm_config = {**_attn_norm_config, "distributed_output_mem_config": ttnn.L1_MEMORY_CONFIG}
            # DECODE ff_norm uses the attn_norm layout (act_shard_hidden, 32-core) so Qwen36MLP's input reshard is a no-op and the norm runs on 32 cores not 8; PREFILL keeps the framework ff config.
            if _norm_mode == Mode.DECODE:
                # Stage A2 (r64b): sized to this step's tile rows (a 64-slot layout breaks narrow buckets)
                _ff_norm_config = self.args.decode_norm_cfg(x.shape[-2])
            else:
                # ff_norm output stays DRAM: L1 keeps the full-width norm resident across the whole MLP,
                # clashing with each matmul's CBs (w1/w3/w2) for no gain. Verified dead end; keep DRAM.
                _ff_norm_config = self.args.get_norm_config("ff", _norm_mode)
        else:
            # In decode the norm output stays in L1 (as the old rms_norm_ttnn(memory_config=L1) did);
            # in prefill the framework RMSNorm returns interleaved DRAM (matches the old None default).
            _attn_norm_config = _ff_norm_config = (
                {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if mode == "decode" else None
            )
        _an = self._addnorm if (mode == "decode" and self._addnorm is not None and self._addnorm.fits(x)) else None
        if mode == "decode" and self._addnorm is not None and not getattr(self, "_an_logged", False):
            self._an_logged = True
            from loguru import logger

            logger.info(
                f"ep2 QWEN36_ADDNORM_FUSED layer {self.layer_num}: {'active' if _an is not None else 'FALLBACK'} "
                f"(x {x.dtype} {x.layout} {list(x.shape)} {x.memory_config().buffer_type})"
            )
        # ep2: single-device decode norms on a width-sharded multi-core grid (the plain RMSNorm runs on one core,
        # ~81 us each, two per layer). Output goes back to interleaved L1, as before. QWEN36_ADDNORM_FUSED wins.
        _sd_sharded_norm = (
            _an is None
            and (not self.tp)
            and mode == "decode"
            and os.environ.get("QWEN36_SD_NORM_SHARDED") == "1"
        )
        # ep2 QWEN36_CCL_AR: x arrives replicated (full width, attn-norm sharded layout): norm without the all-gather
        from models.demos.blackhole.qwen36.tt import tp_common as _tpc

        _ar = mode == "decode" and self.tp and _tpc.ar_active()
        _antp = self._antp if _ar else None
        # ep2 QWEN36_AR_RING=1: the replicated residual lives on the PF receiver ring (8 x 5 cores, 4 tiles each). The
        # norms run on the usual 8 x 4 grid after a reshard (QWEN36_AR_RING_NORM32=1, default: bit-identical to the
        # plain AR path) or, with QWEN36_AR_RING_NORM32=0, on the ring grid itself (other partial sums: 64 layers
        # logits PCC 0.85, argmax 32/33 -> not usable)
        _ring_norm32 = None
        if _ar and _antp is None and getattr(self.args, "_ep2_res_mc", None) is not None:
            if os.environ.get("QWEN36_AR_RING_NORM32", "1") == "1":
                _ring_norm32 = _attn_norm_config["sharded_output_config"]
            else:
                _attn_norm_config = _ff_norm_config = self._ring_norm_config(_attn_norm_config)
        _own_x = False
        if _antp is not None:
            from models.demos.blackhole.qwen36.tt.addnorm_tp import Pending

            if isinstance(x, Pending):  # the previous layer's residual add, fused into this norm
                xr, attn_input = _antp(x.h, x.d, self.attention_norm.norm)
                ttnn.deallocate(x.h)
                ttnn.deallocate(x.d)
                x, _own_x = xr, True
            else:
                attn_input = _antp(x, None, self.attention_norm.norm)[1]
        elif _ar:
            _xn = ttnn.to_memory_config(x, _ring_norm32) if _ring_norm32 is not None else x
            attn_input = self.attention_norm.norm(
                _xn, mode=_norm_mode, in_sharded=True, out_sharded=True, norm_config=_attn_norm_config
            )
            # ep2 r64x: to_memory_config may return a NEW handle on the SAME buffer (ring-only decode measured
            # _xn is x False at the same address as x); deallocating it freed the layer's residual x and the
            # add below failed with "Input Tensor A is not allocated". Free only a genuinely separate buffer.
            if _xn is not x and not _same_device_buffer(_xn, x):
                ttnn.deallocate(_xn)
        elif _an is not None:
            attn_input = _an(x, None, self.attention_norm)[1]
        elif _sd_sharded_norm:
            attn_input = self._sd_norm_sharded(self.attention_norm, x)
        else:
            attn_input = self.attention_norm(x, mode=_norm_mode, norm_config=_attn_norm_config)

        if self.tp:
            # TP modules: input is the gathered (full-dim) norm output [1,1,B/S,dim];
            # output is fractured along dim=3. cos/sin are in rope_tp format.
            if self.is_full_attention:
                if mode == "prefill":
                    # Contract/vLLM path supplies a page_table → paged KV prefill; the
                    # demo path (no page_table) uses the internal concat caches.
                    if page_table is not None:
                        attn_output = self.attention.forward_prefill_paged(
                            attn_input,
                            cos,
                            sin,
                            page_table,
                            chunk_page_table=chunk_page_table,
                            chunk_start_idx=chunk_start_idx if chunk_start_idx is not None else 0,
                            chunk_start_idx_tensor=chunk_start_idx_tensor,
                        )
                    else:
                        attn_output = self.attention.forward_prefill(attn_input, cos, sin)
                else:
                    attn_output = self.attention.forward_decode(
                        attn_input, position_tensor, cos, sin, page_table=page_table
                    )
            else:
                # GDN carries its recurrent/conv state internally (capture_state on
                # prefill, read on decode); it has no paged KV, so page_table is N/A.
                if mode == "prefill":
                    if gdn_collect:
                        # Batched per-user prefill: stash this user's from-scratch state for
                        # assembly into row u of the batched buffers (finalize_pending later).
                        attn_output = self.attention.forward_prefill_collect(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len
                        )
                    else:
                        attn_output = self.attention.forward_prefill(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len, capture_state=True
                        )
                else:
                    attn_output = self.attention.forward_decode(attn_input)
        elif self.is_full_attention:
            attn_output = self.attention.forward(
                attn_input,
                cos,
                sin,
                position_tensor=position_tensor,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )
        else:
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(
                attn_input, mode=deltanet_mode, chunk_size=chunk_size, valid_len=valid_len
            )
        ttnn.deallocate(attn_input)

        if _antp is not None:
            h, ff_input = _antp(x, attn_output, self.ffn_norm.norm)
            ttnn.deallocate(attn_output)
            if _own_x:
                ttnn.deallocate(x)
        elif _ar:
            h = ttnn.add(x, attn_output, memory_config=x.memory_config())
            ttnn.deallocate(attn_output)
            _hn = ttnn.to_memory_config(h, _ring_norm32) if _ring_norm32 is not None else h
            ff_input = self.ffn_norm.norm(_hn, mode=_norm_mode, in_sharded=True, out_sharded=True, norm_config=_ff_norm_config)
            # ep2 r64x: same-buffer handle (see the attention-norm site); deallocating it freed h and the final
            # residual add failed with "Input Tensor A is not allocated".
            if _hn is not h and not _same_device_buffer(_hn, h):
                ttnn.deallocate(_hn)
        elif _an is not None and _an.fits(attn_output):
            h, ff_input = _an(x, attn_output, self.ffn_norm)
            ttnn.deallocate(attn_output)
        else:
            # pf5 QWEN36_PF_RES_L1: bit 0 = the mid-layer residual h in L1, bit 1 = the layer output in L1 (prefill, TP)
            _rl1 = int(os.environ.get("QWEN36_PF_RES_L1", "0")) if (mode == "prefill" and self.tp) else 0
            h = ttnn.add(x, attn_output, memory_config=ttnn.L1_MEMORY_CONFIG) if _rl1 & 1 else ttnn.add(x, attn_output)
            ttnn.deallocate(attn_output)
            if _sd_sharded_norm:
                ff_input = self._sd_norm_sharded(self.ffn_norm, h)
            else:
                ff_input = self.ffn_norm(h, mode=_norm_mode, norm_config=_ff_norm_config)

        ff_output = self.feed_forward.forward(ff_input)
        ttnn.deallocate(ff_input)
        if _antp is not None:  # QWEN36_ADDNORM_TP: h + ff_output is added by the next norm
            from models.demos.blackhole.qwen36.tt.addnorm_tp import Pending

            return Pending(h, ff_output)

        if _ar:
            output = ttnn.add(h, ff_output, memory_config=h.memory_config())
        elif mode == "prefill" and self.tp and int(os.environ.get("QWEN36_PF_RES_L1", "0")) & 2:
            output = ttnn.add(h, ff_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            output = ttnn.add(h, ff_output)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)

        return output

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM wrapper for Qwen3.5/3.6 on Blackhole — a thin tt_transformers Generator subclass.

Hybrid model: 8 paged-KV attention + 24 GDN recurrent-state layers. GDN forbids token-padding and
isn't position-general, so prefill is model-owned (masked-bucket for short prompts / chunk-outer
trace for long, via prefill_dispatch) while Generator drives decode only. GDN + KV state is
model-bound, so the kv_cache contract param is accepted but unused.
"""

import math
import os
from collections import defaultdict
from typing import Mapping, Optional

import torch
from loguru import logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.demos.blackhole.qwen36.tt.common import create_tt_model
from models.demos.blackhole.qwen36.tt.generator_interface import prefill_dispatch, warmup_decode_buckets
from models.tt_transformers.tt.generator import Generator


def _pf4_rel_split():
    from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

    return [getattr(pf, "release_t", None) for pf in getattr(_wp, "_by_mesh", {}).values()]


def _pf_release():
    from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

    _wp.release_all()

_PREFILL_WARMUP_CHUNK = int(os.environ.get("QWEN36_PF_CHUNK", "2048"))  # t2: prefill chunk (trace) size
_PREFILL_WARMUP_BUCKET = 4096
_BLOCK_SIZE = 64


class TT_Qwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # Serve a single visual item per request (B=1, max_concurrency=1). Image and video are both
        # supported, but only ONE modality per request: the model's vision splice keys off a single
        # placeholder token id (image_token_id XOR video_token_id), so a mixed image+video prompt
        # cannot be spliced correctly.
        return {"image": 1, "video": 1}


# --- r64c STDBG: per-step GDN state / logits dump (env QWEN36_STDBG=<file>; off by default) ---
def _stdbg_path():
    return os.environ.get("QWEN36_STDBG", "")


def _stdbg_rows(self, tag, logits=None, model=None, max_rows=8):
    p = _stdbg_path()
    if not p:
        return
    n = getattr(self, "_stdbg_n", 0)
    lim = int(os.environ.get("QWEN36_STDBG_STEPS", "16"))
    if n >= lim:
        return
    self._stdbg_n = n + 1
    parts = [f"STDBG {tag} step={n} bb={getattr(self, '_stdbg_bb', None)}"]
    try:
        if logits is not None:
            lg = logits.float() if hasattr(logits, "float") else logits
            if lg.dim() >= 2:
                row = lg.reshape(lg.shape[0], -1)
                am = row.argmax(dim=-1)[:max_rows].tolist()
                mx = [round(float(v), 4) for v in row.max(dim=-1).values[:max_rows]]
                parts.append(f"logits_rows={row.shape[0]} argmax={am} top1={mx}")
        if model is not None:
            ids = getattr(model, "_ep2_ids", None)
            if ids is not None:
                try:
                    parts.append(f"ids={[int(v) for v in ids.reshape(-1).tolist()][:64]}")
                except Exception:  # noqa: BLE001
                    pass
            dn = next(ly.attention for ly in model.layers if not ly.is_full_attention)
            devs = ttnn.get_device_tensors(dn.rec_state)
            t = ttnn.to_torch(devs[0]).float().reshape(devs[0].shape[0], -1)
            rs = [round(float(v), 4) for v in t.sum(dim=1)]
            try:
                import struct as _st
                _r8 = t.reshape(t.shape[0], -1)[:8, :4].float().reshape(-1).tolist()
                parts.append("recbits=" + ",".join(_st.pack("<f", float(v)).hex() for v in _r8))
                _nl = int(os.environ.get("QWEN36_STDBG_LAYERS", "4"))
                _gd = [ly.attention for ly in model.layers if not ly.is_full_attention][:_nl]
                for _li, _dn in enumerate(_gd):
                    _d0 = ttnn.get_device_tensors(_dn.rec_state)[0]
                    _tt = ttnn.to_torch(_d0).float().reshape(_d0.shape[0], -1)[:8, :4].float().reshape(-1).tolist()
                    parts.append(f"L{_li}=" + ",".join(_st.pack("<f", float(v)).hex() for v in _tt))
            except Exception as _e:  # noqa: BLE001
                parts.append(f"LERR={_e!r}")
            ra = [round(float(v), 4) for v in t.abs().sum(dim=1)]
            parts.append(f"rec_sum={rs} rec_abs={ra}")
            cs = []
            for m in range(dn.K):
                d = ttnn.get_device_tensors(dn.conv_states[m])[0]
                tc = ttnn.to_torch(d).float().reshape(d.shape[-2], -1)
                cs.append([round(float(v), 4) for v in tc.sum(dim=1)])
            parts.append(f"conv_sum={cs}")
    except Exception as e:  # noqa: BLE001  (debug only: never break serving)
        parts.append(f"ERR {e!r}")
    with open(p, "a") as f:
        f.write(" ".join(parts) + "\n")


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)

class Qwen36ForCausalLM(Generator, SupportsMultiModal):
    """vLLM-compatible wrapper for Qwen3.5-9B on Blackhole P150."""

    # Decode bucketing keeps several traces live and refreshes the selected
    # bucket's inputs before replay, so their I/O buffers may safely overlap.
    _tt_allow_decode_trace_buffer_reuse = True

    # supports_async_decode=False: async decode assumes on-device token/position continuity, which
    # corrupts Qwen's GDN scan. supports_sample_on_device=True: on-device sampling is decode-only.
    #
    # ep2 host agent, QWEN36_ASYNC_HOST=1 (default off; needs vLLM --async-scheduling): declare async decode for HOST
    # sampling only (TP=2 has no on-device sampling, see _validate_device_sampling_request). The plugin then submits
    # each decode without waiting (decode_forward(read_from_device=False) + read_decode_output(async_read=True)) and,
    # because a host-sampled step is never "overlap_ok", drains it (readback + host sampling + token applied to the
    # runner state) before it builds the next step's inputs from that state. Tokens and positions stay host-
    # authoritative (the device-token continuity path of Generator.decode_forward is device-sampling only), so the
    # GDN concern above does not apply; what overlaps the card is vLLM's scheduler step and output processing.
    # ep2 pcx QWEN36_PREFIX_CACHE=1 (default off): vLLM prefix caching on. vLLM shares the paged-KV blocks (it sees
    # only the attention layers); the GDN state at the hit boundary comes from the model's snapshot store
    # (Qwen36Model._pcx_*, saved at 2048-token prefill boundaries). Needs --enable-prefix-caching --block-size 2048
    # and the patched vllm/model_executor/models/config.py (mamba_cache_mode stays "none").
    model_capabilities = {
        "supports_prefix_caching": os.environ.get("QWEN36_PREFIX_CACHE", "0") == "1",
        "supports_async_decode": os.environ.get("QWEN36_ASYNC_HOST", "0") == "1",
        "supports_sample_on_device": True,
    }

    def _validate_device_sampling_request(self, requested):
        if not requested:
            return
        if os.environ.get("QWEN36_ASYNC_HOST", "0") == "1":
            raise RuntimeError("QWEN36_ASYNC_HOST=1 is for host sampling only (async device sampling corrupts the GDN scan)")
        for model in self.model:
            if model.sampling is not None:
                continue
            mesh_shape = tuple(int(dim) for dim in model.mesh_device.shape)
            logits_per_device = math.ceil(model.args.vocab_size / model.num_devices)
            raise RuntimeError(
                "Qwen3.6 on-device sampling requires a certified TP topology (1x4 or 1x8) "
                f"with at most 65536 logits/device; got mesh={mesh_shape}, "
                f"vocab={model.args.vocab_size}, logits/device={logits_per_device}. "
                "Unset sample_on_device_mode for host sampling."
            )

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> int:
        """All-user KV capacity (the shared paged-KV token pool).

        QWEN36_MAX_TOKENS_ALL_USERS overrides it with a FIXED pool (set per device+model from the
        tt-inference-server spec's env_vars, mirroring GEMMA4_MAX_TOKENS_ALL_USERS). This decouples
        the pool from max_model_len × max_num_seqs so ONE config serves both a single long request
        (up to max_model_len) and a batch of shorter ones (sum of lengths ≤ pool) — e.g. 524288 =
        1×256K or 8×64K. Without the override, fall back to max_model_len × max_num_seqs (the old
        per-config product) so existing single-mode specs are unchanged."""
        override = os.environ.get("QWEN36_MAX_TOKENS_ALL_USERS")
        if override:
            return int(override)
        if max_model_len is not None:
            return int(max_model_len) * int(max_num_seqs or 1)
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            max_num_seqs=max_num_seqs,
            **kwargs,
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        # Weights dir: MODEL_WEIGHTS_DIR → HF_MODEL → hf_config._name_or_path; a hub id resolves to a local snapshot.
        name_or_path = os.environ.get("MODEL_WEIGHTS_DIR") or os.environ.get("HF_MODEL") or hf_config._name_or_path
        if name_or_path and not os.path.isdir(os.path.expanduser(name_or_path)):
            from huggingface_hub import snapshot_download

            # When offline/CI, resolve from the local cache only so snapshot_download
            # reads the cached refs instead of reaching the HF API (refused by HF_HUB_OFFLINE=1).
            offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
            name_or_path = snapshot_download(name_or_path, local_files_only=offline)
        args, model, _ = create_tt_model(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, hf_model=name_or_path
        )
        # Attach the TT vision tower so prefill can splice image/video embeddings (multimodal path).
        # No-op cost for text-only requests; get_image_features / get_video_features are only invoked
        # when a request actually carries pixel_values / pixel_values_videos.
        model.init_vision_model()
        return cls([model], [args], mesh_device)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate paged KV (8 attn layers) + external GDN state; returns the 8 KV pairs.

        batch_size = max_batch_size (vLLM's max_num_seqs, threaded through initialize_vllm_model):
        the paged KV blocks (kv_cache_shape) already cover all users, and this sizes the per-slot
        GDN recurrent/conv state [B,...] + the decode kv grid. B==1 is the single-sequence path."""
        batch_size = self.model[0].args.max_batch_size
        kv_cache_shape = list(kv_cache_shape)
        self._kv_sub = 1
        bs = int(kv_cache_shape[2])
        _pcx_sub = os.environ.get("QWEN36_PREFIX_CACHE", "0") == "1" and bs > 32  # pcx: 2048-token vLLM blocks
        if (_PREFILL_WARMUP_CHUNK % bs or _pcx_sub) and os.environ.get("QWEN36_KV_SUBBLOCK", "1") == "1" and (
            self.model[0].num_devices == 1 or os.environ.get("QWEN36_KV_SUBBLOCK_TP", "1") == "1"
        ):
            # ep2 devloop (2026-11-03): the TP chunked prefill (_prefill_traced_chunked_tp / _prefill_chunked_eager_tp,
            # the masked tail) has the same block math (chunk // block, cs // block): the same misplaced KV on 2 cards
            # ep2 fix (1-card NIAH "state leak"): vLLM sets the attention block size of this hybrid model to 800 tokens
            # (attention page >= GDN state page), whatever --block-size says. The 1-card chunked prefill fills KV per
            # 2048-token chunk with paged_fill_cache, which starts at a block boundary and writes whole blocks: with
            # 800-token blocks, blocks_per_chunk = 2 (1600 tokens) and every chunk start after the first falls mid-block,
            # so KV landed at wrong offsets and whole blocks of a long prompt were never written. Decode then read KV
            # left there by EARLIER requests (one NIAH answer was the code of the first request). Fix: keep vLLM's
            # blocks but store each as `sub` model blocks of 32 tokens (same memory), and expand every page table
            # (vLLM block b -> model blocks b*sub .. b*sub+sub-1). 2048 is a multiple of 32, so chunks align.
            sub = bs // 32
            assert sub * 32 == bs, f"KV block {bs} is not a multiple of 32"
            self._kv_sub = sub
            kv_cache_shape[0] *= sub
            kv_cache_shape[2] = 32
            logger.info(f"QWEN36_KV_SUBBLOCK: vLLM KV block {bs} tokens stored as {sub} blocks of 32 ({kv_cache_shape})")
        return self.model[0].allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=batch_size)

    def _expand_pt(self, page_table):
        """vLLM page table (blocks of sub*32 tokens) -> model page table (blocks of 32 tokens); identity when sub == 1."""
        sub = getattr(self, "_kv_sub", 1)
        if sub == 1 or page_table is None or not isinstance(page_table, torch.Tensor):
            return page_table
        pt = page_table.to(torch.int32)
        return (pt.unsqueeze(-1) * sub + torch.arange(sub, dtype=torch.int32)).reshape(pt.shape[0], -1).contiguous()

    @staticmethod
    def _has_visual(kwargs, pixel_key):
        """True only when the request carries REAL visual data for this modality. vLLM attaches an
        empty pixel_values placeholder to text requests for a multimodal-registered model, so a
        plain ``is not None`` check misclassifies text as multimodal. Mirrors the emptiness test in
        _gather_user_visual (key absent / empty list / first item None => text-only)."""
        v = kwargs.get(pixel_key)
        return v is not None and len(v) > 0 and v[0] is not None

    @staticmethod
    def _gather_user_visual(kwargs, pixel_key, grid_key):
        """Pull this (B=1) user's patches + (t,h,w) grids for one modality out of the vLLM kwargs.

        Returns (pixel_values, grid_thw) or None when the request carries nothing for this modality.
        Multiple items for the user arrive as lists; concat the patches and stack the grids (same
        shape get_image_features / get_video_features expect: [num_patches, patch_dim] + [N, 3]).
        """
        if pixel_key not in kwargs or len(kwargs[pixel_key]) == 0 or kwargs[pixel_key][0] is None:
            return None
        pixel_values = kwargs[pixel_key][0]
        grid_thw = kwargs[grid_key][0]
        if isinstance(pixel_values, list) and len(pixel_values) > 0:
            pixel_values = torch.concat(pixel_values, dim=0)
            grid_thw = torch.stack([g.to(dtype=torch.int32) for g in grid_thw], dim=0)
        return pixel_values, grid_thw

    def _compute_vision_tokens(self, model, kwargs):
        """Run the vision tower for this (single-user, B=1) request, if it carries images or video.

        Mirrors the Qwen3-VL generator's multimodal check: pull this user's pixels + grid out of
        the vLLM kwargs and return the packed embeddings (ttnn [num_vision_tokens, H]) for prefill
        to splice in. Returns None for a text-only request, so the whole multimodal path is skipped.

        Image and video share the vision tower; dispatching to get_video_features (vs
        get_image_features) is what tells the model to splice into video_token_id placeholders and
        build the video M-RoPE. A request carries at most one visual modality (see
        get_supported_mm_limits); video takes precedence if both are somehow present.
        """
        video = self._gather_user_visual(kwargs, "pixel_values_videos", "video_grid_thw")
        if video is not None:
            return model.get_video_features(*video)

        image = self._gather_user_visual(kwargs, "pixel_values", "image_grid_thw")
        if image is not None:
            return model.get_image_features(*image)

        return None

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """pf4: QWEN36_PF_TIMING also logs the whole prefill_forward (wall, incl. the PF release) per call."""
        if os.environ.get("QWEN36_PF_TIMING") != "1":
            return self._prefill_forward_impl(tokens, page_table, kv_cache, prompt_lens, **kwargs)
        import time as _time

        _t0 = _time.perf_counter()
        r = self._prefill_forward_impl(tokens, page_table, kv_cache, prompt_lens, **kwargs)
        logger.info(
            f"QWEN36_PF_TIMING prefill_forward total_ms={(_time.perf_counter() - _t0) * 1e3:.1f} "
            f"release_ms={getattr(self, '_pf4_rel_ms', -1):.1f} tokens={tuple(tokens.shape)} rel {_pf4_rel_split()}"
        )
        return r

    def _prefill_forward_impl(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """All prefill is model-owned (Generator drives decode only)."""
        self._fi_rec = {}  # ep2 QWEN36_FAST_INPUTS: the trace input buffers may be reused by prefill allocations
        import time as _time

        _tr = _time.perf_counter()
        _pf_release()  # ep2 QWEN36_PF_GCB_MODE=decode: the prefill programs need the global CB L1
        self._pf4_rel_ms = (_time.perf_counter() - _tr) * 1e3
        model = self.model[0]
        page_table = self._expand_pt(page_table)
        if os.environ.get("QWEN36_DEBUG_PT") == "1":
            self._dbg_dec = 0
            _pt = page_table if isinstance(page_table, torch.Tensor) else None
            _T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
            _nb = -(-_T // 64)
            logger.info(
                f"DEBUG_PT prefill: tokens {tuple(tokens.shape)} T {_T} pt {None if _pt is None else tuple(_pt.shape)} "
                f"row0[:8] {None if _pt is None else _pt[0, :8].tolist()} row0[nb-4:nb+4] "
                f"{None if _pt is None else _pt[0, max(0, _nb - 4) : _nb + 4].tolist()} uniq {None if _pt is None else int(_pt[0, :_nb].unique().numel())}/{_nb} kwargs {sorted(kwargs)}"
            )
        if os.environ.get("QWEN36_DEBUG_ZERO_KV") == "1":
            # ep2 devloop DEBUG (state-leak hunt): zero every paged KV block before each prefill, so a request can
            # never read KV left by an earlier one. Slow; never for counted runs.
            # host zeros + host->device copy: no device program, so nothing compiles next to the parked traces
            for k_cache, v_cache in getattr(model, "_paged_kv_caches", None) or []:
                for c in (k_cache, v_cache):
                    z = getattr(self, "_dbg_zero_kv", None)
                    if z is None or list(z.shape) != list(c.shape) or z.dtype != c.dtype:
                        z = self._dbg_zero_kv = ttnn.from_torch(
                            torch.zeros(list(c.shape), dtype=torch.bfloat16), dtype=c.dtype, layout=ttnn.TILE_LAYOUT
                        )
                    ttnn.copy_host_to_device_tensor(z, c)
            logger.info("QWEN36_DEBUG_ZERO_KV: all paged KV blocks zeroed before this prefill")
        if model.tp and model.args.max_batch_size > 1:
            # Batched text prefill into decode slots (MM is B=1). Require real visual data, not a
            # non-None empty pixel_values placeholder from vLLM on text requests.
            assert not self._has_visual(kwargs, "pixel_values") and not self._has_visual(
                kwargs, "pixel_values_videos"
            ), (
                "batched (max_num_seqs>1) serving is text-only; multimodal is single-sequence "
                "(max_concurrency=1). Run the model at max_num_seqs=1 for image/video requests."
            )
            return self._prefill_forward_tp_batched(
                model, tokens, page_table, prompt_lens, kwargs.get("empty_slots"), start_pos=kwargs.get("start_pos")
            )
        vision_tokens = self._compute_vision_tokens(model, kwargs)
        if model.tp:
            return self._prefill_forward_tp(model, tokens, page_table, prompt_lens, vision_tokens=vision_tokens)
        seq_len = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        logger.info(f"Prefilling User 1 up to {seq_len} tokens")
        # Multimodal works WITH the captured trace here: prefill_dispatch routes to the traced
        # path, which splices the image/video rows via a fixed-shape ttnn.where over persistent
        # buffers (compiled at warmup, updated per request by copy_host_to_device — no request-time
        # compile).
        logits = prefill_dispatch(
            model,
            tokens,
            page_table,
            prompt_lens,
            use_trace=kwargs.get("enable_trace", False),
            vision_tokens=vision_tokens,
        )
        logits = ttnn.to_torch(logits)
        # The vLLM runner unpacks (logits, rope_deltas) because the HF config has mrope_section.
        # Zero deltas are returned for all modalities: the multimodal M-RoPE delta is applied
        # entirely model-side (build_request_rope stashes self.rope.rope_delta during prefill, and
        # every decode path offsets the rope position by it), so the value handed back to vLLM is
        # unused for device-side rope and stays zero.
        rope_deltas = torch.zeros(logits.shape[0], dtype=torch.long)
        logger.info(f"Finished prefill up to {seq_len} tokens, starting decode...")
        return logits, rope_deltas

    def _prefill_forward_tp(self, model, tokens, page_table, prompt_lens, vision_tokens=None):
        """TP (B=1) paged prefill via the model-owned masked fixed-bucket path.

        prefill_traced_chunked rounds the prompt up to a fixed bucket and masks the GDN to the
        EXACT valid_len, so prefill runs one of a bounded, pre-warmed program set (the
        compile-clobbers-trace fix) — for <=2048 prompts it is entirely the masked bucket (no
        chunk trace needed). Longer prompts replay the chunk-outer trace (Milestone B). Returns
        host logits [1, 1, vocab] gathered to a single replica."""
        T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        if tokens.shape[1] > T:
            tokens = tokens[:, :T]
        logger.info(f"Prefilling User 1 up to {T} tokens (TP masked-bucket/chunked)")
        # Multimodal is supported on TP too: prefill_traced_chunked splices the image/video rows via
        # a fixed-shape ttnn.where over hidden-sharded persistent buffers (the vision rows are
        # gathered to full hidden on host, placed along seq, then re-sharded), so no request-time
        # compile clobbers the parked trace.
        logits = model.prefill_traced_chunked(
            tokens, page_table, actual_len=T, vision_tokens=vision_tokens
        )  # [1,1,vocab] replicated
        logits = (
            ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
            .reshape(-1, model.args.vocab_size)[:1]
            .float()
            .view(1, 1, -1)
        )
        logger.info(f"Finished prefill up to {T} tokens, starting decode...")
        return logits, torch.zeros(1, dtype=torch.long)

    def _prefill_forward_tp_batched(self, model, tokens, page_table, prompt_lens, empty_slots, start_pos=None):
        """TP batched (max_num_seqs>1) prefill: prefill each request in this step into its decode slot.

        vLLM prefills new requests while other slots decode, so each user's B=1 state is written into
        row empty_slots[u] of the batched GDN buffers without disturbing the live rows (model-owned,
        via prefill_paged_slots). Attention fills each request's blocks via its page-table row.

        tokens:      torch [N, max_T] (rows are the N requests scheduled this prefill step).
        page_table:  torch [N, max_blocks] — row u = request u's blocks.
        prompt_lens: per-request real lengths (row u trimmed to prompt_lens[u]).
        empty_slots: per-request decode slot; defaults to range(N) (mirrors Generator.prefill_forward_text).
        Returns ([N, 1, vocab] host logits, [N] zero rope_deltas — text M-RoPE delta is 0, applied model-side).
        """
        N = tokens.shape[0]
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [tokens.shape[1]] * N
        if empty_slots is None:
            empty_slots = list(range(N))
        empty_slots = [int(s) for s in empty_slots]
        token_ids_list = [tokens[u : u + 1, : plens[u]].to(torch.int32) for u in range(N)]
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        logger.info(f"Prefilling {N} user(s) into slots {empty_slots} (TP batched masked-bucket)")
        cached = None
        if start_pos is not None and os.environ.get("QWEN36_PREFIX_CACHE", "0") == "1":
            # pcx: vLLM's prefix-cache hit per request (tokens whose KV blocks are shared and already written)
            _sp = start_pos.tolist() if hasattr(start_pos, "tolist") else list(start_pos)
            cached = [max(0, int(_sp[u])) for u in range(N)]
        host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens, cached_lens=cached)
        _stdbg_rows(self, f"PREFILL n={N} slots={empty_slots} first_tok={[int(x.reshape(-1).argmax()) for x in host_logits]}", logits=host_logits[0] if host_logits else None, model=model)
        logits = torch.cat([hl.reshape(1, 1, -1) for hl in host_logits], dim=0)  # [N, 1, vocab]
        logger.info(f"Finished batched prefill of {N} user(s), starting decode...")
        return logits, torch.zeros(N, dtype=torch.long)

    def decode_forward(self, *args, **kwargs):
        if os.environ.get("QWEN36_HOST_TIMING", "0") != "0":
            # ep2 host agent: per-phase host timing of the decode step (diagnostics, tt/host_timing.py)
            import time as _time

            from models.demos.blackhole.qwen36.tt import host_timing as _ht

            _ht.install(self)
            _t = _time.perf_counter()
            _ht.enter(_t)
            try:
                return self._decode_forward_impl(*args, _ht_t0=_t, **kwargs)
            finally:
                _ht.leave(_t, _time.perf_counter())
        return self._decode_forward_impl(*args, **kwargs)

    def _decode_forward_impl(self, *args, _ht_t0=None, **kwargs):
        from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

        _wp.decode_all()  # ep2 QWEN36_PF: trace captures from here on are decode captures (GCB recreated before them)
        if _wp.recreate_pending() and os.environ.get("QWEN36_KEEP_DECODE_TRACES", "0") == "1":
            # ep2 t3c: recreate the GCB at the SAME L1 address (ensure() checks it) and restart the prefetcher, then keep
            # every decode trace (as DFlash2 DF_NO_RECAP does for its verify trace): no re-capture after each prefill
            self._fi_rec = {}
            _wp.recreated()
        if _wp.recreate_pending():
            # ep2 QWEN36_PF_GCB_MODE=decode: the GCB was freed for a prefill; decode traces captured with the old one
            # read wrong data once it is recreated (64 layers: diverged at step 9). Drop every decode trace: the
            # Generator re-captures the one it needs (no precompile: its programs exist; the precompile pass would
            # run an extra eager decode step and advance the state).
            for _ids, _ins, _outs in getattr(self, "_bucket_trace_store", {}).values():
                for _k in list(_ids.keys()):
                    for _tid in (_ids[_k] or {}).values():
                        ttnn.release_trace(self.model[0].mesh_device, _tid)
                    _ids[_k], _ins[_k], _outs[_k] = None, None, None
            for _k in list(self.trace_ids_decode.keys()):
                if self.trace_ids_decode[_k]:
                    for _tid in self.trace_ids_decode[_k].values():
                        try:
                            ttnn.release_trace(self.model[0].mesh_device, _tid)
                        except Exception:  # noqa: BLE001  (already released via the bucket store)
                            pass
                    self.trace_ids_decode[_k] = None
            kwargs["skip_trace_precompile"] = True
            self._fi_rec = {}
            self._ep2_traces_dropped = True
            _wp.recreated()
        # Traced decode (single-device and TP): trace captured at pos 0 in warmup, replayed here.
        # Valid for TP — GDN state is in fixed in-place buffers, and prefill only replays pre-warmed programs.
        if not getattr(self, "_decode_logged", False):
            self._decode_logged = True
            logger.info("Decode trace replay active (Qwen)")
        model = self.model[0]
        # Batched serving: apply vLLM's condense slot_remap to the per-slot GDN recurrent/conv state
        # BEFORE the decode trace reads it. The plugin remaps its own buffers (and the seed RNG via
        # super().decode_forward), but GDN state is model-internal, so mirror the same reindex here.
        # slot_remap is passed through unchanged so the seed-RNG remap inside super() still runs.
        if model.tp and model.args.max_batch_size > 1:
            slot_remap = kwargs.get("slot_remap")
            if slot_remap is not None:
                model._remap_gdn_slots(slot_remap)
        # Decode bucketing (default on; TT_DECODE_BUCKETING=0 off): slice host inputs to the
        # smallest power-of-2 width >= active prefix [0:num_active) before the base forward.
        # No runner edit / output re-pad — plugin reads unpadded_batch_size in slot order.
        # Each width keeps its own trace metadata, inputs, and output.
        args = list(args)

        def _read(name, pos):
            if name in kwargs:
                return kwargs[name]
            return args[pos] if pos < len(args) else None

        def _write(name, pos, val):
            if name in kwargs:
                kwargs[name] = val
            elif pos < len(args):
                args[pos] = val

        tokens = _read("tokens", 0)
        if getattr(self, "_kv_sub", 1) > 1:
            _write("page_table", 2, self._expand_pt(_read("page_table", 2)))
        if os.environ.get("QWEN36_DEBUG_PT") == "1" and getattr(self, "_dbg_dec", 0) < 3:
            self._dbg_dec = getattr(self, "_dbg_dec", 0) + 1
            _sp, _pt = _read("start_pos", 1), _read("page_table", 2)
            _p = int(_sp.reshape(-1)[0]) if _sp is not None else -1
            _b = _p // 64
            logger.info(
                f"DEBUG_PT decode: start_pos {None if _sp is None else _sp.reshape(-1)[:2].tolist()} pt "
                f"{None if _pt is None else tuple(_pt.shape)} row0[b-4:b+2] {None if _pt is None else _pt[0, max(0, _b - 4) : _b + 2].tolist()} row0[:8] {None if _pt is None else _pt[0, :8].tolist()}"
            )
        if os.environ.get("TT_DECODE_BUCKETING", "1") == "1" and tokens is not None:
            start_pos = _read("start_pos", 1)
            width = int(tokens.shape[0])
            num_active = int((start_pos != -1).sum()) if start_pos is not None else width
            num_active = max(1, min(num_active, width))
            bucket = min(width, 1 << max(0, (num_active - 1).bit_length()))  # smallest pow2 >= num_active
            # Keep full width when slot_remap is set: remap indexes the full slot space (tokens /
            # GDN). Rare (row moves only); bucketing resumes next step. Check kw + positional #10.
            if _read("slot_remap", 10) is not None and os.environ.get("R64C_REMAP_BUCKET", "1") != "1":
                bucket = width
            self._stdbg_bb = (width, int(bucket), int(num_active))
            if bucket < width:
                _write("tokens", 0, tokens[:bucket])
                if start_pos is not None:
                    _write("start_pos", 1, start_pos[:bucket])
                page_table = _read("page_table", 2)
                if page_table is not None:
                    _write("page_table", 2, page_table[:bucket])
                tokens = tokens[:bucket]

        if tokens is not None:
            B = int(tokens.shape[0])
            if getattr(self, "_fi_B", None) != B or _read("slot_remap", 10) is not None:
                # ep2 QWEN36_FAST_INPUTS: decode-bucket traces may share (corruptible) input buffers, and a slot remap
                # runs eager device copies: only consecutive steps of one bucket keep the record of the inputs.
                self._fi_rec = {}
            self._fi_B = B
            store = getattr(self, "_bucket_trace_store", None)
            if store is None:
                store = self._bucket_trace_store = {}
            if B not in store:
                store[B] = (defaultdict(lambda: None), defaultdict(lambda: None), defaultdict(lambda: None))
            self.trace_ids_decode, self.trace_inputs_decode, self.trace_output_decode = store[B]
            if (
                getattr(self, "_ep2_traces_dropped", False)
                and os.environ.get("QWEN36_RECAP_NO_PRECOMPILE", "1") == "1"
                and not any(self.trace_ids_decode[_k] for _k in (False, True))
            ):
                # ep2 t3c: a bucket whose trace was dropped after a prefill (above) is re-captured later, at its next
                # use. Without skip_trace_precompile the Generator first runs one EAGER decode step ("Done Compiling
                # Model"), which advances the GDN state and writes KV of every active row once more: buckets 2..Bmax/2
                # (the ones reached by a shrink or a slot move, not right after a prefill) diverged from bucket 1 / Bmax
                # (slot check 1/5, LOG 20:2x). The programs of every width are compiled at warm-up.
                kwargs["skip_trace_precompile"] = True
            # Key the sampling trace by bucket width too: Generator binds it to one logits tensor
            # by identity, and each decode-bucket width has its own.
            for _m in self.model:
                _sm = getattr(_m, "sampling", None)
                if _sm is not None and hasattr(_sm, "set_trace_bucket"):
                    _sm.set_trace_bucket(B)
        if _ht_t0 is not None:
            import time as _time

            from models.demos.blackhole.qwen36.tt import host_timing as _ht

            _ht.add("wrap", _time.perf_counter() - _ht_t0)
        return super().decode_forward(*args, **kwargs)

    def _decode_forward_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        on_device_sampling=False,
        reset_batch=False,
        skip_precompile=False,
    ):
        """ep2 host agent, QWEN36_FAST_INPUTS=1|2 (default 0 = the Generator's own refresh).

        The Generator rewrites all four decode trace inputs every step (this model sets
        _tt_vllm_always_refresh_decode_trace_inputs: rope is host-built). =1: write the token every step, but the
        position + rope and the page table only when they differ from what this trace's input buffers already hold
        (a host-side record of the last written values, per buffer object). =2: also, right after the replay is
        enqueued, build and write the next step's position + rope (every active position + 1) while the card runs;
        the command queue orders those writes after the replay. The next step checks the record and writes again
        only if vLLM asks for something else (new request, prefill, slot changes)."""
        mode = os.environ.get("QWEN36_FAST_INPUTS", "0")
        fast = (
            mode != "0"
            and not on_device_sampling
            and self.data_parallel == 1
            and bool(self.trace_ids_decode[False])
            and getattr(self, "_prev_on_device_sampling", False) is False
        )
        if not fast:
            self._fi_rec = {}
            return super()._decode_forward_trace_text(
                tokens,
                current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                on_device_sampling=on_device_sampling,
                reset_batch=reset_batch,
                skip_precompile=skip_precompile,
            )
        self._prev_on_device_sampling = False
        model = self.model[0]
        ins = self.trace_inputs_decode[False][0]
        rec = getattr(self, "_fi_rec", None)
        if rec is None:
            rec = self._fi_rec = {}

        def held(idx, val):
            r = rec.get(idx)
            return r is not None and r[0] is ins[idx] and r[1].shape == val.shape and torch.equal(r[1], val)

        tok = tokens[0]
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(tok.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), ins[0]
        )
        cp = current_pos[0]
        if isinstance(cp, torch.Tensor):
            pos_vec = cp.to(torch.int32).reshape(-1)
        else:
            pos_vec = torch.full((tok.shape[0],), int(cp), dtype=torch.int32)
        assert pos_vec.shape[0] == tok.shape[0], f"current_pos length {pos_vec.shape[0]} != batch {tok.shape[0]}"
        key = torch.cat([pos_vec, torch.tensor([int(model.rope.rope_delta)], dtype=torch.int32)])
        if not held(1, key):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(pos_vec, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), ins[1]
            )
            ttnn.copy_host_to_device_tensor(model.decode_rope_host(pos_vec), ins[2])
            rec[1] = (ins[1], key)
        pt = page_table[0] if page_table is not None else None
        if pt is not None and ins[3] is not None and not held(3, pt):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), ins[3])
            rec[3] = (ins[3], pt.clone())
        if page_table is not None and (
            self.prev_page_table is None
            or any(not torch.equal(a, b) for a, b in zip(self.prev_page_table, page_table))
        ):
            # keep the Generator's own record consistent (its other paths compare against it)
            self.prev_page_table = tuple(p.clone() for p in page_table)
        for i, trace_id in self.trace_ids_decode[False].items():
            ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)
        if mode == "2":
            nxt = torch.where(pos_vec >= 0, pos_vec + 1, pos_vec)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(nxt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), ins[1])
            ttnn.copy_host_to_device_tensor(model.decode_rope_host(nxt), ins[2])
            rec[1] = (ins[1], torch.cat([nxt, key[-1:]]))
        return self.trace_output_decode[False]

    def process_decode_output_host(self, tt_out, is_tokens=False):
        """ep2 (QWEN36_LEAN_HOST=1, host sampling, one model): skip the generic path's per-step fp32 copies (torch.cat of
        the logits and a full-size torch.ones log-prob placeholder that host sampling never reads)."""
        if (
            os.environ.get("QWEN36_LEAN_HOST") == "1"
            and not is_tokens
            and len(self.model) == 1
            and isinstance(tt_out[0], ttnn.Tensor)
        ):
            logits = self.model[0].process_output_decode(tt_out[0], self.model_args[0].max_batch_size, S=1)
            _stdbg_rows(self, "DECODE", logits=logits, model=self.model[0])
            return logits, torch.ones(1).expand(logits.shape)
        if (
            os.environ.get("QWEN36_LEAN_TUPLE") == "1"
            and os.environ.get("QWEN36_LEAN_HOST") == "1"
            and not is_tokens
            and len(self.model) == 1
            and isinstance(tt_out[0], tuple)
            and isinstance(tt_out[0][0], ttnn.Tensor)
            and tt_out[0][1] is None
        ):
            # ep2 host agent: Generator.read_decode_output hands over [(logits, None)] (a tuple: the model returns
            # (logits, None)), so the lean branch above never ran, in the dev loop and served alike: the generic path
            # built a full-size fp32 torch.ones log-prob placeholder [8, 1, vocab] and two torch.cat copies every step
            # (2.0-2.8 ms per token, measured). Host sampling never reads the placeholder (the runner drops it:
            # finalize_decode `tt_out, _ = tt_out`). Same logits values; the persistent buffer is returned as is.
            logits = self.model[0].process_output_decode(tt_out[0][0], self.model_args[0].max_batch_size, S=1)
            _stdbg_rows(self, "DECODE", logits=logits, model=self.model[0])
            return logits, torch.ones(1).expand(logits.shape)
        return super().process_decode_output_host(tt_out, is_tokens=is_tokens)

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        self._fi_rec = {}
        _pf_release()  # ep2 QWEN36_PF_GCB_MODE=decode: the prefill programs need the global CB L1
        # Capture the chunk-prefill trace + warm the masked-bucket set so requests only replay
        # pre-compiled programs (compile-clobbers-trace fix). Guard name must match the plugin's reset.
        if not enable_trace:
            return
        if getattr(self, "already_warmed_up_prefill", False):
            return
        self.already_warmed_up_prefill = True
        # Size the chunk-trace page table to the full KV cache (not a hardcoded 4096) so served ISL
        # isn't capped; still captures one chunk — just a bigger page-table tensor.
        if kv_cache:
            # Round to a multiple of 32: paged/chunked SDPA needs the page-table stick % 32 == 0.
            num_blocks = math.ceil(int(kv_cache[0][0].shape[0]) / 32) * 32
        else:
            num_blocks = math.ceil(_PREFILL_WARMUP_BUCKET / _BLOCK_SIZE)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        model = self.model[0]
        chunk = _PREFILL_WARMUP_CHUNK
        # Batched serving (max_num_seqs>1): the decode buffers are [B,...], but prefill runs B=1. Bind
        # the PERSISTENT B=1 GDN prefill scratch and capture the chunk trace against IT, so long prompts
        # (>chunk_size) replay the traced chunk-outer path per user instead of the slower eager fallback.
        # The scratch is not freed (prefill_paged_slots rebinds it per request); the batched decode
        # buffers are restored before the decode-trace warmup captures at [B,...].
        batched = model.tp and model.args.max_batch_size > 1
        logger.info(
            f"Starting Qwen prefill warmup: chunk-prefill trace{' (batched, B=1 scratch)' if batched else ''} "
            f"(chunk={chunk}, page_table_blocks={num_blocks})..."
        )
        prev = model._bind_gdn_prefill_scratch() if batched else None
        try:
            model.capture_prefill_trace_chunked(
                self.mesh_device, page_table, chunk_size=chunk, capture_chunk_trace=True
            )
        finally:
            if prev is not None:
                model._unbind_gdn_prefill_scratch(prev)

    def warmup_model_decode(self, *args, **kwargs):
        # Defer to WarmupForwardMixin, which warms the paged-SDPA + GDN decode path at pos 0.
        # Drop stale `non_greedy_decoding_on_device` from the old vLLM plugin; no-op for Qwen.
        kwargs.pop("non_greedy_decoding_on_device", None)
        self._fi_rec = {}
        self._validate_device_sampling_request(kwargs.get("can_sample_on_device", False))
        r = warmup_decode_buckets(self, super().warmup_model_decode, *args, **kwargs)
        if kwargs.get("enable_trace", False) and os.environ.get("QWEN36_GC_FREEZE", "0") == "1":
            # ep2 t3c A33: the warm-up heap goes to the permanent generation (later automatic GC passes skip it;
            # QWEN36_PF_NO_GC removed the explicit collect before each prefill)
            import gc

            gc.collect()
            gc.freeze()
        return r

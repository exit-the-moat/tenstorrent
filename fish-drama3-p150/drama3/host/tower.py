# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drama 3's codebook tower ("audio decoder") and its sampling, in plain torch on the host CPU.

A port of SGLang-Omni's ``fishaudio_drama3/{audio_decoder,decoder_layers,sampling}.py`` (the PyTorch sampler
path, ``sampler=None``). 4 shared + 9 residual dense layers (dim 2048, 16 / 2 heads, head_dim 256, gated
attention, interleaved RoPE on the first 64 dims, Gemma (1 + w) norms), each with its own KV cache along the
sequence. Per frame it takes the backbone's final-norm hidden state, samples the main token (semantic or eos)
with RAS, then code 0 .. 9. FP32 math from the BF16 checkpoint weights.
"""

import math

import torch
import torch.nn.functional as F

from models.demos.blackhole.drama3.tt.weight_mapping import read_tensors

RAS_WINDOW = 16
RAS_TEMPERATURE, RAS_TOP_P, RAS_TOP_K = 1.0, 0.95, 30
CODEBOOK_TOP_K = 30  # the codebook heads draw with top-k at most 30 whatever the request asks
RESIDUAL_CODEBOOK_SIZE = 1024  # the codec's residual codebooks; heads 0..8 have 4096 outputs
MIN_TEMPERATURE = 1e-5


def gemma_norm(x, w, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)


class TowerLayer:
    def __init__(self, sd, p, cfg, embedding_key):
        f = lambda k: sd[p + k].float()  # noqa: E731
        self.emb = f(embedding_key)  # [4096, H]
        self.enorm, self.hnorm = f("enorm.weight"), f("hnorm.weight")
        self.eh = f("eh_proj.weight")  # [H, 2H]
        a = "transformer_layer.attention."
        self.wqkv, self.wo = f(a + "wqkv.weight"), f(a + "wo.weight")  # rows [q | gate | k | v]
        self.qn, self.kn = f(a + "q_norm.weight"), f(a + "k_norm.weight")
        self.in_norm = f("transformer_layer.attention_norm.weight")
        self.post_norm = f("transformer_layer.ffn_norm.weight")
        ff = "transformer_layer.feed_forward."
        self.w1, self.w2, self.w3 = f(ff + "w1.weight"), f(ff + "w2.weight"), f(ff + "w3.weight")
        self.final_norm = f("final_layernorm.weight")
        self.cfg = cfg
        self.k_cache = self.v_cache = None

    def reset(self, max_len):
        c = self.cfg
        self.k_cache = torch.zeros(max_len, c["kv_heads"], c["head_dim"])
        self.v_cache = torch.zeros(max_len, c["kv_heads"], c["head_dim"])

    def rope(self, x, pos):  # x [N, heads, hd], interleaved (GPT-J) on the first rotary_dim dims
        rd = self.cfg["rotary_dim"]
        inv = 1.0 / self.cfg["rope_base"] ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd)
        ang = pos.float()[:, None] * inv[None]
        cos, sin = ang.cos()[:, None], ang.sin()[:, None]
        e, o = x[..., 0:rd:2], x[..., 1:rd:2]
        rot = torch.stack((e * cos - o * sin, o * cos + e * sin), -1).flatten(-2)
        return torch.cat([rot, x[..., rd:]], -1)

    def attention(self, x, pos):
        c = self.cfg
        n, nh, nkv, hd, eps = x.shape[0], c["heads"], c["kv_heads"], c["head_dim"], c["eps"]
        qw, kvw = nh * hd, nkv * hd
        q, g, k, v = (x @ self.wqkv.T).split([qw, qw, kvw, kvw], -1)
        q = self.rope(gemma_norm(q.view(n, nh, hd), self.qn, eps), pos)
        k = self.rope(gemma_norm(k.view(n, nkv, hd), self.kn, eps), pos)
        self.k_cache[pos] = k
        self.v_cache[pos] = v.view(n, nkv, hd)
        end = int(pos.max()) + 1
        K = self.k_cache[:end].repeat_interleave(nh // nkv, 1).transpose(0, 1)  # [nh, end, hd]
        V = self.v_cache[:end].repeat_interleave(nh // nkv, 1).transpose(0, 1)
        mask = torch.arange(end)[None, :] <= pos[:, None]  # causal over the request's own rows
        att = F.scaled_dot_product_attention(q.transpose(0, 1), K, V, attn_mask=mask[None])  # [nh, n, hd]
        att = att.transpose(0, 1).reshape(n, qw) * torch.sigmoid(g)
        return att @ self.wo.T

    def __call__(self, h, code_ids, label_is_semantic, shared_input, pos):
        eps = self.cfg["eps"]
        e = torch.where(label_is_semantic[:, None], self.emb[code_ids.clamp(0, self.emb.shape[0] - 1)], 0.0)
        if shared_input is not None:
            e = e + shared_input
        mixed = torch.cat([gemma_norm(e, self.enorm, eps), gemma_norm(h, self.hnorm, eps)], -1) @ self.eh.T
        hidden = mixed + self.attention(gemma_norm(mixed, self.in_norm, eps), pos)
        ff_in = gemma_norm(hidden, self.post_norm, eps)
        mlp = (F.silu(ff_in @ self.w1.T) * (ff_in @ self.w3.T)) @ self.w2.T
        return gemma_norm(mlp + hidden, self.final_norm, eps)


def sample_top_k_top_p(logits, temperature, top_k, top_p, gen, keep_crossing=False):
    """One token from a 1-D logit row: Fish's sample_top_k_top_p (sampler=None path)."""
    scaled = logits.float() / max(temperature, MIN_TEMPERATURE)
    sorted_logits, ids = torch.sort(scaled, descending=True)
    cum = torch.cumsum(torch.softmax(sorted_logits, -1), -1)
    drop = cum > top_p
    if keep_crossing:
        drop[1:] = drop[:-1].clone()
    drop |= torch.arange(sorted_logits.numel()) >= top_k
    drop[0] = False
    probs = torch.softmax(sorted_logits.masked_fill(drop, float("-inf")), -1)
    noise = torch.empty_like(probs).exponential_(1, generator=gen)
    return int(ids[torch.argmax(probs / noise)])


class CodebookTower:
    def __init__(self, model_path, wmap, full_config):
        dc = full_config["audio_decoder_config"]
        self.cfg = dict(
            heads=dc["n_head"],
            kv_heads=dc["n_local_heads"],
            head_dim=dc["head_dim"],
            eps=dc["norm_eps"],
            rope_base=dc["rope_base"],
            rotary_dim=int(dc["head_dim"] * dc["rotary_percent"]),
        )
        assert dc["rotary_interleaved"] and dc["vq_backbone_codebooks"] == 1 and dc["n_layer"] == 4
        self.n_cb, self.cb_size = dc["num_codebooks"], dc["vocab_size"]
        self.sem0, self.sem1 = full_config["semantic_start_token_id"], full_config["semantic_end_token_id"]
        self.eos = full_config["eos_token_id"]
        keys = [k for k in wmap if k.startswith(("audio_decoder.", "audio_decoder_mtp.", "audio_decoder_mtp_shared."))]
        sd = read_tensors(model_path, wmap, keys)
        self.codebook_embeddings = sd["audio_decoder.codebook_embeddings.weight"]  # bf16, also for the backbone input
        self.cb_f32 = self.codebook_embeddings.float()
        self.shared = [
            TowerLayer(sd, f"audio_decoder_mtp_shared.layers.{i}.", self.cfg, "semantic_embedding.weight")
            for i in range(dc["shared_enrichment_layers"])
        ]
        self.residual = [
            TowerLayer(sd, f"audio_decoder_mtp.layers.{i}.", self.cfg, "codebook_embedding.weight")
            for i in range(self.n_cb - 1)
        ]
        self.heads = [sd[f"audio_decoder_mtp.heads.{i}.weight"].float() for i in range(self.n_cb - 1)]
        self.candidates = torch.tensor(sorted({self.eos, *range(self.sem0, self.sem1 + 1)}))
        del sd

    def reset(self, max_len):
        for layer in self.shared + self.residual:
            layer.reset(max_len)

    def is_semantic(self, ids):
        return (ids >= self.sem0) & (ids <= self.sem1)

    def shared_input(self, frames):
        """Codes 2..9 of semantic input frames, summed from codebook_embeddings (rows k*4096 + code)."""
        rows = frames[:, 3 : 1 + self.n_cb].clamp(0, self.cb_size - 1) + torch.arange(2, self.n_cb) * self.cb_size
        s = self.cb_f32[rows].sum(1)
        return torch.where(self.is_semantic(frames[:, 0])[:, None], s, 0.0)

    @torch.no_grad()
    def prefill(self, hidden, positions, input_frames, next_frames, collect=False):
        """Fill the tower caches of prompt rows (row p is labelled with frame p + 1). collect: per-layer outputs."""
        lab = self.is_semantic(next_frames[:, 0])
        h = hidden.float()
        si = self.shared_input(input_frames)
        outs = []
        for i, layer in enumerate(self.shared):
            h = layer(h, next_frames[:, 1], lab, si if i == 0 else None, positions)
            outs.append(h) if collect else None
        for r, layer in enumerate(self.residual):
            h = layer(h, next_frames[:, 1 + r], lab, None, positions)
            outs.append(h) if collect else None
        return outs

    @torch.no_grad()
    def step(self, cand_logits, hidden, pos, input_frame, window, gen, temperature=1.0, top_p=0.9, top_k=None, collect=False):
        """Sample the next frame [11] from the backbone row at `pos` (cand_logits over self.candidates)."""
        top_k = top_k or len(self.candidates)
        first = int(self.candidates[sample_top_k_top_p(cand_logits, temperature, top_k, top_p, gen, keep_crossing=True)])
        ras = int(self.candidates[sample_top_k_top_p(cand_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, gen, keep_crossing=True)])
        main = ras if (first in window and self.sem0 <= first < self.sem0 + self.cb_size) else first
        code0 = min(max(main - self.sem0, 0), self.cb_size - 1)
        p = torch.tensor([pos])
        lab = self.is_semantic(torch.tensor([main]))
        h = hidden.float().reshape(1, -1)
        si = self.shared_input(input_frame.reshape(1, -1))
        c0 = torch.tensor([code0])
        outs, head_logits = [], []
        for i, layer in enumerate(self.shared):
            h = layer(h, c0, lab, si if i == 0 else None, p)
            outs.append(h[0]) if collect else None
        codes = [code0]
        k = min(top_k, CODEBOOK_TOP_K)
        for r, (layer, head) in enumerate(zip(self.residual, self.heads)):
            h = layer(h, torch.tensor([codes[-1]]), lab, None, p)
            logits = (h @ head.T)[0]
            logits[RESIDUAL_CODEBOOK_SIZE:] = float("-inf")  # codes the codec cannot decode
            if collect:
                outs.append(h[0])
                head_logits.append(logits[:RESIDUAL_CODEBOOK_SIZE].clone())
            codes.append(sample_top_k_top_p(logits, temperature, k, top_p, gen))
        frame = torch.tensor([main, *codes], dtype=torch.long)
        return (frame, first != main, outs, head_logits) if collect else (frame, first != main)

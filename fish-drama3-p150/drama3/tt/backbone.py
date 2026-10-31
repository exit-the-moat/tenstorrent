# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drama 3 backbone on one P150 for generation: host-built input embeddings in, final-norm hidden states and
the 4,097 candidate logits (eos + semantic band) out.

Uses the qwen36 single-device modules from the warm tensor cache (see device_check.py). Prefill runs the
largest multiple of 32 of the prompt (qwen36's MoE prefill needs it); the remaining prompt rows and every
generated frame go through single-row decode on the non-paged ("concat") attention cache and the GDN
recurrent state, so no padding row ever enters the KV cache or the recurrent state.
"""

import time

import torch

import ttnn
from models.demos.blackhole.drama3.tt.device_check import build_layer, dram_used_gb, to_host
from models.demos.blackhole.drama3.tt.weight_mapping import load_drama3_state_dict
from models.tt_transformers.tt.common import Mode

TILE = 32


def _upload(mesh, x):
    """[N, H] fp32 host -> [1, N, H] bf16 TILE in DRAM."""
    return ttnn.from_torch(
        x.reshape(1, x.shape[0], x.shape[-1]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _rank3(y):
    return ttnn.reshape(y, (1, y.shape[-2], y.shape[-1])) if len(y.shape) == 4 else y


class Drama3Backbone:
    def __init__(self, mesh, args, cache, wmap, candidates, log=print):
        from models.common.rmsnorm import RMSNorm
        from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup

        self.mesh, self.args = mesh, args
        t0 = time.time()
        self.layers = [build_layer(mesh, args, i, cache, wmap)[0] for i in range(args.n_layers)]
        sd = load_drama3_state_dict(args.CKPT_DIR, norm=True, lm_head=True, wmap=wmap)
        self.norm = RMSNorm(
            device=mesh,
            dim=args.dim,
            state_dict=sd,
            weight_key="norm",
            weight_cache_path=cache,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
        )
        self.n_cand = len(candidates)
        rows = torch.zeros(((self.n_cand + TILE - 1) // TILE) * TILE, args.dim, dtype=torch.bfloat16)
        rows[: self.n_cand] = sd["output.weight"][candidates]
        del sd
        self.head = ttnn.as_tensor(
            rows,
            preprocess=lambda t: t.T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache / "drama3.output.candidates.bf16",
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.head_ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.rope = Qwen36RoPESetup(mesh, args)
        self.build_s = time.time() - t0
        log(f"backbone resident: DRAM {dram_used_gb(mesh)[0]:.2f} GB, built in {self.build_s:.0f}s")

    def reset(self):
        for layer in self.layers:
            if layer.is_full_attention:
                layer.attention.reset_cache()
            else:
                layer.attention.reset_state(1)

    def _out(self, x, n):
        xn = self.norm(x, mode=Mode.PREFILL)
        logits = ttnn.linear(xn, self.head, dtype=ttnn.float32, compute_kernel_config=self.head_ckc)
        normed = to_host(xn, n)
        lg = to_host(logits, n)[:, : self.n_cand]
        ttnn.deallocate(xn)
        ttnn.deallocate(logits)
        return normed, lg

    def prefill(self, emb, collect=False):
        """emb [P, H] fp32, P % 32 == 0 -> (final-norm [P, H], candidate logits [P, n_cand], per-layer list)."""
        P = emb.shape[0]
        assert P % TILE == 0 and P > 0
        self.rope.build_request_rope(torch.zeros(1, P, dtype=torch.long))  # text-style 1D positions
        cos, sin = self.rope.get_prefill_rot_mats(0, P)
        x = _upload(self.mesh, emb)
        per_layer = []
        for layer in self.layers:
            y = _rank3(layer.forward(x, cos=cos, sin=sin, mode="prefill"))
            ttnn.deallocate(x)
            x = y
            if collect:
                per_layer.append(to_host(x, P))
        normed, lg = self._out(x, P)
        ttnn.deallocate(x)
        return normed, lg, per_layer

    def decode(self, emb_row, pos, collect=False):
        """One row at sequence position pos -> (final-norm [H], candidate logits [n_cand], per-layer list)."""
        cos, sin = self.rope.get_rot_mats(torch.full((1, 1), pos, dtype=torch.long))
        n_kv = self.args.n_kv_heads
        cur = ttnn.from_torch(
            torch.full((n_kv,), pos, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh
        )
        x = _upload(self.mesh, emb_row.reshape(1, -1))
        per_layer = []
        for layer in self.layers:
            if layer.is_full_attention:
                y = layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=cur)
            else:
                y = layer.forward(x, mode="decode")
            y = _rank3(y)
            ttnn.deallocate(x)
            x = y
            if collect:
                per_layer.append(to_host(x, 1)[0])
        normed, lg = self._out(x, 1)
        ttnn.deallocate(x)
        ttnn.deallocate(cur)
        return normed[0], lg[0], per_layer

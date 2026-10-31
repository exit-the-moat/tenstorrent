# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port5 Track B: the fused one-token GDN decode (tt/gdn_fused_decode.py) on ONE layer vs the ttnn op chain and vs
an fp32 torch reference built from the same device inputs (mega in-projection row, conv partial sums, state), after
a real voice-prompt prefill; then the traced device time of both mixers.

    (fast env) python -m models.demos.blackhole.drama3.tests.check_gdn_fused [--layer 0] [--steps 8] [--out f.json]
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)
MODEL = "/models/drama-3"


def h(t):
    return ttnn.to_torch(t).float()


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def torch_ref(mega, Q, taps, S, aneg, dtb, onw, eps):
    """fp32 reference of the fused body: mega [12352], Q 3 x [8192], taps 4 x [8192], S [32, 128, 128]."""
    x = mega[:8192]
    a, b, z = mega[8192:8224], mega[8224:8256], mega[8256:8256 + 4096]
    y = F.silu(Q[0] + taps[3] * x)
    Qn = [Q[1] + taps[2] * x, Q[2] + taps[1] * x, taps[0] * x]
    q = y[:2048].reshape(16, 128).repeat_interleave(2, 0)
    k = y[2048:4096].reshape(16, 128).repeat_interleave(2, 0)
    v = y[4096:].reshape(32, 128)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128**-0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(b)
    decay = torch.exp(aneg * F.softplus(a + dtb))
    sd = S * decay[:, None, None]
    u = beta[:, None] * (v - torch.einsum("hk,hkv->hv", k, sd))
    Sn = sd + k[:, :, None] * u[:, None, :]
    o = torch.einsum("hk,hkv->hv", q, Sn)
    o = o / torch.sqrt((o * o).mean(-1, keepdim=True) + eps) * onw
    o = o * F.silu(z.reshape(32, 128))
    return o.reshape(-1), Sn, Qn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8, help="consecutive decode steps compared (state carried)")
    ap.add_argument("--out", default="/opt/ep0/port5/check_gdn_fused.json")
    a = ap.parse_args()
    from models.demos.blackhole.drama3.host.prompt import FrameEmbedder
    from models.demos.blackhole.drama3.tt.backbone_paged import PagedBackbone
    from models.demos.blackhole.drama3.tt.gdn_fused_decode import FusedGDNDecode
    from models.demos.blackhole.drama3.tt.weight_mapping import read_tensors, weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.tt_transformers.tt.common import Mode

    cfg = json.load(open(f"{MODEL}/config.json"))
    wmap = weight_map(MODEL)
    keys = ["model.language_model.embed_tokens.weight", "audio_decoder.codebook_embeddings.weight"]
    t = read_tensors(MODEL, wmap, keys)
    E = FrameEmbedder(t[keys[0]], t[keys[1]], cfg["semantic_start_token_id"], cfg["semantic_end_token_id"])(torch.load("/opt/ep0/day3/close.rows.pt"))
    cand = torch.tensor(sorted({cfg["eos_token_id"], *range(cfg["semantic_start_token_id"], cfg["semantic_end_token_id"] + 1)}))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE, trace_region_size=256 << 20)
    res = dict(layer=a.layer)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        bb = PagedBackbone(mesh, args, args.weight_cache_path(), wmap, cand)
        bb.reset()
        T = E.shape[0] - a.steps
        bb.prefill(E[:T])
        layer = bb.layers[a.layer]
        assert not layer.is_full_attention
        dn, w = layer.attention, layer.attention.weights
        ckc = dn.compute_kernel_config_decode
        fused = FusedGDNDecode(dn, mesh)
        Q = list(dn.split_conv_state)
        snap = lambda: (ttnn.clone(dn.recurrent_state), [ttnn.clone(q) for q in Q])  # noqa: E731

        def restore(s):
            ttnn.copy(s[0], dn.recurrent_state)
            for q, src in zip(Q, s[1]):
                ttnn.copy(src, q)

        host_c = dict(
            taps=[h(tp).flatten()[:8192] for tp in w.fused_conv_weight_taps],
            aneg=h(w.A_neg).flatten()[:32],
            dtb=h(w.dt_bias).flatten()[:32],
            onw=h(w.o_norm_weight).flatten()[:128],
        )
        start = snap()
        steps = []
        # chain A: the ttnn op chain over the steps; chain B: the fused kernel from the same start state
        outs = {}
        for name in ("chain", "fused"):
            restore(start)
            outs[name] = []
            for s_ in range(a.steps):
                bb._stage(E[T + s_], T + s_)
                xn = layer.attention_norm(bb.x_buf, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
                mega = h(ttnn.linear(xn, w.mega_fused_weight, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ckc)).flatten()[:12352]
                S0, Q0 = h(dn.recurrent_state).reshape(32, 128, 128), [h(q).flatten()[:8192] for q in Q]
                if name == "chain":
                    out = dn.forward(xn, mode="recurrent")
                    o_pre = None
                else:
                    out = fused.forward(xn, ckc)
                    o_pre = h(fused.out).flatten()[:4096]
                ttnn.synchronize_device(mesh)
                ref_o, ref_S, ref_Q = torch_ref(mega, Q0, host_c["taps"], S0, host_c["aneg"], host_c["dtb"], host_c["onw"], dn.norm_eps)
                S1, Q1 = h(dn.recurrent_state).reshape(32, 128, 128), [h(q).flatten()[:8192] for q in Q]
                outs[name].append(dict(out=h(out).flatten()[:2048], S=S1, Q=Q1, ref_S=ref_S, ref_Q=ref_Q, ref_o=ref_o, o_pre=o_pre))
        for s_ in range(a.steps):
            c, f = outs["chain"][s_], outs["fused"][s_]
            r = dict(
                step=s_,
                out_pcc_fused_vs_chain=round(pcc(f["out"], c["out"]), 6),
                state_pcc_chain_vs_ref=round(pcc(c["S"], c["ref_S"]), 6),
                state_pcc_fused_vs_ref=round(pcc(f["S"], f["ref_S"]), 6),
                state_pcc_fused_vs_chain=round(pcc(f["S"], c["S"]), 6),
                conv_pcc_fused_vs_ref=round(min(pcc(x_, y_) for x_, y_ in zip(f["Q"], f["ref_Q"])), 6),
                conv_pcc_chain_vs_ref=round(min(pcc(x_, y_) for x_, y_ in zip(c["Q"], c["ref_Q"])), 6),
                o_pcc_fused_vs_ref=round(pcc(f["o_pre"], f["ref_o"]), 6),
                o_maxabs_fused_vs_ref=round(float((f["o_pre"] - f["ref_o"]).abs().max()), 5),
            )
            steps.append(r)
            print(json.dumps(r), flush=True)
        res["steps"] = steps

        # traced device time per mixer (norm + mixer, as profile_decode's gdn_mixer)
        bb._stage(E[-1], E.shape[0])
        def mixer(use_fused):
            def f_():
                xn = layer.attention_norm(bb.x_buf, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
                return fused.forward(xn, ckc) if use_fused else dn.forward(xn, mode="recurrent")
            return f_

        for name, fn in (("chain", mixer(False)), ("fused", mixer(True))):
            fn()
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            for _ in range(5):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(100):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            res[f"{name}_mixer_ms"] = round((time.time() - t0) / 100 * 1e3, 4)
            ttnn.release_trace(mesh, tid)
            print(name, "mixer traced", res[f"{name}_mixer_ms"], "ms", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise

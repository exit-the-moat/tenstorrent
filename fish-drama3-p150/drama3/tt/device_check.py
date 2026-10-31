# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drama 3 backbone on one P150 ((1,1) mesh) with the qwen36 modules, checked against the golden CPU reference.

Two modes:
  --layers 0,3,...   teacher-forced: each layer is built alone (its weights converted into the tensor
                     cache on a cache miss), fed the golden input of that layer, compared with the golden
                     output, then freed. This is also how the cache is filled, one layer at a time.
  --chain            all 40 layers resident at once, embedding -> layers -> final norm -> LM head over Fish's
                     4,097 candidate rows; every layer output and the logits compared with the golden set.

Prompt: the golden 42 tokens, zero-padded to 64 rows; only the 42 real rows are compared.

    HF_MODEL=/opt/ep0/drama3_shim TT_CACHE_PATH=/opt/ep0/cache/drama3 \
        python -m models.demos.blackhole.drama3.tt.device_check --layers 0
"""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.drama3.tt.weight_mapping import load_drama3_state_dict, weight_map

GOLDEN = Path(os.environ.get("DRAMA3_GOLDEN", "/opt/ep0/golden"))
PAD_TO = 64
HEAD_DTYPE = os.environ.get("DRAMA3_HEAD_DTYPE", "bf16")  # candidate LM head: bf16 (17 MB) or bfp8


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def golden(name):
    return torch.load(GOLDEN / name)


def dram_used_gb(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return v.num_banks * v.total_bytes_allocated_per_bank / 1e9, v.num_banks * v.total_bytes_per_bank / 1e9


def to_device(mesh, h):
    """[S, H] fp32 host -> [1, PAD_TO, H] bf16 TILE in DRAM (the single-device layer layout), zero rows past S."""
    x = torch.zeros(1, PAD_TO, h.shape[-1])
    x[0, : h.shape[0]] = h
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def to_host(t, n):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(-1, t.shape[-1])[:n]


def reset(layer):
    if layer.is_full_attention:
        layer.attention.reset_cache()
    else:
        layer.attention.reset_state(1)


def build_layer(mesh, args, i, cache, wmap):
    from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer

    t0 = time.time()
    # Warm cache: the routed experts load from their cache files, so they are not read from the checkpoint.
    # Day 5: the gate/up dtype is per layer (QWEN36_EXPERT_GU_BFP8_LAYERS); a BFP8 layer needs its BFP8 files
    from models.demos.blackhole.qwen36.tt.moe.weights import gate_up_dtype_for_cache

    gu = "BFLOAT8_B" if gate_up_dtype_for_cache(cache / f"layers.{i}") == ttnn.bfloat8_b else "BFLOAT4_B"
    tags = {"gate_proj": gu, "up_proj": gu, "down_proj": "BFLOAT8_B"}
    warm = all(any((cache / f"layers.{i}").glob(f"moe.experts.{n}_dtype_{d}_*")) for n, d in tags.items())
    sd = load_drama3_state_dict(args.CKPT_DIR, [i], wmap=wmap, experts=not warm)
    t_read = time.time() - t0
    layer = Qwen36DecoderLayer(mesh, args, sd, i, cache)
    del sd
    return layer, t_read, time.time() - t0 - t_read


def free_layer(layer):
    """Deallocate every device tensor reachable from the layer's modules."""
    seen = set()

    def walk(o, depth=0):
        if id(o) in seen or depth > 6:
            return
        seen.add(id(o))
        if isinstance(o, ttnn.Tensor):
            if o.storage_type() == ttnn.StorageType.DEVICE and o.is_allocated():
                o.deallocate(True)
            return
        if isinstance(o, (list, tuple)):
            for x in o:
                walk(x, depth + 1)
        elif isinstance(o, dict):
            for x in o.values():
                walk(x, depth + 1)
        elif hasattr(o, "__dict__") and type(o).__module__.startswith(("models.", "dataclasses")):
            for x in vars(o).values():
                walk(x, depth + 1)
        elif hasattr(o, "__dataclass_fields__"):
            for f in o.__dataclass_fields__:
                walk(getattr(o, f), depth + 1)

    walk(layer)


def run_layer(layer, x, cos, sin):
    reset(layer)
    y = layer.forward(x, cos=cos, sin=sin, mode="prefill")
    # the residual add broadcasts to the 4D [1, 1, S, H] the framework RMSNorm returns; the
    # single-device GDN / attention layers take [1, S, H]
    return ttnn.reshape(y, (1, y.shape[-2], y.shape[-1])) if len(y.shape) == 4 else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="")
    ap.add_argument("--chain", action="store_true")
    ap.add_argument("--out", default="/opt/ep0/day2")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_PREFILL_CHUNK_SIZE", "32")

    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
    from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        cache = args.weight_cache_path()
        log(f"mesh {tuple(mesh.shape)}, cache {cache}, model_name {args.model_name}, DRAM {dram_used_gb(mesh)}")
        manifest = json.load(open(GOLDEN / "manifest.json"))
        ids = manifest["prompt_ids"]
        n = len(ids)
        wmap = weight_map(args.CKPT_DIR)
        rope = Qwen36RoPESetup(mesh, args)
        rope.build_request_rope(torch.tensor([ids]))  # text only: plain 1D RoPE
        cos, sin = rope.get_prefill_rot_mats(0, PAD_TO)

        results = []
        if a.layers:
            for i in [int(s) for s in a.layers.split(",")]:
                layer, t_read, t_build = build_layer(mesh, args, i, cache, wmap)
                used = dram_used_gb(mesh)[0]
                x_in = golden("embedding.pt" if i == 0 else f"layer_{i - 1:02d}.pt")["hidden"]
                ref = golden(f"layer_{i:02d}.pt")["hidden"]
                t0 = time.time()
                y = run_layer(layer, to_device(mesh, x_in), cos, sin)
                got = to_host(y, n)
                t_run = time.time() - t0
                p = pcc(got, ref)
                # the layer's own contribution (output - input), a stricter view than the residual
                p_delta = pcc(got - x_in, ref - x_in)
                rows = [pcc(got[r], ref[r]) for r in range(n)]
                rec = dict(
                    layer=i,
                    type=args.attention_type_list[i],
                    pcc=round(p, 6),
                    pcc_delta=round(p_delta, 6),
                    pcc_row_min=round(min(rows), 6),
                    max_abs_err=round((got - ref).abs().max().item(), 3),
                    read_s=round(t_read, 1),
                    build_s=round(t_build, 1),
                    run_s=round(t_run, 2),
                    dram_with_layer_gb=round(used, 3),
                )
                results.append(rec)
                log(json.dumps(rec))
                ttnn.deallocate(y)
                free_layer(layer)
                del layer
                __import__('gc').collect()
                os.sync()
            with open(out / "teacher_forced.jsonl", "a") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

        if a.chain:
            chain(mesh, args, cache, wmap, ids, cos, sin, manifest, out)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


def chain(mesh, args, cache, wmap, ids, cos, sin, manifest, out):
    from models.common.rmsnorm import RMSNorm

    n = len(ids)
    t0 = time.time()
    layers = []
    for i in range(args.n_layers):
        layer, t_read, t_build = build_layer(mesh, args, i, cache, wmap)
        layers.append(layer)
        log(f"built layer {i} read {t_read:.1f}s build {t_build:.1f}s DRAM {dram_used_gb(mesh)[0]:.2f} GB")
    t_layers = time.time() - t0

    sd = load_drama3_state_dict(args.CKPT_DIR, embed=True, norm=True, lm_head=True, wmap=wmap)
    emb = sd["tok_embeddings.weight"]
    x0 = emb[torch.tensor(ids)].float()  # host embedding (Drama 3 needs host-built inputs for semantic rows)
    norm = RMSNorm(
        device=mesh,
        dim=args.dim,
        state_dict=sd,
        weight_key="norm",
        weight_cache_path=cache,
        weight_dtype=ttnn.bfloat16,
        add_unit_offset=True,
        eps=args.norm_eps,
    )
    sem0, sem1 = manifest["semantic_candidates"].split("semantic ")[1].split(" ")[0].split("..")
    eos = int(manifest["semantic_candidates"].split("eos ")[1].split(" ")[0])
    cand = torch.tensor(sorted({eos, *range(int(sem0), int(sem1) + 1)}))
    head_rows = torch.zeros(((len(cand) + 31) // 32) * 32, args.dim, dtype=torch.bfloat16)
    head_rows[: len(cand)] = sd["output.weight"][cand]
    del sd, emb
    head_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[HEAD_DTYPE]
    lm_head = ttnn.as_tensor(
        head_rows,
        preprocess=lambda t: t.T.contiguous(),
        dtype=head_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache / f"drama3.output.candidates.{HEAD_DTYPE}",
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    head_ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    def head(t):
        return ttnn.linear(t, lm_head, dtype=ttnn.float32, compute_kernel_config=head_ckc)

    used, total = dram_used_gb(mesh)
    log(f"all layers + norm + candidate head resident: DRAM {used:.2f} / {total:.2f} GB, built in {t_layers:.0f}s")

    x = to_device(mesh, x0)
    per_layer = []
    saved = {}
    t0 = time.time()
    for i, layer in enumerate(layers):
        y = run_layer(layer, x, cos, sin)
        ttnn.deallocate(x)
        x = y
        got = to_host(x, n)
        ref = golden(f"layer_{i:02d}.pt")["hidden"]
        saved[f"layer_{i:02d}"] = got.clone()
        per_layer.append(
            dict(
                layer=i,
                type=args.attention_type_list[i],
                pcc=round(pcc(got, ref), 6),
                pcc_last_row=round(pcc(got[n - 1], ref[n - 1]), 6),
            )
        )
        log(json.dumps(per_layer[-1]))
    from models.tt_transformers.tt.common import Mode

    xn = norm(x, mode=Mode.PREFILL)
    logits = head(xn)
    t_fwd = time.time() - t0
    normed = to_host(xn, n)
    lg = to_host(logits, n)[:, : len(cand)]
    ref_norm = golden("final_norm.pt")["hidden"]
    ref_last = golden("logits_last.pt")["logits"][cand]
    last = lg[n - 1]
    top_dev = [int(cand[j]) for j in torch.topk(last, 5).indices]
    top_ref = [int(cand[j]) for j in torch.topk(ref_last, 5).indices]

    # Separate the two error sources at the last position:
    # the device head on the golden final-norm rows (head error alone), and the reference FP32 head on
    # the device final-norm rows (backbone + final-norm error alone).
    head_only = to_host(head(to_device(mesh, ref_norm)), n)[n - 1, : len(cand)]
    w_cand = load_drama3_state_dict(args.CKPT_DIR, lm_head=True, wmap=wmap)["output.weight"][cand].float()
    backbone_only = normed[n - 1] @ w_cand.T
    saved.update(final_norm=normed.clone(), logits_last=last.clone(), head_only=head_only, backbone_only=backbone_only)
    torch.save(saved, out / f"device_hidden_{HEAD_DTYPE}.pt")

    def top5(v):
        return [int(cand[j]) for j in torch.topk(v, 5).indices]

    diag = dict(
        head_dtype=HEAD_DTYPE,
        final_norm_pcc_last_row=round(pcc(normed[n - 1], ref_norm[n - 1]), 6),
        head_only_pcc=round(pcc(head_only, ref_last), 6),
        head_only_top5=top5(head_only),
        backbone_only_pcc=round(pcc(backbone_only, ref_last), 6),
        backbone_only_top5=top5(backbone_only),
    )
    log(json.dumps(diag))
    res = dict(
        per_layer=per_layer,
        diagnostics=diag,
        final_norm_pcc=round(pcc(normed, ref_norm), 6),
        last_logits_pcc_candidates=round(pcc(last, ref_last), 6),
        top5_device=top_dev,
        top5_reference=top_ref,
        top1_match=top_dev[0] == top_ref[0],
        top5_overlap=len(set(top_dev) & set(top_ref)),
        dram_used_gb=round(used, 3),
        dram_total_gb=round(total, 3),
        build_s=round(t_layers, 1),
        forward_s_incl_readback=round(t_fwd, 2),
    )
    json.dump(res, open(out / f"chain_{HEAD_DTYPE}.json", "w"), indent=1)
    log(json.dumps({k: v for k, v in res.items() if k != "per_layer"}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise

#!/usr/bin/env python3
"""DFlash2 speculative-decoding server for Qwen3.8-27B on Tenstorrent (Episode 2 night 1, agent "dflash").

OpenAI-compatible subset, so the bench, check/exact.py and check/gsm8k.py run unchanged:
  POST /v1/chat/completions (stream + non-stream, usage, min_tokens, max_tokens, ignore_eos, chat_template_kwargs)
  POST /tokenize, POST /detokenize, GET /v1/models
One request at a time (batch 1); others wait in the queue. Greedy only (temperature must be 0).

Admin (bind the port to 127.0.0.1 on the host): POST /dflash/exec {"file": path} runs a Python file in the worker
thread with `S` (this server's state) in scope, so code can be fixed without reloading the 27B model (~14 min).
GET /dflash/stats returns the last requests' acceptance stats (also appended to the DF_STATS file).

Env: MESH_DEVICE (P150x2 | P150), DF_MODE (spec | plain), DF_TRACE (1 = trace the verify pass), DF_SDPA_ROWS,
DF_POOL (KV tokens, default 65536), DF_LEN (max context, default 65536), DF_LAYERS (debug: layer subset, no drafter),
DF_NO_DRAFTER=1, DF_PORT (8080 in the container).
"""
import io
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import faulthandler

import torch

faulthandler.enable()
import signal as _sg

faulthandler.register(_sg.SIGUSR1, all_threads=True)  # ep2 ms1: kill -USR1 dumps every thread's Python stack  # a segfault in a ttnn call prints the Python stack (the 03:52 crash had none)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import df_best  # noqa: E402  QWEN38_BEST=2card|1card: the whole gated switch set, before anything reads one (df_best.py)
df_best.apply()
import df_auto  # noqa: E402  ep2 dfa: DF_AUTO=1 fills the gated multi-stream set (before NS and the qwen36 import)

df_auto.apply()
# ep2 dfpc DF_PREFIX_CACHE=1 (default off): prefix caching for the multi-stream server (df_pcache.py). The model side is the pcx GDN snapshot store alone (QWEN36_PCX_STORE; not QWEN36_PREFIX_CACHE, which
# also switches vLLM's KV sub-blocks) with saves at the last two trace boundaries (QWEN36_PCX_SAVE_PREV).
PCACHE = os.environ.get("DF_PREFIX_CACHE", "0") == "1"
if PCACHE:
    os.environ.setdefault("QWEN36_PCX_STORE", "1")
    os.environ.setdefault("QWEN36_PCX_SAVE_PREV", "1")
    _ns = int(os.environ.get("DF_STREAMS", "1"))
    os.environ.setdefault("QWEN36_PREFIX_CACHE_SLOTS", os.environ.get("DF_PCACHE_SLOTS", os.environ.get("DF_PREFIX_CACHE_SLOTS", str(2 * _ns))))

HF = os.environ.get("HF_MODEL", "/models/qwen38-bf16")
DRAFT = os.environ.get("DF_DRAFT", "/models/qwen38-dflash2")
STATS = os.environ.get("DF_STATS", "/tmp/dflash-stats.jsonl")
T = 8
NS = int(os.environ.get("DF_STREAMS", "1"))  # ep2 ms1: parallel DFlash2 streams (1 = the single-stream server)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class State:
    pass


S = State()
S.ready = False
S.status = "starting"
S.jobs = queue.Queue()
S.last_stats = []
S.mode = os.environ.get("DF_MODE", "spec")


# ---------------------------------------------------------------------------- model bring-up
def bring_up():
    import ttnn
    from transformers import AutoTokenizer

    S.ttnn = ttnn
    S.tok = AutoTokenizer.from_pretrained(HF)
    S.chat_template = open(os.path.join(HF, "chat_template.jinja")).read()
    gc = json.load(open(os.path.join(HF, "generation_config.json")))
    eos = gc.get("eos_token_id")
    S.eos = set(eos if isinstance(eos, list) else [eos])
    if S.tok.eos_token_id is not None:
        S.eos.add(S.tok.eos_token_id)
    log("eos ids", sorted(S.eos))
    if os.environ.get("DF_FAKE") == "1":  # HTTP/streaming test without a device
        S.max_len = int(os.environ.get("DF_LEN", "65536"))
        S.fake = S.tok.encode(" the quick brown fox jumps over the lazy dog.", add_special_tokens=False)
        S.ready, S.status = True, "ready (FAKE)"
        log("READY (FAKE)")
        return

    from vllm_tt_plugin.worker import open_mesh_device

    mesh_env = os.environ.get("MESH_DEVICE", "P150x2")
    tt_config = {"trace_region_size": 1 << 30, "l1_small_size": 24576}
    if mesh_env == "P150x2":
        tt_config["fabric_config"] = "FABRIC_1D"
    S.status = "opening mesh"
    S.mesh = open_mesh_device(tt_config, "all")
    g = S.mesh.compute_with_storage_grid_size()
    log(f"mesh open: {S.mesh.get_num_devices()} devices, grid {g.x}x{g.y}")

    from models.demos.blackhole.qwen36.tt.common import create_tt_model
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

    S.status = "loading model"
    layers = os.environ.get("DF_LAYERS")
    layers = [int(x) for x in layers.split(",")] if layers else None
    S.max_len = int(os.environ.get("DF_LEN", "65536"))
    t0 = time.time()
    one = S.mesh.get_num_devices() == 1 and os.environ.get("QWEN36_TP1") != "1"  # df1c TP1: full server; 1-card mode: drafter tests only (the verify pass is the TP code path)
    # r64d Stage B: DF_MS_CHAIN=4 -> model at CHAIN*T rows (32); NCHAIN chains of streams share it, each chain a
    # separate <=32-row verify pass (above 32 rows the model is not row-exact: binv 33 = 0/5, GSM8K 64w 0/60).
    _ch = int(os.environ.get("DF_MS_CHAIN", "0"))
    _kc = int(os.environ.get("DF_SPEC_KCAP", "0") or 0)  # r64t: model rows = NS * the per-step depth cap
    # r64t: the model stays at 64 rows: the batched drafter's all-reduce needs 8 rows per stream on the residual grid
    # (64 rows), while every verify pass is NS * k <= 32 rows (a narrower pass in a 64-row model is the proven r64g
    # shape: the 32-row subset verifiers ran at the same cost there).
    _spec = max(NS * max(1, min(_kc, 32 // max(1, NS), 8)), min(64, 8 * NS)) if _kc else 0
    _mb = 1 if one else (_ch * T if (_ch > 0 and _ch < NS) else (_spec or T * NS))
    log(f"model max_batch_size {_mb} (streams {NS}, chain {_ch}, spec kcap {_kc})")
    S.args, S.model, _ = create_tt_model(S.mesh, max_batch_size=_mb, max_seq_len=131072 if not one else 16384,
                                         layer_indices=layers)
    log(f"model loaded in {time.time() - t0:.0f}s ({len(S.model.layers)} layers)")
    S.gen = Qwen36ForCausalLM([S.model], [S.args], S.mesh)
    pool = int(os.environ.get("DF_POOL", "65536"))
    # t3c DF_BLOCK: the KV block size (default 64; vLLM serves this model with 800-token blocks: the plain prefill at 1k
    # takes ~0.45 s there vs ~0.86 s here)
    S.block = int(os.environ.get("DF_BLOCK", "64"))
    S.nb = -(-pool // S.block)
    nkv = S.args.n_kv_heads // min(S.mesh.get_num_devices(), S.args.n_kv_heads)
    S.kv = S.gen.allocate_kv_cache((S.nb, nkv, S.block, S.args.head_dim), ttnn.bfloat16, len(S.model._attention_layer_indices))
    S.page_row = torch.arange(S.nb, dtype=torch.int32).reshape(1, S.nb)
    if NS > 1:  # ep2 ms1: stream s owns blocks [s * nbs, (s + 1) * nbs): its own page-table row of width nbs
        nbs = S.nb // NS
        S.page_rows = [torch.arange(s * nbs, (s + 1) * nbs, dtype=torch.int32).reshape(1, nbs) for s in range(NS)]
        S.page_row = S.page_rows[0]
        S.max_len = min(S.max_len, nbs * S.block)
    log(f"KV pool {pool} tokens ({S.nb} blocks)")

    import df_verify

    dcfg = json.load(open(os.path.join(DRAFT, "config.json")))
    S.tap_ids = dcfg["dflash_config"]["target_layer_ids"]
    S.ver = df_verify.Verifier(S.model, S.tap_ids, S.page_row, T=T, log=log) if not (one or NS > 1) else None
    install_prefill_taps()
    S.drafter = None
    if layers is None and os.environ.get("DF_NO_DRAFTER") != "1":
        S.status = "loading drafter"
        import df_drafter

        S.drafter = df_drafter.TtDrafter(S.mesh, DRAFT, S.model, log=log)
    S.status = "warmup"
    if NS > 1:
        import df_ms

        df_ms.bring_up_ms(sys.modules[__name__], NS)
        S.pcache = None
        if PCACHE:  # ep2 dfpc: index + the KV page-copy program (compiled here, before any trace)
            import df_pcache

            S.pcache = df_pcache.PrefixCache(sys.modules[__name__])
        if os.environ.get("DF_MS_WARMUP", "1") == "1":
            warmup()
    elif not one:
        warmup()
    if os.environ.get("DF_TIMERS", "0") == "1":
        _time_pf_methods()
    if os.environ.get("QWEN36_GC_FREEZE", "0") == "1":
        # ep2 t3c A33: the warm-up heap goes to the permanent generation (automatic GC passes skip it from now on)
        import gc

        gc.collect()
        gc.freeze()
    S.ready = True
    S.status = "ready"
    log("READY")


# ---------------------------------------------------------------------------- prefill with taps
def install_prefill_taps():
    """Hook the masked-bucket prefill so the tap layers' outputs for the last PF positions reach the host."""
    import ttnn

    model = S.model
    if getattr(S, "taps_installed", False):
        return
    S.taps_installed = True
    S.pf_cap = None  # dict while capturing: {"S": prompt len, "need_from": pos, "rows": {layer: [(pos0, tensor)]}}
    S.pf_cur = None
    orig_chunk = model._forward_prefill_chunk_masked_tp

    def chunk(token_buf, valid_len, chunk_start, *a, **k):
        S.pf_cur = (int(chunk_start), int(valid_len))
        try:
            return orig_chunk(token_buf, valid_len, chunk_start, *a, **k)
        finally:
            S.pf_cur = None

    model._forward_prefill_chunk_masked_tp = chunk
    if (not model.tp):
        orig_chunk1 = model._forward_prefill_chunk_masked

        def chunk1(token_buf, valid_len, chunk_start, *a, **k):
            S.pf_cur = (int(chunk_start), int(valid_len))
            try:
                return orig_chunk1(token_buf, valid_len, chunk_start, *a, **k)
            finally:
                S.pf_cur = None

        model._forward_prefill_chunk_masked = chunk1
    idx = list(model.layer_indices)
    for lid in S.tap_ids:
        if lid not in idx:
            continue
        layer = model.layers[idx.index(lid)]
        orig = layer.forward

        def fwd(*a, _orig=orig, _lid=lid, **k):
            out = _orig(*a, **k)
            mc = getattr(S, "mtr_cap", None)
            if mc is not None and k.get("mode") == "prefill":
                # ep2 ms2 DF_MTRACE: the tap copy of a traced masked bucket, inside its compile pass / capture (the
                # captured copy is the bucket's persistent tap buffer, rewritten by every replay)
                mc[_lid] = ttnn.clone(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                return out
            cap = S.pf_cap
            if cap is not None and k.get("mode") == "prefill" and S.pf_cur is not None:
                cs, vl = S.pf_cur
                lo = max(cs, cap["need_from"])
                hi = cs + vl
                if hi > lo and (os.environ.get("DF_TAP_DEFER", "0") == "1" or os.environ.get("DF_TAPS_DEV", "0") == "1"):
                    # ep2 t3c DF_TAP_DEFER=1: a device copy of the chunk output (fixed shape per bucket, compiled by the
                    # DF_WARM_LENS prompts) instead of a blocking read here: the host keeps enqueueing the rest of the
                    # prefill; the copies are read after the prefill (_tap_reads)
                    ct = getattr(S, "mtr_ctap", None)
                    if ct and not S.mtr_off and vl == ct[0][_lid].shape[-2]:
                        # ms2 DF_MTRACE: a full chunk's tap copy outlives the traced tail's replay, so it goes to a
                        # buffer allocated before every trace (an eager copy could sit in a trace's scratch); two sets
                        # alternate by chunk (the window spans at most two full chunks)
                        dst = ct[(cs // vl) % 2][_lid]
                        ttnn.copy(out, dst)
                        cap.setdefault("defer", []).append((_lid, cs, lo, hi, dst))
                    else:
                        cap.setdefault("defer", []).append((_lid, cs, lo, hi, ttnn.clone(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)))
                elif hi > lo:
                    # read the whole chunk output (a fixed shape per bucket) and cut on the host: a device slice with a
                    # prompt-dependent extent would compile a new program per prompt length, i.e. after the traces
                    # are parked (the hang rule)
                    _tr = time.perf_counter()
                    if S.mesh.get_num_devices() > 1 and out.shape[-1] < S.args.dim:  # fractured residual
                        h = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(S.mesh, dim=-1))
                    else:
                        h = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(S.mesh, dim=0))[:1] if S.mesh.get_num_devices() > 1 else ttnn.to_torch(out)
                    h = h.reshape(-1, h.shape[-1])[lo - cs: hi - cs]
                    cap["rows"].setdefault(_lid, []).append((lo, h.to(torch.bfloat16)))
                    S.tap_read_s = getattr(S, "tap_read_s", 0.0) + time.perf_counter() - _tr
            return out

        layer.forward = fwd


def _taps_dev(cap, n, need):
    """t3c DF_TAPS_DEV=1: the drafter's [1, 1, PF_ROWS, 5 * dim] context window built on the card from the deferred tap
    chunk copies: per chunk all-gather to full width (TP), row-major + one zero row appended, a row gather
    (ttnn.embedding, host-built uint32 indices: window row -> chunk row, else the zero row), chunks of one tap layer summed
    (every row comes from exactly one chunk, the others add zeros: exact), the tap layers concatenated by columns. Op shapes
    depend only on the prefill bucket of a chunk (compiled by the DF_WARM_LENS prompts) and the fixed window."""
    import df_drafter
    from models.tt_transformers.tt.ccl import tt_all_gather

    ttnn, mesh = S.ttnn, S.mesh
    nd = mesh.get_num_devices()
    PF, H = df_drafter.PF_ROWS, S.args.dim
    tap0 = n - need
    p0 = tap0 - tap0 % 8  # window row w = position p0 + w (the drafter aligns the first row to lane 0)
    rep_m = ttnn.ReplicateTensorToMesh(mesh) if nd > 1 else None
    if getattr(S, "taps_zrow", None) is None:
        S.taps_zrow = ttnn.from_torch(torch.zeros(1, H, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                      layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep_m)
    acc = {}
    w = torch.arange(PF) + p0
    keep = getattr(S, "mtr_keep", ())  # ms2 DF_MTRACE: the traced buckets' tap buffers are never freed
    for lid, cs, lo, hi, t in cap.pop("defer"):
        if nd > 1 and t.shape[-1] < H:  # fractured residual -> full width
            # (tt_all_gather frees its input: a kept buffer goes in as a copy)
            src = ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG) if id(t) in keep else t
            g = tt_all_gather(src, mesh, S.model.tt_ccl, cluster_axis=None, dim=3, topology=ttnn.Topology.Linear)
            if src is not t or id(t) not in keep:
                ttnn.deallocate(src)
        else:
            g = t
        C = g.shape[-2]
        rm = ttnn.reshape(ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT), (C, H))
        if id(g) not in keep:
            ttnn.deallocate(g)
        table = ttnn.concat([rm, S.taps_zrow], dim=0)  # [C + 1, H], row C = zeros
        ttnn.deallocate(rm)
        idx = torch.where((w >= lo) & (w < hi), w - cs, torch.full_like(w, C)).to(torch.int32).reshape(1, PF)
        idx_d = ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep_m)
        e = ttnn.embedding(idx_d, table, layout=ttnn.TILE_LAYOUT)  # [1, PF, H]
        for x in (idx_d, table):
            ttnn.deallocate(x)
        if lid in acc:
            s2 = ttnn.add(acc[lid], e)
            ttnn.deallocate(acc[lid])
            ttnn.deallocate(e)
            acc[lid] = s2
        else:
            acc[lid] = e
    zero = None
    parts = []
    for lid in S.tap_ids:
        if lid not in acc:  # (a tap layer with no rows: cannot happen with need > 0; keep the shape)
            if zero is None:
                zero = ttnn.from_torch(torch.zeros(1, PF, H, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep_m)
            parts.append(zero)
        else:
            parts.append(acc[lid])
    full = ttnn.concat(parts, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for p_ in parts:
        if p_ is not zero:
            ttnn.deallocate(p_)
    if zero is not None:
        ttnn.deallocate(zero)
    return ttnn.reshape(full, (1, 1, PF, len(S.tap_ids) * H))


def _tap_reads(cap):
    """t3c DF_TAP_DEFER: the deferred tap chunk copies -> host rows (the same cut as the inline read), copies freed."""
    ttnn = S.ttnn
    _tr = time.perf_counter()
    for lid, cs, lo, hi, t in cap.pop("defer"):
        if S.mesh.get_num_devices() > 1 and t.shape[-1] < S.args.dim:  # fractured residual
            h = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(S.mesh, dim=-1))
        else:
            h = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(S.mesh, dim=0))[:1] if S.mesh.get_num_devices() > 1 else ttnn.to_torch(t)
        if id(t) not in getattr(S, "mtr_keep", ()):
            ttnn.deallocate(t)
        h = h.reshape(-1, h.shape[-1])[lo - cs: hi - cs]
        cap["rows"].setdefault(lid, []).append((lo, h.to(torch.bfloat16)))
    S.tap_read_s = getattr(S, "tap_read_s", 0.0) + time.perf_counter() - _tr


# ---------------------------------------------------------------------------- ms2 DF_MTRACE: traced masked buckets
# The plain preset's QWEN36_PF_MTRACE (pf4 A5, tt/mtrace.py: each masked bucket's forward captured once, replayed after
# copying the request's tokens / rope / page table / masks into persistent buffers) under df_server, with the tap
# layers' outputs copied inside the trace into per-bucket buffers. The eager masked forward costs ~0.36 s of host
# dispatch at any bucket up to 1024 (the card waits); the replay is the same programs. Differences to model._mtrace_run:
# the bucket's K/V rows past the segment go to the request's own next blocks (eager: its last block; the model code
# sends them to block 0, which is stream 0's first block here), and a segment whose bucket would run past the
# request's page-table row takes the eager path. Buffers and programs before the first trace (mtrace_prepare, in the
# warm-up after the bucket compile), the captures after every other trace (mtrace_capture, end of the warm-up).
def mtrace_prepare():
    import ttnn
    from models.demos.blackhole.qwen36.tt import mtrace

    model = S.model
    want = [b for b in mtrace.buckets() if b in model._PREFILL_MASK_BUCKETS]
    assert want and os.environ.get("QWEN36_FIR_K") == "1", "DF_MTRACE needs QWEN36_PF_MTRACE buckets and QWEN36_FIR_K=1"
    S.mtr, S.mtr_bufs, S.mtr_keep, S.mtr_off = {}, {}, set(), False
    S.mtr_eager = model.prefill_masked_bucket
    rep = ttnn.ReplicateTensorToMesh(S.mesh)
    bs, km1 = S.block, 3
    pt = S.page_row

    def up(t, dt, layout):
        return ttnn.from_torch(t, dtype=dt, layout=layout, device=S.mesh, mesh_mapper=rep)

    for b in sorted(want):
        cos_t, sin_t = model._rope_tp_cos_sin_torch(0, b)
        sel = torch.zeros(1, km1, km1 + b, dtype=torch.float32)
        for j in range(km1):
            sel[:, j, b - 1 + j] = 1.0
        m = torch.ones(1, b, 1, dtype=torch.float32)
        S.mtr_bufs[b] = {
            "tok": up(torch.zeros(1, b, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "cos": up(cos_t, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "sin": up(sin_t, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "pt": up(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            "cpt": up(pt[:, : -(-b // bs)].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            "csi": up(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            "fir_sel": up(sel, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "fir_sel_p": up(sel[:, :, :km1].contiguous(), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "fir_sel_x": up(sel[:, :, km1:].contiguous(), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "m": up(m, ttnn.float32, ttnn.TILE_LAYOUT),
            "mq": up(m, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        }
    prev = model._bind_gdn_prefill_scratch()
    try:
        for b, bufs in S.mtr_bufs.items():  # compile pass (eager), as model._mtrace_capture_all does before capturing
            mtrace.CTX, S.mtr_cap = bufs, {}
            model._reset_gdn_state_for_new_sequence()
            h = model._mtrace_body(bufs, b)
            ttnn.deallocate(h)
            for t in S.mtr_cap.values():  # the copy _taps_dev makes of a kept tap buffer, compiled per bucket
                ttnn.deallocate(ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            if b == 2048:  # a full chunk's tap copies (set 0; set 1 cloned below): see the tap hook
                S.mtr_ctap = [dict(S.mtr_cap)]
            else:
                for t in S.mtr_cap.values():
                    ttnn.deallocate(t)
            ttnn.synchronize_device(S.mesh)
    finally:
        mtrace.CTX, S.mtr_cap = None, None
        model._reset_gdn_state_for_new_sequence()
        model._unbind_gdn_prefill_scratch(prev)
    assert getattr(S, "mtr_ctap", None), "DF_MTRACE needs the 2048 bucket in QWEN36_PF_MTRACE"
    S.mtr_ctap.append({k: ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG) for k, t in S.mtr_ctap[0].items()})
    S.mtr_keep.update(id(t) for c in S.mtr_ctap for t in c.values())
    # one long prompt (a full chunk + a tail, eager: no trace yet) compiles the tap copy into those buffers
    _, taps, _ = prefill([1000 + (i % 5000) for i in range(2048 + 100)], slot=0, page_row=S.page_row)
    if isinstance(taps, tuple) and taps[0] == "dev":
        ttnn.deallocate(taps[1])
    ttnn.synchronize_device(S.mesh)
    log(f"[mtrace] buckets {sorted(S.mtr_bufs)} compiled")


def mtrace_capture():
    import ttnn
    from models.demos.blackhole.qwen36.tt import mtrace
    from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

    model = S.model
    _wp.release_all()  # as every prefill: the prefill programs run with the prefetcher's global CB freed
    prev = model._bind_gdn_prefill_scratch()
    try:
        for b, bufs in S.mtr_bufs.items():
            mtrace.CTX, S.mtr_cap = bufs, {}
            model._reset_gdn_state_for_new_sequence()
            tid = ttnn.begin_trace_capture(S.mesh, cq_id=0)
            out = model._mtrace_body(bufs, b)
            ttnn.end_trace_capture(S.mesh, tid, cq_id=0)
            ttnn.synchronize_device(S.mesh)
            S.mtr[b] = (tid, bufs, out, dict(S.mtr_cap))
            S.mtr_keep.update(id(t) for t in S.mtr_cap.values())
    finally:
        mtrace.CTX, S.mtr_cap = None, None
        model._reset_gdn_state_for_new_sequence()
        model._unbind_gdn_prefill_scratch(prev)
    model.prefill_masked_bucket = _mtr_masked_bucket
    log(f"[mtrace] {len(S.mtr)} masked-bucket prefill traces captured (taps {sorted(next(iter(S.mtr.values()))[3])})")


def _mtr_masked_bucket(token_ids, page_table, actual_len, chunk_start=0, bucket=None, flex_sdpa=True, vision_tokens=None,
                       vis_row_offset=0):
    """model.prefill_masked_bucket with the traced buckets (S.mtr); anything else takes the eager path."""
    import ttnn

    model = S.model
    if bucket is None:
        bucket = model._mask_bucket_for(actual_len)
    ent = S.mtr.get(bucket)
    bs, km1 = S.block, 3
    nb = -(-bucket // bs)
    blk0 = chunk_start // bs
    W = int(page_table.shape[1])
    if ent is None or vision_tokens is not None or blk0 + nb > W or S.mtr_off or not flex_sdpa:
        return S.mtr_eager(token_ids, page_table, actual_len, chunk_start=chunk_start, bucket=bucket, flex_sdpa=flex_sdpa,
                           vision_tokens=vision_tokens, vis_row_offset=vis_row_offset)
    tid, bufs, out, taps = ent
    rep = ttnn.ReplicateTensorToMesh(S.mesh)

    def put(t, dt, layout, buf):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dt, layout=layout, mesh_mapper=rep), buf)

    if chunk_start == 0:
        model._reset_gdn_state_for_new_sequence()
        model._build_request_rope(token_ids[:, :actual_len], None)
    tok = torch.zeros(1, bucket, dtype=torch.int32)
    tok[:, :actual_len] = token_ids[:, :actual_len].to(torch.int32)
    put(tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, bufs["tok"])
    cos_t, sin_t = model._rope_tp_cos_sin_torch(chunk_start, bucket)
    put(cos_t, ttnn.bfloat16, ttnn.TILE_LAYOUT, bufs["cos"])
    put(sin_t, ttnn.bfloat16, ttnn.TILE_LAYOUT, bufs["sin"])
    Wb = int(bufs["pt"].shape[-1])
    pt = page_table[:, :Wb]
    if pt.shape[1] < Wb:
        pt = torch.cat([pt, torch.zeros(pt.shape[0], Wb - pt.shape[1], dtype=pt.dtype)], dim=1)
    put(pt.to(torch.int32).contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, bufs["pt"])
    put(page_table[:, blk0:blk0 + nb].to(torch.int32).contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, bufs["cpt"])
    put(torch.tensor([chunk_start], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, bufs["csi"])
    sel = torch.zeros(1, km1, km1 + bucket, dtype=torch.float32)
    for j in range(km1):
        sel[:, j, actual_len + j] = 1.0
    put(sel, ttnn.bfloat16, ttnn.TILE_LAYOUT, bufs["fir_sel"])
    put(sel[:, :, :km1].contiguous(), ttnn.bfloat16, ttnn.TILE_LAYOUT, bufs["fir_sel_p"])
    put(sel[:, :, km1:].contiguous(), ttnn.bfloat16, ttnn.TILE_LAYOUT, bufs["fir_sel_x"])
    m = torch.zeros(1, bucket, 1, dtype=torch.float32)
    m[:, :actual_len, :] = 1.0
    put(m, ttnn.float32, ttnn.TILE_LAYOUT, bufs["m"])
    put(m, ttnn.bfloat16, ttnn.TILE_LAYOUT, bufs["mq"])
    ttnn.execute_trace(S.mesh, tid, cq_id=0, blocking=False)
    S.mtr_n = getattr(S, "mtr_n", 0) + 1
    cap = S.pf_cap
    if cap is not None:  # the eager tap hook's entries, from the bucket's tap buffers
        lo, hi = max(chunk_start, cap["need_from"]), chunk_start + actual_len
        if hi > lo:
            for lid in S.tap_ids:
                if lid in taps:
                    cap.setdefault("defer", []).append((lid, chunk_start, lo, hi, taps[lid]))
    return model._masked_bucket_logits_tp(out, actual_len, bucket)


def prefill(ids, slot=0, page_row=None, cached_len=None):
    """Prompt -> slot 0. Returns (first-token logits [vocab] float, taps host [n, 25600], first tap position).
    cached_len (ep2 dfpc DF_PREFIX_CACHE): KV pages [0, cached_len) of page_row hold this prefix; the model resumes
    from its GDN snapshot store (None: no cache, today's path)."""
    from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

    _tq0 = time.perf_counter()
    _wp.release_all()  # ep2 QWEN36_PF_GCB_MODE=decode: the prefill programs need the global CB L1
    S.pf_t = {"release": round(time.perf_counter() - _tq0, 3)}
    for _pf in _wp._by_mesh.values():  # t3c: the gc.collect share of the release (ms)
        S.pf_t.update(getattr(_pf, "release_t", None) or {})
    import df_drafter

    n = len(ids)
    need = min(n, df_drafter.PF_ROWS - 7) if S.drafter is not None else 0
    S.pf_cap = {"S": n, "need_from": n - need, "rows": {}} if need else None
    toks = torch.tensor(ids, dtype=torch.int32).reshape(1, -1)
    S.tap_read_s = 0.0
    try:
        if (not S.model.tp):  # 1-card drafter tests: one masked bucket (prompt <= 2048)
            assert n <= 2048, "1-card test mode: prompts up to 2048 tokens"
            lg = S.ttnn.to_torch(S.model.prefill_masked_bucket(toks, S.page_row, actual_len=n))
        else:
            lg = S.model.prefill_paged_slots([toks], S.page_row if page_row is None else page_row, [slot], valid_lens=[n],
                                             **({"cached_lens": [int(cached_len)]} if cached_len is not None else {}))[0]
            if getattr(S, "pcache", None) is not None and not S.pcache.compiled:
                S.pcache.compile()  # ep2 dfpc: the first (warm-up) prefill, before any trace
    finally:
        cap, S.pf_cap = S.pf_cap, None
    return _prefill_post(lg, cap, n, need, _tq0)


def prefill_gen(ids, slot, page_row):
    """ms2 DF_MS_ILV: prefill() as a generator that yields after each full chunk of a long prompt (> 2048 tokens, TP), so
    the multi-stream worker runs a step of the live streams in between; returns prefill()'s result. The same segments,
    programs and order as prefill() -> prefill_paged_slots -> prefill_traced_chunked -> _prefill_chunked_eager_tp (no chunk
    trace in this server) with QWEN36_GDN_SLOT_DEV. At a yield the GDN prefill scratch is unbound (the decode bindings
    back), the eager GDN masks are cleared (rebuilt by the next segment) and the step may recreate the prefetcher's global
    CB (released again here). What carries over: the scratch state, the request's own KV blocks and, with DF_MTRACE, the
    full-chunk tap copies (buffers allocated before every trace). Shorter prompts: prefill() unchanged, no yield."""
    import ttnn
    from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp
    from models.demos.blackhole.qwen36.tt.gdn import fused_chunk as _fc

    model = S.model
    chunk = model._chunked_chunk_size or 2048
    n = len(ids)
    # DF_MS_ILV_MIN: prompts up to this many tokens prefill whole (default 2048: every prompt with a split point)
    lim = max(chunk, int(os.environ.get("DF_MS_ILV_MIN", "2048")))
    if ((not model.tp) or n <= lim or model._chunked_trace_id is not None or not getattr(S, "mtr_ctap", None)
            or os.environ.get("QWEN36_GDN_SLOT_DEV") != "1" or PCACHE):  # ep2 dfpc: no resume in the ILV loop
        return prefill(ids, slot=slot, page_row=page_row)
    import df_drafter

    _tq0 = time.perf_counter()
    _wp.release_all()
    S.pf_t = {"release": round(time.perf_counter() - _tq0, 3)}
    need = min(n, df_drafter.PF_ROWS - 7) if S.drafter is not None else 0
    S.pf_cap = {"S": n, "need_from": n - need, "rows": {}} if need else None
    toks = torch.tensor(ids, dtype=torch.int32).reshape(1, -1)
    S.tap_read_s = 0.0
    num_full, tail = n // chunk, n % chunk
    prev = model._bind_gdn_prefill_scratch()
    lg = None
    try:
        model._build_request_rope(toks[:, :n], None)
        model._reset_gdn_state_for_new_sequence()
        last = None
        for c in range(num_full):
            cs = c * chunk
            if last is not None:
                ttnn.deallocate(last)
            model._set_vision_merge(toks[:, cs: cs + chunk], None, model._vis_row_offset_for(toks, cs))
            last = model._forward_prefill_chunk_masked_tp(toks[:, cs: cs + chunk], chunk, cs, page_row, chunk, flex_sdpa=True)
            ttnn.synchronize_device(model.device)
            if c == num_full - 1 and tail == 0:
                break
            ttnn.deallocate(last)
            last = None
            cap_y, S.pf_cap = S.pf_cap, None  # (no prefill hook runs in a step; kept off for safety)
            model._unbind_gdn_prefill_scratch(prev)
            _fc.masks_clear()
            prev = None
            S.ilv_y = getattr(S, "ilv_y", 0) + 1
            yield cs + chunk
            _wp.release_all()
            prev = model._bind_gdn_prefill_scratch()
            S.pf_cap = cap_y
        if tail > 0:
            cs = num_full * chunk
            dev_lg = model.prefill_masked_bucket(toks[:, cs:n], page_row, actual_len=tail, chunk_start=cs, flex_sdpa=True,
                                                 vision_tokens=None, vis_row_offset=model._vis_row_offset_for(toks, cs))
        else:
            dev_lg = model._masked_bucket_logits_tp(last, chunk, chunk)
            ttnn.deallocate(last)
        comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
        lg = ttnn.to_torch(dev_lg, mesh_composer=comp).reshape(-1, model.args.vocab_size)[:1].float().view(1, 1, -1)
        ttnn.deallocate(dev_lg)
        model._write_gdn_slot_from_scratch(int(slot), prev)
    finally:
        if prev is not None:
            model._unbind_gdn_prefill_scratch(prev)
        _fc.masks_clear()
        cap, S.pf_cap = S.pf_cap, None
    return _prefill_post(lg, cap, n, need, _tq0)


def _prefill_post(lg, cap, n, need, _tq0):
    """prefill()'s second half: the drafter's tap window from the captured tap rows; returns prefill()'s result."""
    import df_drafter

    taps = None
    if cap and cap.get("defer") and os.environ.get("DF_TAPS_DEV", "0") == "1":
        _tr = time.perf_counter()
        taps = ("dev", _taps_dev(cap, n, need), need)  # t3c: the window stays on the card
        S.tap_read_s = getattr(S, "tap_read_s", 0.0) + time.perf_counter() - _tr
        cap = None
    elif cap and cap.get("defer"):
        _tap_reads(cap)
    if cap and os.environ.get("DF_TAPS_FULL", "0") == "1":
        # ep2 t3b DF_TAPS_FULL=1: the drafter's [PF_ROWS, 25600] context buffer built here in one pass (each tap piece
        # copied once to its final rows / columns; only the pad rows zeroed): prefill_context takes it as it is
        tap0 = n - need
        lead = tap0 - (tap0 - tap0 % 8)
        H5 = S.args.dim
        full = getattr(S, "taps_full", None)  # t3c: one persistent host buffer (no 106 MB first-touch per request)
        if full is None or full.shape != (df_drafter.PF_ROWS, len(S.tap_ids) * H5):
            full = S.taps_full = torch.empty(df_drafter.PF_ROWS, len(S.tap_ids) * H5, dtype=torch.bfloat16)
        full[:lead].zero_()
        full[lead + need:].zero_()
        for ti, lid in enumerate(S.tap_ids):
            for lo, h in cap["rows"].get(lid, []):
                r0 = lead + lo - tap0
                full[r0: r0 + h.shape[0], ti * H5:(ti + 1) * H5] = h
        taps = ("full", full, need)
    elif cap:
        parts = []
        for lid in S.tap_ids:
            rows = sorted(cap["rows"].get(lid, []), key=lambda r: r[0])
            parts.append(torch.cat([r[1] for r in rows], dim=0))
        taps = torch.cat(parts, dim=1)
        assert taps.shape[0] == need, (taps.shape, need)
    S.pf_split = round(S.tap_read_s, 3)
    S.pf_t["total"] = round(time.perf_counter() - _tq0, 3)
    return lg.reshape(-1)[: S.model.vocab_size].float(), taps, n - need


# ---------------------------------------------------------------------------- generation
class Req:
    def __init__(self, ids, max_new, min_new, ignore_eos, stream_q=None, penalties=None):
        self.ids = ids
        self.pen = None
        if penalties and (penalties.get("repetition_penalty", 1.0) != 1.0 or penalties.get("presence_penalty", 0.0) != 0.0
                          or penalties.get("frequency_penalty", 0.0) != 0.0):
            self.pen = Penalties(ids, float(penalties.get("repetition_penalty", 1.0)),
                                 float(penalties.get("presence_penalty", 0.0)),
                                 float(penalties.get("frequency_penalty", 0.0)), S.model.vocab_size)
        self.max_new = max_new
        self.min_new = min_new
        self.ignore_eos = ignore_eos
        self.q = stream_q  # gets lists of new token ids, then None
        self.out = []
        self.stats = {}
        self.done = threading.Event()
        self.error = None


class Penalties:
    """vLLM v1 sampler penalties, same float32 ops in the same order (vllm/_custom_ops.apply_repetition_penalties_torch,
    model_executor/layers/utils.apply_penalties): repetition (prompt or output tokens), then frequency, then presence.
    Applied after the min_tokens EOS mask, before the greedy argmax."""

    def __init__(self, prompt_ids, rp, pp, fp, vocab):
        self.rp = torch.tensor([rp], dtype=torch.float32)
        self.pp = torch.tensor([pp], dtype=torch.float32)
        self.fp = torch.tensor([fp], dtype=torch.float32)
        self.prompt_mask = torch.zeros(vocab, dtype=torch.bool)
        self.prompt_mask[torch.tensor(prompt_ids, dtype=torch.long)] = True
        self.counts = torch.zeros(vocab, dtype=torch.int64)

    def apply(self, lg):
        lg = lg.clone()
        out_mask = self.counts > 0
        pen = torch.where(self.prompt_mask | out_mask, self.rp.expand_as(lg), torch.ones_like(lg))
        lg *= torch.where(lg > 0, 1.0 / pen, pen)
        lg -= self.fp * self.counts
        lg -= self.pp * out_mask
        return lg

    def add(self, t):
        self.counts[t] += 1


def host_pick(lg, n_out, req):
    """Greedy pick from host logits with the request's min_tokens EOS mask and penalties (vLLM order)."""
    lg = lg.clone()
    if not req.ignore_eos and n_out < req.min_new:
        for e in S.eos:
            lg[e] = -float("inf")
    if req.pen is not None:
        lg = req.pen.apply(lg)
    t = int(torch.argmax(lg))
    if req.pen is not None:
        req.pen.add(t)
    return t


def pick(row_ids, i, n_out, req, host_rows=None):
    """Greedy id of verify row i (device argmax), or from host logits when the request has penalties."""
    if host_rows is not None:
        return host_pick(host_rows[i], n_out, req)
    t = row_ids[i]
    if not req.ignore_eos and n_out < req.min_new and t in S.eos:
        lg = S.ver.host_logits()[i]  # a whole-tensor read: no new device program while traces are parked
        for e in S.eos:
            lg[e] = -float("inf")
        t = int(torch.argmax(lg))
    return t


def generate_fake(req):
    out = []
    while len(out) < req.max_new:
        new = [S.fake[(len(out) + k) % len(S.fake)] for k in range(3)]
        if not req.ignore_eos and len(out) >= max(req.min_new, 40):
            new[-1] = sorted(S.eos)[0]
        new = new[: req.max_new - len(out)]
        out.extend(new)
        if req.q:
            req.q.put(new)
        time.sleep(0.01)
        if new and new[-1] in S.eos:
            break
    req.out, req.stats = out, {"fake": True, "completion_tokens": len(out)}
    return out


def generate(req):
    if os.environ.get("DF_FAKE") == "1":
        return generate_fake(req)
    ids = req.ids
    t0 = time.perf_counter()
    logits, taps, tap0 = prefill(ids)
    first = host_pick(logits, 0, req)
    t_pf = time.perf_counter() - t0
    out = [first]
    if req.q:
        req.q.put([first])
    start = len(ids)  # position of `first` (not yet through the target)
    steps, acc, t_draft, t_ver, t_com = 0, [], 0.0, 0.0, 0.0
    stop = (first in S.eos and not req.ignore_eos)
    t_bs = t_pc = 0.0
    if not stop:
        tq = time.perf_counter()
        S.ver.begin_sequence()
        t_bs = time.perf_counter() - tq
        if S.drafter is not None and S.mode == "spec":
            S.drafter.prefill_context(taps, list(range(tap0, start)))
        t_pc = time.perf_counter() - tq - t_bs
        if os.environ.get("DF_EARLY_GCB", "0") == "1" and os.environ.get("DF_NO_RECAP", "0") == "1":
            # ep2 t3b DF_EARLY_GCB=1: recreate the prefetcher's global CB now (host work) while the card still runs the
            # drafter context ops queued above, instead of at the first verify (CQ order keeps it after them)
            from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

            tq = time.perf_counter()
            _wp.decode_all()
            if _wp.recreate_pending():
                _wp.recreated()
            S.ver.timing["early_gcb_ms"] = round(1000 * (time.perf_counter() - tq), 1)
    first_step = None
    prestage = os.environ.get("DF_PRESTAGE", "0") == "1"
    prev_a = None
    if S.drafter is not None:
        S.drafter.pre = None
    while not stop and len(out) < req.max_new and start + T < S.max_len:
        anchor = out[-1]
        ta = time.perf_counter()
        chained = (S.drafter is not None and S.mode == "spec" and getattr(S.drafter, "chain_dst", None) is not None
                   and S.drafter.trace_id is not None and S.ver.trace_id is not None)
        if chained:
            # ep2 t3b DF_CHAIN: drafter trace -> verify trace back to back; the drafts come back with the verify
            if prestage and getattr(S.drafter, "pre", None) and prev_a is not None:
                S.drafter.draft_async_pre(prev_a, anchor)  # DF_PRESTAGE: staged during the previous verify
            else:
                S.drafter.draft_async(anchor, start)
            tb = time.perf_counter()
            block, post = S.ver.run_chained(
                start, pre=(lambda st=start: S.drafter.prepare_all(st, S.ver.out[2])) if prestage else None)
            assert block[0] == anchor, (block[0], anchor)
        else:
            if S.drafter is not None and S.mode == "spec":
                draft = S.drafter.draft(anchor, start)
            else:
                draft = [anchor] * (T - 1)  # plain mode: the verify pass with junk drafts (accepts ~0)
            tb = time.perf_counter()
            block = [anchor] + list(draft)
            post = S.ver.run(block, start)
        rows = S.ver.host_logits() if req.pen is not None else None
        tc = time.perf_counter()
        a = 0
        new = []
        while True:
            t = pick(post, a, len(out) + len(new), req, rows)
            new.append(t)
            if a < T - 1 and t == block[a + 1]:
                a += 1
                continue
            break
        # new = [posterior_0 .. posterior_a] = accepted drafts d1..da + bonus; slots 0..a went through the target
        room = req.max_new - len(out)
        if not req.ignore_eos:
            for k, t in enumerate(new):
                if t in S.eos and len(out) + k + 1 >= req.min_new:
                    new = new[: k + 1]
                    stop = True
                    break
        if len(new) > room:
            new = new[:room]
        if S.drafter is not None and S.mode == "spec":
            # ep2 t3: the taps are read before the commit trace runs (with DF_X_PERSIST the commit traces are not
            # re-captured after a verify re-capture, so their temporaries may sit where the new verify's outputs are)
            S.drafter.set_step_context(S.ver.out[2], a + 1, start)
        S.ver.commit(a)
        prev_a = a if chained else None
        td = time.perf_counter()
        out.extend(new)
        if req.q:
            req.q.put(new)
        steps += 1
        acc.append(a + 1)
        t_draft += tb - ta
        t_ver += tc - tb
        t_com += td - tc
        if first_step is None:
            first_step = (round(1000 * (tb - ta), 1), round(1000 * (tc - tb), 1), round(1000 * (td - tc), 1))
        start += a + 1
    S.mesh_sync()
    wall = time.perf_counter() - t0
    probe = _probe(start, out[-1]) if os.environ.get("DF_PROBE", "0") == "1" and steps else None
    req.out = out
    req.stats = dict(
        t=time.strftime("%Y-%m-%dT%H:%M:%S"), mode=S.mode, prompt_tokens=len(ids), completion_tokens=len(out),
        steps=steps, accept_mean=round(sum(acc) / len(acc), 3) if acc else None,
        tokens_per_step=round((len(out) - 1) / steps, 3) if steps else None,
        prefill_s=round(t_pf, 3), decode_s=round(wall - t_pf, 3),
        decode_tps=round((len(out) - 1) / (wall - t_pf), 2) if wall > t_pf and len(out) > 1 else None,
        draft_ms=round(1000 * t_draft / max(1, steps), 2), verify_ms=round(1000 * t_ver / max(1, steps), 2),
        commit_ms=round(1000 * t_com / max(1, steps), 2), accept_hist=[acc.count(i) for i in range(1, T + 1)],
        begin_ms=round(1000 * t_bs, 1), ctx_ms=round(1000 * t_pc, 1), first_step_ms=first_step,
        ctx_split=getattr(S.drafter, "ctx_timing", None), pf_s=getattr(S, "pf_split", None),
        recreate_ms=S.ver.timing.pop("recreate_ms", None), early_gcb_ms=S.ver.timing.pop("early_gcb_ms", None),
        pf_t=getattr(S, "pf_t", None), gcb_t=getattr(S, "gcb_t", None),
        step_t=_step_t(), probe=probe,
        tag=getattr(req, "tag", ""), trace=S.ver.trace_id is not None, sdpa_rows=S.ver.sdpa_rows,
        penalties=None if req.pen is None else [float(req.pen.rp), float(req.pen.pp), float(req.pen.fp)],
    )
    try:
        with open(STATS, "a") as f:
            f.write(json.dumps(req.stats) + "\n")
    except Exception:
        pass
    S.last_stats = (S.last_stats + [req.stats])[-50:]
    log("req", json.dumps(req.stats))
    return out



def _probe(start, anchor):
    """t3c DF_PROBE=1 (measurement only, after the request's last step, state no longer used): per-step split of the
    served DFlash2 step. Each entry = (wall ms per step incl. the final sync, host ms per step to enqueue), n back to
    back: the drafter trace alone, the verify trace alone, both (the device span of a step without host gaps), the full
    host step as served (draft_async + run_chained incl. the 3 read backs), and the read backs alone on an idle card."""
    ttnn, m, d, v = S.ttnn, S.mesh, S.drafter, S.ver
    if d is None or d.trace_id is None or v.trace_id is None or getattr(d, "chain_dst", None) is None:
        return None
    from models.demos.blackhole.qwen36.ep2_dflash.df_verify import dev_rows
    n = int(os.environ.get("DF_PROBE_N", "16"))

    def rep(fn):
        S.mesh_sync()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        t1 = time.perf_counter()
        S.mesh_sync()
        t2 = time.perf_counter()
        return [round(1000 * (t2 - t) / n, 3), round(1000 * (t1 - t) / n, 3)]

    def ex(tid):
        ttnn.execute_trace(m, tid, cq_id=0, blocking=False)

    def reads():
        idx_t, mx_t = v.out[0] if v.shard_argmax else (v.out[0], None)
        dev_rows(idx_t, m)
        if mx_t is not None:
            dev_rows(mx_t, m)
        dev_rows(v.inputs[0], m)

    def serial():  # the served step with a sync after each replay (no overlap of host work and device)
        d.draft_async(anchor, start)
        S.mesh_sync()
        v.run_chained(start)

    r = dict(n=n)
    r["drafter"] = rep(lambda: ex(d.trace_id))
    r["verify"] = rep(lambda: ex(v.trace_id))
    r["both"] = rep(lambda: (ex(d.trace_id), ex(v.trace_id)))
    r["both_staged"] = rep(lambda: (d.draft_async(anchor, start), v._stage(None, start), ex(v.trace_id)))
    r["step"] = rep(lambda: (d.draft_async(anchor, start), v.run_chained(start)))
    r["serial"] = rep(serial)
    r["reads"] = rep(reads)
    r["sync"] = rep(lambda: None)
    return r


def _step_t():
    """t3b: per-step host phases since the last call (ms): drafter stage, drafter enqueue, verify stage, verify enqueue+PF."""
    r = []
    for o in (S.drafter, S.ver):
        st = getattr(o, "step_t", None) if o is not None else None
        if st and st[0]:
            r += [round(1000 * st[1] / st[0], 2), round(1000 * st[2] / st[0], 2)]
            o.step_t = [0, 0.0, 0.0, 0.0]
    return r


def _time_pf_methods():
    """t3b: timers around the prefetcher's GCB recreate pieces (S.gcb_t, ms of the last recreate)."""
    from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

    for pf in _wp._by_mesh.values():
        if getattr(pf, "_t3b_timed", False):
            continue
        pf._t3b_timed = True
        for name in ("_make_gcb", "_launch"):
            orig = getattr(pf, name)

            def wrapped(*a, _o=orig, _n=name, **k):
                t0 = time.perf_counter()
                try:
                    return _o(*a, **k)
                finally:
                    S.__dict__.setdefault("gcb_t", {})[_n] = round(1000 * (time.perf_counter() - t0), 1)

            setattr(pf, name, wrapped)


def mesh_sync():
    S.ttnn.synchronize_device(S.mesh)


S.mesh_sync = mesh_sync


def warmup():
    """Compile every program shape (prefill buckets, verify, commit for each a, drafter) before any trace."""
    import ttnn

    t0 = time.time()
    # every masked-bucket prefill program (masked + full per bucket, and the KV fill widths), on the B=1 GDN scratch
    # exactly like the vLLM plugin's warmup_model_prefill does
    prev = S.model._bind_gdn_prefill_scratch()
    try:
        S.model.warmup_prefill_masked_buckets(S.page_row)
    finally:
        S.model._unbind_gdn_prefill_scratch(prev)
    log(f"warmup: prefill buckets compiled in {time.time() - t0:.0f}s")
    mtr = os.environ.get("DF_MTRACE", "0") == "1"  # ms2: traced masked-bucket prefill (default off)
    if mtr:
        mtrace_prepare()
    lens = [int(x) for x in os.environ.get("DF_WARM_LENS", "100,1024,3000,4096").split(",")]
    if NS > 1:
        import df_ms

        df_ms.warmup_ms(sys.modules[__name__], lens)
        if mtr:
            mtrace_capture()
        log(f"warmup done in {time.time() - t0:.0f}s")
        return
    for L in lens:
        if L + 2 * T + 64 >= S.max_len:
            continue
        ids = [1000 + (i % 5000) for i in range(L)]
        r = Req(ids, max_new=3 * T, min_new=0, ignore_eos=True)
        generate(r)
        log(f"warmup prompt {L}: {len(r.out)} tokens, {r.stats.get('verify_ms')} ms/verify")
        if os.environ.get("DF_CHAIN", "0") == "1" and S.drafter is not None and getattr(S.drafter, "chain_dst", None) is None:
            S.drafter.link_verify(S.ver.inputs[0])  # the next warm-up prompts compile the combine ops eagerly
    for a in range(T):  # every commit slice program
        S.ver.commit(a)
    ttnn.synchronize_device(S.mesh)
    if os.environ.get("DF_TRACE", "0") == "1":
        ids = [1000 + i for i in range(64)]
        _, taps, tap0 = prefill(ids)
        S.ver.begin_sequence()
        S.ver.capture_trace([ids[-1]] * T, len(ids))
        if os.environ.get("DF_TRACE_COMMIT", "1") == "1":
            S.ver.capture_commit_traces()
        if os.environ.get("DF_BEGIN_TRACE", "0") == "1":
            S.ver.capture_begin_trace()
        if S.drafter is not None and os.environ.get("DF_TRACE_DRAFT", "1") == "1":
            S.drafter.prefill_context(taps, list(range(tap0, len(ids))))
            S.drafter.capture_trace(ids[-1], len(ids))
        r = Req(ids, max_new=4 * T, min_new=0, ignore_eos=True)
        generate(r)
        log(f"warmup traced: {r.stats.get('verify_ms')} ms/verify")
    if mtr:
        mtrace_capture()
    log(f"warmup done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------- worker
def _dispatch(job):
    kind = job[0]
    if kind == "exec_fn":
        job[1]()
    elif kind == "gen":
        req = job[1]
        try:
            generate(req)
        except Exception as e:
            req.error = repr(e)
            log("generate failed:", traceback.format_exc())
        finally:
            if req.q:
                req.q.put(None)
            req.done.set()
    elif kind == "exec":
        path, box, ev = job[1], job[2], job[3]
        buf = io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                g = {"S": S, "torch": torch, "log": log, "__name__": "__dflash_exec__"}
                exec(compile(open(path).read(), path, "exec"), g)
            box["ok"] = True
        except Exception:
            box["ok"] = False
            buf.write(traceback.format_exc())
        box["out"] = buf.getvalue()[-20000:]
        ev.set()


def worker():
    """The only thread that touches the device: bring-up, requests and admin exec run here in order."""
    while True:
        job = S.jobs.get()
        try:
            _dispatch(job)
        except Exception:
            log("worker error", traceback.format_exc())


# ---------------------------------------------------------------------------- HTTP
def render(messages, kw):
    return S.tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, chat_template=S.chat_template, **(kw or {})
    )


def encode(text):
    return S.tok.encode(text, add_special_tokens=False)


class Detok:
    def __init__(self):
        self.ids, self.sent = [], ""

    def push(self, new):
        self.ids.extend(new)
        text = S.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�"):
            return ""
        d = text[len(self.sent):]
        self.sent = text
        return d


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            if not S.ready:
                return self._json(503, {"status": S.status})
            return self._json(200, {"object": "list", "data": [{"id": "qwen38", "object": "model"}]})
        if self.path.startswith("/dflash/stats"):
            pc = getattr(S, "pcache", None)
            return self._json(200, {"status": S.status, "stats": S.last_stats,
                                    **({"pcache": pc.info()} if pc is not None else {})})
        if self.path.startswith("/health"):
            return self._json(200 if S.ready else 503, {"status": S.status})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = self._body()
        except Exception as e:
            return self._json(400, {"error": repr(e)})
        if self.path == "/dflash/exec":
            box, ev = {}, threading.Event()
            S.jobs.put(("exec", body["file"], box, ev))
            ev.wait()
            return self._json(200, box)
        if not S.ready:
            return self._json(503, {"error": "not ready", "status": S.status})
        if self.path == "/tokenize":
            if "messages" in body:
                ids = encode(render(body["messages"], body.get("chat_template_kwargs")))
            else:
                ids = S.tok.encode(body.get("prompt", ""), add_special_tokens=body.get("add_special_tokens", True))
            return self._json(200, {"tokens": ids, "count": len(ids), "max_model_len": S.max_len})
        if self.path == "/detokenize":
            return self._json(200, {"prompt": S.tok.decode(body.get("tokens", []))})
        if self.path == "/v1/chat/completions":
            return self.chat(body)
        return self._json(404, {"error": "not found"})

    def chat(self, body):
        if float(body.get("temperature", 1.0)) != 0.0:
            return self._json(400, {"error": "this server is greedy only: set temperature 0"})
        ids = encode(render(body["messages"], body.get("chat_template_kwargs")))
        max_new = body.get("max_tokens") or body.get("max_completion_tokens") or (S.max_len - len(ids) - 2 * T)
        max_new = max(1, min(int(max_new), S.max_len - len(ids) - 2 * T))
        stream = bool(body.get("stream"))
        pen = {k: body[k] for k in ("repetition_penalty", "presence_penalty", "frequency_penalty") if body.get(k) is not None}
        req = Req(ids, max_new, int(body.get("min_tokens") or 0), bool(body.get("ignore_eos")),
                  queue.Queue() if stream else None, penalties=pen)
        req.tag = body["messages"][-1].get("content", "")[:60] if body.get("messages") else ""
        rid = "chatcmpl-" + uuid.uuid4().hex[:16]
        created = int(time.time())
        S.jobs.put(("gen", req))
        if not stream:
            req.done.wait()
            if req.error:
                return self._json(500, {"error": req.error})
            out = req.out
            fin = "stop" if (out and out[-1] in S.eos and not req.ignore_eos) else "length"
            text = S.tok.decode(out, skip_special_tokens=True)
            return self._json(200, {
                "id": rid, "object": "chat.completion", "created": created, "model": "qwen38",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text, "reasoning_content": None},
                             "finish_reason": fin}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)},
                "dflash": req.stats,
                **({"output_ids": list(out)} if body.get("return_ids") else {}),  # ep2 dfpc: exactness checks
            })
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        dt = Detok()
        base = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": "qwen38"}
        send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        n = 0
        while True:
            new = req.q.get()
            if new is None:
                break
            n += len(new)
            d = dt.push(new)
            if d:
                send({**base, "choices": [{"index": 0, "delta": {"content": d}, "finish_reason": None}]})
        req.done.wait()
        out = req.out
        fin = "stop" if (out and out[-1] in S.eos and not req.ignore_eos) else "length"
        send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": fin}]})
        if (body.get("stream_options") or {}).get("include_usage"):
            send({**base, "choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                                                    "total_tokens": len(ids) + len(out)},
                  "dflash": req.stats, **({"output_ids": list(out)} if body.get("return_ids") else {})})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    port = int(os.environ.get("DF_PORT", "8080"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"http on :{port} (admin + API; API answers 503 until READY)")
    if NS > 1:
        import df_ms

        threading.Thread(target=df_ms.ms_worker, args=(sys.modules[__name__],), daemon=True).start()
    else:
        threading.Thread(target=worker, daemon=True).start()

    def up():
        try:
            bring_up()
        except Exception:
            S.status = "bring-up FAILED: " + traceback.format_exc()[-3000:]
            log("BRINGUP_FAILED", traceback.format_exc())

    # bring-up runs in the worker thread too, so every device call stays on one thread
    S.jobs.put(("exec_fn", up))

    # ep2 prefetch: SIGTERM -> clean close on the worker thread (traces released, prefetcher stopped, mesh closed, fabric
    # off): a 2-card process that never closes the mesh leaves the ethernet cores stuck (teardown finding)
    import signal

    def _close():
        import ttnn

        try:
            for o, m in [(getattr(S, "ver", None), ("release_trace", "release_traces"))] + [(d, ("release_trace",)) for d in (getattr(S, "drafters", None) or [getattr(S, "drafter", None)])]:
                for name in m if o is not None else ():
                    try:
                        getattr(o, name)()
                    except Exception as e:  # noqa: BLE001
                        log(f"close: {name}: {e}")
            try:
                from models.demos.blackhole.qwen36.tt import weight_prefetch as _wp

                _wp.stop_all()
            except Exception as e:  # noqa: BLE001
                log(f"close: prefetcher stop: {e}")
            if S.mesh.get_num_devices() == 1 and os.environ.get("DF_CLOSE1", "close") == "close":
                # df1c: one card = close the mesh WITHOUT a fabric call (the plugin reset_fabric skips 1 device).
                # Measured on card 0: exit without close, and close + set_fabric_config(DISABLED), both left
                # ethernet core 27-25 inactive (the next open timed out; tt-smi -r needed)
                ttnn.close_mesh_device(S.mesh)
                log("ep2-prefetch: clean close done (1 card: mesh closed, no fabric call)")
                return
            ttnn.close_mesh_device(S.mesh)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            log("ep2-prefetch: clean close done (mesh closed, fabric disabled)")
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, lambda *_: S.jobs.put(("exec_fn", _close)))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()

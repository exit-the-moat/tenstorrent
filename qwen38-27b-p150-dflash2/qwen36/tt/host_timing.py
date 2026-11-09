# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Modified for Exit the Moat Episode 2 (2026-11-02): new file (agent "host"), diagnostics only.
"""QWEN36_HOST_TIMING=1|2 (default off): host-side timing of each decode step (Qwen36ForCausalLM.decode_forward).

Phases (ms per step, wall clock on the host):
  total      Qwen36ForCausalLM.decode_forward, start to return (what the dev loop / the runner waits for)
  wrap       the qwen36 wrapper before Generator.decode_forward (PF bookkeeping, bucketing, trace-store switch)
  trace_text Generator._decode_forward_trace_text (input refresh + execute_trace)
    prep     model.prepare_decode_inputs_host (host tensors: tokens, pos, rope cos/sin, page table)
    h2d      generator.copy_host_to_device (the four input writes)
    pfq      the prefetcher request queued before the replay (QWEN36_PF)
  dev_wait   =2 only: ttnn.synchronize_device right after the replay is enqueued (device run + launch latency)
  read       Generator.read_decode_output (blocking .cpu(): with =1 this includes waiting for the device)
  post       process_decode_output_host (to_torch + fp32 copy)
  gap        from the end of one decode_forward to the start of the next (< 1 s only): the caller's own time per step
             (served: vLLM scheduler, input batch, host sampling, output processing)
QWEN36_HOST_TIMING_EVERY=N (default 0 = never): log the medians every N steps and start over (for served runs).
=2 adds the synchronize (a diagnostic split; not for counted runs). report() returns medians over the recorded steps.
"""
import os
import sys
import time
from collections import defaultdict

_ACC = defaultdict(list)
_LAST = [None, 0]  # end of the previous decode_forward, steps since the last log


def mode():
    return os.environ.get("QWEN36_HOST_TIMING", "0")


def add(label, dt):
    _ACC[label].append(dt * 1000.0)


def reset():
    _ACC.clear()
    _LAST[0] = None


def enter(t):
    if _LAST[0] is not None and t - _LAST[0] < 1.0:
        add("gap", t - _LAST[0])


def leave(t0, t):
    add("total", t - t0)
    dump = os.environ.get("QWEN36_HOST_TIMING_DUMP")
    if dump:  # t1: one line per decode step: wall time, total, previous post (output wait + readback), previous gap
        p, g = _ACC.get("post"), _ACC.get("gap")
        with open(dump, "a") as f:
            f.write(f"{t:.5f} {(t - t0) * 1000:.3f} {p[-1] if p else -1:.3f} {g[-1] if g else -1:.3f}\n")
    _LAST[0] = t
    _LAST[1] += 1
    every = int(os.environ.get("QWEN36_HOST_TIMING_EVERY", "0") or 0)
    if every and _LAST[1] >= every:
        from loguru import logger

        logger.info(f"QWEN36_HOST_TIMING ms/step over {_LAST[1]} steps: {report()}")
        _ACC.clear()
        _LAST[1] = 0


def report(skip=2):
    out = {}
    for k, v in _ACC.items():
        v = v[skip:] if len(v) > skip + 2 else v
        s = sorted(v)
        out[k] = dict(n=len(v), med=round(s[len(s) // 2], 4), mn=round(s[0], 4), mean=round(sum(v) / len(v), 4))
        if len(s) >= 20:  # t1: the tail (p90, the 3 largest, count over 2x median)
            out[k].update(p90=round(s[int(len(s) * 0.9)], 3), top=[round(x, 2) for x in s[-3:]],
                          n2x=sum(1 for x in s if x > 2 * s[len(s) // 2]))
    return out


def _timed(fn, label):
    def w(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            add(label, time.perf_counter() - t)

    w._ht_orig = fn
    return w


def install(gen):
    """Wrap the generator's per-step pieces (instance attributes; the image's generator module gets one guarded
    wrapper around copy_host_to_device that looks the accumulator up at call time)."""
    if getattr(gen, "_ht_installed", False):
        return
    gen._ht_installed = True
    import ttnn

    gm = sys.modules["models.tt_transformers.tt.generator"]
    if not hasattr(gm.copy_host_to_device, "_ht_orig"):
        orig = gm.copy_host_to_device

        def c2d(*a, **k):
            t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                h = sys.modules.get("models.demos.blackhole.qwen36.tt.host_timing")
                if h is not None and h.mode() != "0":
                    h.add("h2d", time.perf_counter() - t)

        c2d._ht_orig = orig
        gm.copy_host_to_device = c2d

    m = gen.model[0]
    m.prepare_decode_inputs_host = _timed(m.prepare_decode_inputs_host, "prep")
    inner = gen._decode_forward_trace_text
    sync = mode() == "2"

    def trace_text(*a, **k):
        t = time.perf_counter()
        out = inner(*a, **k)
        t1 = time.perf_counter()
        add("trace_text", t1 - t)
        if sync:
            ttnn.synchronize_device(m.mesh_device)
            add("dev_wait", time.perf_counter() - t1)
        return out

    gen._decode_forward_trace_text = trace_text
    gen.read_decode_output = _timed(gen.read_decode_output, "read")
    gen.process_decode_output_host = _timed(gen.process_decode_output_host, "post")
    install_vllm()
    try:
        from models.demos.blackhole.qwen36.tt import weight_prefetch as wp

        q = wp.Prefetcher.queue
        if not hasattr(q, "_ht_orig"):
            wp.Prefetcher.queue = _timed(q, "pfq")
    except Exception:  # noqa: BLE001
        pass


def install_vllm():
    """Served runs: also time the vLLM side of each step (class-level wrappers, once per process; no-ops in the dev
    loop where these modules are not loaded): v_sched = scheduler.schedule, v_upd = update_from_output, v_build =
    runner.build_model_input, v_exec = runner.execute_model (includes decode_forward), v_sample = runner.sample_tokens
    (output wait + readback + host sampling + runner output), v_tok = runner._get_output_tokens (host sampling),
    v_apply = runner.apply_and_build_runner_output."""
    targets = [
        ("vllm_tt_plugin.model_runner", "TTModelRunner",
         [("build_model_input", "v_build"), ("execute_model", "v_exec"), ("sample_tokens", "v_sample"),
          ("_get_output_tokens", "v_tok"), ("apply_and_build_runner_output", "v_apply")]),
        ("vllm_tt_plugin.scheduler", "TTScheduler", [("schedule", "v_sched"), ("update_from_output", "v_upd")]),
    ]
    for mod, cls, meths in targets:
        m = sys.modules.get(mod)
        c = getattr(m, cls, None) if m is not None else None
        if c is None:
            continue
        for name, label in meths:
            f = getattr(c, name, None)
            if f is None or hasattr(f, "_ht_orig"):
                continue
            setattr(c, name, _timed(f, label))


def probe(gen, reps=5):
    """Dev-loop helper: time the pieces of the logits readback on the last decode output (no device program runs)."""
    import torch
    import ttnn

    outs = [v for v in gen.trace_output_decode[False]]
    tt = outs[0][0] if isinstance(outs[0], tuple) else outs[0]
    R = {"shape": list(tt.shape), "dtype": str(tt.dtype), "layout": str(tt.layout)}

    def tm(f):
        best = 1e9
        for _ in range(reps):
            t = time.perf_counter()
            r = f()
            best = min(best, time.perf_counter() - t)
        return r, round(best * 1000, 4)

    host, R["cpu_ms"] = tm(lambda: tt.cpu())
    dev = ttnn.get_device_tensors(host)
    R["shard_shape"] = list(dev[0].shape)
    sh, R["to_torch_one_ms"] = tm(lambda: ttnn.to_torch(dev[0]))
    R["torch_shape"] = list(sh.shape)
    R["torch_dtype"] = str(sh.dtype)
    flat = sh.reshape(-1, sh.shape[-1])
    buf = torch.zeros(8, flat.shape[-1] * len(dev), dtype=torch.float32)
    _, R["copy8_ms"] = tm(lambda: buf[: min(8, flat.shape[0]), : flat.shape[-1]].copy_(flat[: min(8, flat.shape[0])]))
    _, R["copy1_ms"] = tm(lambda: buf[:1, : flat.shape[-1]].copy_(flat[:1]))
    _, R["post_ms"] = tm(lambda: gen.process_decode_output_host([host]))
    return R


def probe2(gen, reps=7):
    """Like probe, but a fresh host copy for every rep (median), and each piece of the lean post path separately."""
    import torch
    import ttnn

    outs = [v for v in gen.trace_output_decode[False]]
    tt = outs[0][0] if isinstance(outs[0], tuple) else outs[0]
    m = gen.model[0]
    keys = ("cpu", "get_dev", "to_torch0", "to_torch1", "reshape_copy", "post_full")
    T = {k: [] for k in keys}
    for _ in range(reps):
        t = time.perf_counter()
        host = tt.cpu()
        t1 = time.perf_counter()
        dev = ttnn.get_device_tensors(host)
        t2 = time.perf_counter()
        a = ttnn.to_torch(dev[0])
        t3 = time.perf_counter()
        b = ttnn.to_torch(dev[1])
        t4 = time.perf_counter()
        buf = getattr(m, "_lean_logits_buf", None)
        if buf is not None:
            buf[:1, : a.shape[-1]].copy_(a.reshape(-1, a.shape[-1])[:1])
            buf[:1, a.shape[-1] :].copy_(b.reshape(-1, b.shape[-1])[:1])
        t5 = time.perf_counter()
        host2 = tt.cpu()
        t6 = time.perf_counter()
        gen.process_decode_output_host([host2])
        t7 = time.perf_counter()
        for k, v in zip(keys, (t1 - t, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t7 - t6)):
            T[k].append(v * 1000)
    return {k: round(sorted(v)[len(v) // 2], 4) for k, v in T.items()}


def probe3(gen, reps=9, B=1, pos=1000):
    """Dev-loop helper: the input-refresh pieces (host tensor builds and the four H2D writes into the live trace
    inputs, with the same values the next step would write anyway). Run only after a measured run."""
    import torch
    import ttnn

    m = gen.model[0]
    ins = gen.trace_inputs_decode[False][0]
    keys = ("torch_rope", "ft_tok", "ft_pos", "ft_rope", "ft_pt", "w_tok", "w_pos", "w_rope", "w_pt", "w_all4_nb",
            "sync_after_nb", "c2d_fn")
    T = {k: [] for k in keys}
    pt = gen.prev_page_table[0] if getattr(gen, "prev_page_table", None) else torch.zeros(B, 32, dtype=torch.int32)
    pt = pt[:B]
    tokens = torch.zeros(B, 1, dtype=torch.long)
    from models.tt_transformers.tt.common import copy_host_to_device

    for _ in range(reps):
        c = [time.perf_counter()]
        pos_vec = torch.full((B,), pos, dtype=torch.int32)
        rd = m.args.rope_head_dim
        inv_freq = 1.0 / (m.args.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
        freqs = torch.outer(pos_vec.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().reshape(1, B, 1, rd).to(torch.bfloat16)
        sin = emb.sin().reshape(1, B, 1, rd).to(torch.bfloat16)
        rp = torch.cat([cos, sin], dim=0)
        c.append(time.perf_counter())
        a = ttnn.from_torch(tokens.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        c.append(time.perf_counter())
        b = ttnn.from_torch(pos_vec, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        c.append(time.perf_counter())
        r = ttnn.from_torch(rp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        c.append(time.perf_counter())
        p = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        c.append(time.perf_counter())
        hs = (a, b, r, p)
        for h, d in zip(hs, ins):
            ttnn.copy_host_to_device_tensor(h, d)
            c.append(time.perf_counter())
        for h, d in zip(hs, ins):
            ttnn.copy_host_to_device_tensor(h, d, cq_id=0)
        c.append(time.perf_counter())
        ttnn.synchronize_device(m.mesh_device)
        c.append(time.perf_counter())
        copy_host_to_device(host_tensors=hs, device_tensors=ins)
        c.append(time.perf_counter())
        for k, i in zip(keys, range(len(c) - 1)):
            T[k].append((c[i + 1] - c[i]) * 1000)
    R = {k: round(sorted(v)[len(v) // 2], 4) for k, v in T.items()}
    R["shapes"] = [list(t.shape) for t in ins if t is not None]
    return R


def async_check(dev, n_prompt=1000, n_steps=33):
    """Dev-loop helper for QWEN36_ASYNC_HOST: run the same teacher-forced steps twice after one prefill each, once
    the sync way (decode_forward(read_from_device=True), as the dev loop does) and once the way the plugin's async
    path does (decode_forward(read_from_device=False) + read_decode_output(async_read=True) + event wait +
    process_decode_output_host), and compare the logits bit for bit."""
    import torch
    import ttnn

    gen = dev.gen
    out = {}
    for mode in ("sync", "async"):
        seqs = dev.model.args.max_batch_size
        prompt, cont = dev.tokens(n_prompt, n_steps)
        _, _, pt = dev.decode_inputs(seqs, 0, 0)
        if seqs > 1:
            gen.prefill_forward(torch.tensor([prompt]), pt[:1], dev.kv, torch.tensor([n_prompt]), enable_trace=True,
                                empty_slots=[0])
        else:
            gen.prefill_forward(torch.tensor([prompt]), pt, dev.kv, torch.tensor([n_prompt]), enable_trace=True)
        rows, times = [], []
        for i in range(n_steps - 1):
            tok, pos, _ = dev.decode_inputs(seqs, 0, 0)
            tok[0, 0] = cont[i]
            pos[0] = n_prompt + i
            t = time.perf_counter()
            if mode == "sync":
                o = gen.decode_forward(tok, pos, page_table=pt, kv_cache=dev.kv, enable_trace=True, read_from_device=True)
            else:
                d = gen.decode_forward(tok, pos, page_table=pt, kv_cache=dev.kv, enable_trace=True, read_from_device=False)
                host, evs = gen.read_decode_output(d, async_read=True)
                for e in evs:
                    ttnn.event_synchronize(e)
                o = gen.process_decode_output_host(host)
            lg = o[0] if isinstance(o, tuple) else o
            rows.append(lg.reshape(-1, lg.shape[-1])[0].float().clone())
            times.append((time.perf_counter() - t) * 1000)
        out[mode] = (torch.stack(rows), sorted(times)[len(times) // 2])
    a, b = out["sync"][0], out["async"][0]
    diff = (a != b).any(dim=-1).nonzero()
    return dict(bit_identical=bool(torch.equal(a, b)), first_diff=int(diff[0]) if len(diff) else None,
                max_abs=float((a - b).abs().max()), sync_ms=round(out["sync"][1], 3), async_ms=round(out["async"][1], 3))

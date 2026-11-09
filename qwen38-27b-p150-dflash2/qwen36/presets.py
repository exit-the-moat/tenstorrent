# SPDX-License-Identifier: Apache-2.0
"""ep2 presets: QWEN36_PRESET=best2card | best1card turns on exactly the gated best switch set (2026-11-03, Target 0).

Read once, when the qwen36 package is first imported (models/demos/blackhole/qwen36/__init__.py), before any module
reads its switches. Each switch is set with os.environ.setdefault, so an explicit env var always overrides the preset.
QWEN36_DFLASH2=1 adds the DFlash2 (df_server.py) switches on top: the set served losslessly on ep2/dfl-g7 (gate g7,
not g8: PF_LMHEAD and AR_RING were never run with the verify pass, so they default to off there).

  best2card  = gate g8 as served (host gate_g8.log, 04:36): 38.4 tok/s 1 user 1k, exact 5/5, NIAH 6/6, GSM8K 58/60.
               PF_MLP_WIDE stays 1: the =0 recommendation (fuse, -0.17 ms dev bench) is not re-gated.
               + QWEN36_PF_PREQ=1 (Target 1, served gate ep2-t1-gate: exact 5/5 vs g8, GSM8K 58/60 with all 60 texts
               identical to g8, 1 user 1k 39.4 / 39.6, 8 users 1k 142 / 144 total). Off under QWEN36_DFLASH2 (never
               run with the verify pass). The plugin-side switches EP2_GREEDY_FAST / EP2_SP_CACHE need the mounted
               model_runner.py and their own env vars: vllm_patch/README.md.
               + Target 2 (prefill, served gate ep2-t2-gate): QWEN36_GDN_SLOT_DEV=1, QWEN36_AGMM_GU=10,9,
               QWEN36_AGMM_QKV=10,9, QWEN36_PF_BUCKETS=128,...,1280,1536,1792,2048 (server warm-up 56 -> 145 s).
               Off under QWEN36_DFLASH2.
  best1card  = 1 card, BF4 + 4-bit KV (QWEN_SDPA_KV4) + both 1-card KV fixes (the KV cast is in the code,
               QWEN36_KV_SUBBLOCK defaults to on) + the 1-card fused kernels (devloop gate1c_kv4.sh): 14.48 tok/s,
               exact 5/5 across starts, NIAH 6/6, GSM8K 59/60.

python3 presets.py <dir> writes <name>.env (plain KEY=VALUE files, the same sets) for launchers that don't import
the package.
"""
import os

_GDN_PREFILL = {"QWEN36_GDN_CONV1D": "0", "QWEN36_GDN_PREFILL_DRAM": "1", "QWEN36_GDN_UNFUSED_RS": "1"}

PRESETS = {
    "best2card": {
        **_GDN_PREFILL,
        "QWEN36_NAIVE_BF4": "1", "QWEN_SDPA_BF8": "1", "QWEN36_COMPACT_LOGITS": "1", "QWEN36_LEAN_HOST": "1",
        "QWEN36_GDN_FUSED_DECODE": "1", "QWEN36_ATTN_FUSED": "1",
        "QWEN36_PF": "1", "QWEN36_PF_RPB": "5", "QWEN36_PF_GCB_KB": "400", "QWEN36_PF_GCB_MODE": "decode",
        "QWEN36_LEAN_TUPLE": "1", "QWEN36_PF_LMHEAD": "1", "QWEN36_FAST_INPUTS": "2", "QWEN36_GDN_FAST": "6",
        "QWEN36_ATTN_SPLIT": "6", "QWEN36_CCL_AR": "1", "QWEN36_CCL_AR_LINKS": "4", "QWEN36_CCL_AR_NOC1": "1",
        "QWEN36_PF_MLP_WIDE": "1", "QWEN36_PF_QKV_WIDE": "1", "QWEN36_AR_RING": "1",
        "QWEN36_PF_PREQ": "1",  # t1 (served gate ep2-t1-gate, 2026-11-03 08:4x)
        # t2 (prefill, served gate ep2-t2-gate, 2026-11-03 10:1x): GDN prefill state to the decode slot on the card,
        # wider prefill AGMM grids (10x9 instead of 8x9), more masked buckets (1k prompts no longer pay for 2048)
        "QWEN36_GDN_SLOT_DEV": "1", "QWEN36_AGMM_GU": "10,9", "QWEN36_AGMM_QKV": "10,9",
        "QWEN36_PF_BUCKETS": "128,256,512,1024,1280,1536,1792,2048,2176,2304,2560",
        # t4 A1 (gated by t3b step 0, served ep2-t3b-s0dam: exact 5/5 vs g8, GSM8K 58/60): greedy ids on the card. Needs
        # the runner patch
        "QWEN36_DEV_ARGMAX": "1",
        # t3c (ep2/t3c only, served ep2-t3c-p27: exact 5/5, TTFT 1k -5 %): no gc.collect in the PF release before a prefill
        "QWEN36_PF_NO_GC": "1",
        # t3c A29 (served ep2-t3c-a29p8 / a29p32 / a29d: exact 5/5, slot + batch invariance 5/5 at 8 / 16 / 32 users):
        # the GDN prefill scratch -> decode slot as one program (kernels/slotcopy_reader.cpp): -70 ms per prefill
        "QWEN36_GDN_SLOT_K": "1",
        # t3c A30 (served ep2-t3c-a30: exact 5/5, slot + batch invariance 5/5): keep the decode traces across the GCB
        # recreate after a prefill (no per-request re-capture): 1 user 1k 40.0 -> 41.1 tok/s, 8 users 243 -> 258 total
        "QWEN36_KEEP_DECODE_TRACES": "1",
        # t3c A31 (served ep2-t3c-a31: exact 5/5, slot + binv 5/5): host trims around the prefill (cached get_arch_name,
        # PF zero frame with device zeros, GDN valid_len masks once per prefill): TTFT 1k 0.58 -> 0.50 s
        "QWEN36_HOST_TRIM": "1",
        # t3c A33 (served ep2-t3c-a33p / a33d: exact 5/5, numbers unchanged): gc.collect + gc.freeze once after the warm-up,
        # so the automatic GC (no explicit collect per prefill since QWEN36_PF_NO_GC) skips the long-lived heap
        "QWEN36_GC_FREEZE": "1",
        # pf4 A2 (day 4, served ep2-pf4-firk3: exact 5/5 vs g8; dev test bit-identical out + carry state): the GDN prefill
        # FIR conv as one program (kernels/firk_*.cpp: shifted tilizes + the stock mul/addcmul/silu math per tile),
        # 8k TTFT 2.781 -> 2.362 s (2,976 -> 3,510 tok/s). QWEN36_FIR_RM (A1, row-major data movement only) is the fallback
        "QWEN36_FIR_K": "1",
        # pf4 A3 (served ep2-pf4-rs4: exact 5/5 vs g8; dev bench bit-identical): the prefill reduce-scatters (GDN out fp32,
        # MLP down, attention out) on 4 ethernet links instead of 2: 8k TTFT 2.362 -> 2.279 s (3,639 tok/s)
        "QWEN36_PF_RS4": "1",
        # pf4 A5 (served ep2-pf4-mt1 / mt2: exact 5/5 vs g8): every masked prefill bucket (short prompts, long-prompt
        # tails) as a trace captured at warm-up (tt/mtrace.py), valid_len tensors as persistent buffers: 8k TTFT
        # 2.279 -> 2.132 s, 26-token prompt prefill 350 -> 166 ms. Needs QWEN36_FIR_K=1. Off under QWEN36_DFLASH2
        "QWEN36_PF_MTRACE": "128,256,512,1024,1280,1536,1792,2048,2176,2304,2560",
        # pf4 A6 (served ep2-pf4-f2c: exact 5/5 vs g8; dev A/B logits + GDN states bit-identical at 100/2048/4096): the FIR
        # writes q / k / v as 3 tensors and the output gate is one program (kernels/gatek_*.cpp: concat_heads index map,
        # z read in place from qkvzab, the stock silu (bf16 math) and fp32 SFPU mul): 8k TTFT 2.131 -> 2.068 s
        "QWEN36_GDN_FUSE2": "1",
        # pf4 A9 (served ep2-pf4-tab: exact 5/5 vs g8; 4-layer sweeps bit-identical logits): prefill AGMM M/N block sizes
        # by row count (the M split runs over the 10 grid columns: M_block 8 left 9 of 10 idle at 128 rows): 8k TTFT
        # 2.068 -> 2.007 s, 1k 0.388 -> 0.372 s, 26-token prompt prefill 166 -> 136 ms
        "QWEN36_AGMM_GU_TAB": "256:1,32/2560:4,32", "QWEN36_AGMM_QKV_TAB": "4096:4,16",
        # pf4 A10 (served ep2-pf4-wrap: exact 5/5 vs g8; sweeps bit-identical): 1D in0-mcast prefill matmuls at <= 128
        # rows (every core reads its own weight columns): MLP down 200 -> 137 us per layer in the 128 bucket
        "QWEN36_PF_MM1D": "128",
        # pf4 A12 (served ep2-pf4-zf3: exact 5/5 vs g8, slot 5/5, binv 8 5/5 x2, decode unchanged): the PF_PREQ drain
        # before every prefill (the request queued ahead after the last decode step) as a trace replay of the zero frame
        # (persistent zero inputs in DRAM) instead of ~320 eager matmuls: drain 44 -> 20 ms, 1k TTFT 0.370 -> 0.347 s
        "QWEN36_PF_DRAIN_TRACE": "1",
        # pf6 A1b (served ep2-pf6-p6h8 / p6h8b: exact 5/5 vs g8, long prompts 3/3 vs c880178, decode unchanged): only the
        # first 8 of 327 weights of the next decode request are queued ahead (the tail goes with the next head right
        # after the replay is enqueued), so the drain before a prefill is a head-only replay: 19.8 -> 0.5-1.1 ms,
        # 1k TTFT 0.314 -> 0.290-0.297 s
        "QWEN36_PF_PREQ_HEAD": "8",
        # pf4 A14 (served ep2-pf4-firt: exact 5/5 vs g8, long prompts 2.1k/5k/8.2k token-identical to c880178; unit test
        # bit-identical): the full-chunk FIR reads the qkv tiles straight from qkvzab (kernels/firt_*.cpp: untilize in
        # the kernel, shifted blocks assembled by the reader): no qkv slice / untilize op, 8k TTFT 1.980 -> 1.936 s
        "QWEN36_FIR_T": "1",
        # pf4 A16 (served ep2-pf4-mm16b: exact 5/5 vs g8, long prompts 3/3 vs c880178): GDN out-proj / attn wo and MLP
        # down on an 11x10 grid (subblock 1x3; K blocking unchanged: an in0_block_w change broke the long prompts), AGMM
        # QKV subblock 1x2: 8k TTFT 1.943 -> 1.920 s
        "QWEN36_PF_MMCFG": "3072,5120,256:11,10,1,3/8704,5120,256:11,10,1,3", "QWEN36_AGMM_QKV_SUB": "1,2",
        # pf5 A1 (served ep2-pf5-a1: exact 5/5 vs g8, long prompts 3/3 vs c880178; 4-layer logits identical): the GDN
        # out-proj partial is already fp32 (fp32 gated input), so its typecast to fp32 was a 203 us copy per GDN layer:
        # skipped. Attention prefill: nlp_create_qkv_heads on the fused [q|k|v] block (no q / kv slices). 8k TTFT
        # 1.924 -> 1.885 s, 1k 0.341 -> 0.330 s
        "QWEN36_PF_TC_SKIP": "1", "QWEN36_PF_DM5": "1",
        # pf5 A2 + A3 (served ep2-pf5-a23: exact 5/5 vs g8, long prompts 3/3 vs c880178; unit tests bit-identical): the
        # traced full chunk skips the decode conv window copies (copied once from conv_carry after the last full chunk
        # when no tail follows), masked buckets run the FIR_T kernel too (new state by the same one-hot select, split
        # into carry and qkvzab parts): 1k TTFT 0.330 -> 0.320 s, 8k 1.885 -> 1.864 s
        "QWEN36_PF_WIN_DEFER": "1", "QWEN36_FIR_TM": "1",
        # pf5 B4 + B5 + C1 (served ep2-pf5-b45: exact 5/5 vs g8, long prompts 3/3 vs c880178; 4-layer logits identical):
        # prefill reduce-scatter outputs in L1, masked buckets without the q/k/v mask multiplies (beta/g masks stay),
        # the FIR one-hot select parts as their own MTRACE buffers: 1k TTFT 0.320 -> 0.312 s, 8k 1.864 -> 1.849 s
        "QWEN36_PF_RS_L1": "1", "QWEN36_GDN_NOQKVMASK": "1", "QWEN36_FIR_TM_SPLIT": "1",
        # pf5 D1 (served ep2-pf5-mg2: exact 5/5 vs g8, long prompts 3/3 vs c880178, the 2,111- and 8,203-token prompts
        # take the merged path): a tail of <= 128 tokens runs with the last full chunk as one traced 2176 bucket instead
        # of a 2048 chunk + a 128 bucket: 8k TTFT 1.849 -> 1.751 s, 32k 8.075 -> 7.960 s 
        # (2176 added to QWEN36_PF_BUCKETS and QWEN36_PF_MTRACE above)
        # pf5 D2 (served ep2-pf5-tm5: exact 5/5, long prompts 3/3; ep2-pf5-lx2m: 2,411 / 4,312 / 8,492-token prompts
        # token-identical to the unmerged path lx2b): tails up to 512 merged (buckets 2304, 2560 added above, GU block
        # table extended to 2560 rows = output partition only): 2,412-token TTFT 0.628 -> 0.551 s, 8,492 1.918 -> 1.859 s;
        # warm-up ~141 -> ~182 s
        "QWEN36_PF_MERGE_TAIL": "512",
    },
    "best1card": {
        **_GDN_PREFILL,
        "QWEN36_NAIVE_BF4": "1", "QWEN36_COMPACT_LOGITS": "1", "QWEN36_LEAN_HOST": "1", "QWEN36_SD_MM_CFG": "1",
        "QWEN36_GDN_FUSED_DECODE": "1", "QWEN36_ADDNORM_FUSED": "1", "QWEN36_ATTN_FUSED": "1", "QWEN_SDPA_KV4": "1",
    },
}

# DFlash2 (df_server.py, 2 cards): the switches of the lossless served run (devloop dfl2/g7.env, 04:36)
DFLASH2 = {
    # pf4: the traced masked buckets are not run under df_server (its prefill taps the layer outputs)
    "QWEN36_PF_MTRACE": "",
    "QWEN36_PF_KEEP_RANK": "1", "QWEN36_PF_LMHEAD": "0", "QWEN36_AR_RING": "0", "QWEN36_PF_PREQ": "0",
    # t2 prefill switches under df_server too (t3c A22, served ep2-t3c-a22: exact 5/5 vs g8 + dfl-g7; TTFT 1k 1.45 -> 1.27 s,
    # wl code_implement short wall 3.79 -> 2.97 s) with every masked bucket and its tap reads warmed before the traces
    "QWEN36_GDN_SLOT_DEV": "1", "QWEN36_AGMM_GU": "10,9", "QWEN36_AGMM_QKV": "10,9",
    "QWEN36_PF_BUCKETS": "128,256,512,1024,1280,1536,1792,2048",
    "DF_WARM_LENS": "100,200,400,1000,1200,1500,1700,2000,3000,4096",
    # t3c A23 (served ep2-t3c-a23: exact 5/5 vs g8 + dfl-g7): no gc.collect in the PF release before each prefill (the
    # global CB is freed by refcount; release() still checks the range is free): release 0.34 -> 0.003 s of every TTFT
    "QWEN36_PF_NO_GC": "1",
    # t3c A26 (served ep2-t3c-a26: exact 5/5 vs g8 + dfl-g7, tokens/step identical): the drafter context window built on
    # the card from device copies of the tap chunks (no tap reads to the host, no 106 MB upload): TTFT 1k 0.94 -> 0.53 s,
    # 4k 3.01 -> 1.29 s, ctx 26 -> 8 ms (DF_TAP_DEFER / DF_TAPS_FULL are then unused)
    "DF_TAPS_DEV": "1",
    "QWEN36_DEV_ARGMAX": "0",  # vLLM runner path only
    "DF_TRACE": "1", "DF_SDPA_ROWS": "0", "DF_MAX_GATHER": "1", "DF_PF_RECOMMIT": "1",
    "DF_POOL": "65536", "DF_LEN": "65536", "DF_DRAFT_MLP": "bfp4",
    # t3b A1 (served ep2-t3b-a1norecap: exact 5/5 vs g8 + dfl-g7, GSM8K 59/60 with all 60 texts = t3-final): the verify
    # trace captured before a prefill is replayed after the GCB recreate (no per-request re-capture)
    "DF_NO_RECAP": "1",
    # t3b A3 (served ep2-t3b-a3ctx: exact 5/5, acceptance identical): drafter context ring built on the card
    "DF_CTX_DEV": "1",
    # t3b A4 (served ep2-t3b-a4begin: exact 5/5, acceptance identical): begin_sequence as one trace (27 -> 0.1 ms)
    "DF_BEGIN_TRACE": "1",
    # t3b A5 (served ep2-t3b-a5gcb: exact 5/5, acceptance identical): GCB recreated after the drafter context is queued
    "DF_EARLY_GCB": "1",
    # t3b A6 (served ep2-t3b-a6chain: exact 5/5, acceptance identical): drafter -> verify replays back to back, the
    # drafts combined on the card into the verify token input (df_chain.py), one read back per step
    "DF_CHAIN": "1",
    # t3b A7 (served ep2-t3b-a7d1d: exact 5/5; drafter numerics change, acceptance within +-1.5 %, code prompts equal):
    # explicit 1D mcast decode program configs on 64 cores for the drafter matmuls (replay 9.35 -> ~7.4 ms)
    "DF_DRAFT_1D": "64",
    # t3b A9 (served ep2-t3b-a9xwin: exact 5/5, acceptance identical): the GDN verify window built by one data-movement
    # program (kernels/dfwin_reader.cpp) instead of slice + 3 untilize + concat + tilize + copy (step 43.6 -> 40.0 ms)
    "DF_XWIN": "1",
    # t3b A11 (served ep2-t3b-a11gall: exact 5/5, acceptance identical): the gates of all 8 chained GDN slots in one
    # compute pass, the writer expands each slot's row into the uniform decay / beta tiles (per slot 21.9 -> 16.8 us)
    "DF_GATES_ALL": "1",
    # t3b A13 (served ep2-t3b-a13pre: exact 5/5, acceptance identical): the next draft's host inputs prepared for every
    # accepted count while the verify runs (DF_PRESTAGE) + no commit program (DF_NOCOMMIT, A10): draft stage 1.1 -> 0.5 ms
    "DF_PRESTAGE": "1", "DF_NOCOMMIT": "1",
    # t3b A14 (served ep2-t3b-a14kvf: exact 5/5): K and V of a verify row in one fused paged update (V resharded one
    # core row down: the fused op needs disjoint cores): 16 -> 9 programs per attention layer
    "DF_KVF_SKIP": "1",
    # t3b A15 (served ep2-t3b-a15pro: exact 5/5): the state-independent GDN work of all 8 slots (conv, q / k l2 norms) in
    # a prologue with rows = slots, each slot gets its rows from the reader (per slot 16.8 -> ~10 us compute)
    "DF_GDN_PRO": "1",
    # t3b A18 (served ep2-t3b-a18rms: exact 5/5): the GDN output rms norm / z gate of all 8 slots once at the end of the
    # chain (o rows exchanged per slot as 64-byte rows into o_all), not per slot
    "DF_RMSALL": "1",
    # t3c A21 (served ep2-t3c-a21: exact 5/5 vs g8 + dfl-g7): the drafter context buffer [2080, 25600] built once in the
    # prefill from the tap pieces (persistent host buffer), not cat + zeros + copy per request: ctx 58 -> 26 ms
    "DF_TAPS_FULL": "1",
    # t3 (Target 3): ccl's drafter switches (ep2/ccl-draft 42ef2f4, in ep2_dflash/df_drafter.py; served ep2-t3-a5:
    # acceptance unchanged, draft 21.2 -> 13.5 ms). The t3 verify switches (DF_GDN_XROW / XCONV / CHAIN1, DF_KV_SKIP,
    # DF_PCACHE, DF_X_PERSIST / X_DIRECT, DF_COMMIT_K, DF_AR) default to on in ep2_dflash/df_verify.py itself.
    "DF_RING_POS": "1", "DF_SHIFT_MM": "1", "DF_CONV_BIAS": "1", "DF_RMS_SH": "1", "DF_DRAFT_TP": "1",
    "DF_STAGE_PACK": "1", "DF_DRAFT_MAX_GATHER": "1",
}


def switches(name, dflash2=False):
    if name not in PRESETS:
        raise ValueError(f"QWEN36_PRESET={name!r}: unknown (have {sorted(PRESETS)})")
    s = dict(PRESETS[name])
    if dflash2:
        s.update(DFLASH2)
    return s


def apply():
    name = os.environ.get("QWEN36_PRESET", "")
    if not name or os.environ.get("_QWEN36_PRESET_APPLIED") == name:
        return
    df = os.environ.get("QWEN36_DFLASH2") == "1"
    s = switches(name, df)
    over = {k: os.environ[k] for k in s if k in os.environ and os.environ[k] != s[k]}
    for k, v in s.items():
        os.environ.setdefault(k, v)
    os.environ["_QWEN36_PRESET_APPLIED"] = name
    tag = " +DFLASH2" if df else ""
    print(f"QWEN36_PRESET={name}{tag}: {len(s)} switches, overridden by explicit env: {over or 'none'}", flush=True)


if __name__ == "__main__":
    import sys

    for n in PRESETS:
        for df in (False, True):
            if df and n != "best2card":
                continue
            fn = os.path.join(sys.argv[1], n + ("-dflash2" if df else "") + ".env")
            with open(fn, "w") as f:
                extra = " + QWEN36_DFLASH2=1" if df else ""
                f.write(f"# generated by models/demos/blackhole/qwen36/presets.py (ep2/best): QWEN36_PRESET={n}{extra}\n")
                f.writelines(f"{k}={v}\n" for k, v in switches(n, df).items())
            print(fn)

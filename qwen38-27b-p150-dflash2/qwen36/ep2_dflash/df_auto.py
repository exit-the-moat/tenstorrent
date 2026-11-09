"""DF_AUTO=1 : ONE flag for the gated DFlash2 multi-stream configuration (ep2/dfa).

Self-contained: everything start_t.sh / start_u.sh passed as EXTRA_ENV is filled in here, with setdefault semantics
(an explicit env var always wins). Must run before the qwen36 package is imported (its presets.py reads the QWEN36_*
switches once at import), so df_server.py calls apply() right at the top. The launcher drops the keys in KEYS from the
preset env file (best2card-dflash2.env sets QWEN36_PF=1, DF_POOL=65536, DF_WARM_LENS=...) so the values here apply, and
mounts the one-line model_config.py patch (supported_batches += 64, 128: the model is built at 64 rows for the batched
drafter's all-reduce, every verify pass stays <= 32 rows).

The schedule (Stages T/U/V, r64t/r64u/r64v):
  k = max(1, min(8, 32 // live))   every verify pass is live * k <= 32 rows: ONE tile row, never a 64-row pass
  speculate while live <= 6        +153 % per stream at 1 live, +52 % at 4, +15 % at 6; loses from 7
  live >= 7: the batched plain lane (the model's own batched decode, rows = stream slots, buckets 8/16/32, one tile row)
  QWEN36_PF=1 + QWEN36_PF_RING_ONLY=1   (dfa a2, 8 Nov) the prefetcher streams the decode weights for every pass
                                   (all <= 32 rows = PF_MAX_ROWS), and its ring is the only resident copy: weights
                                   1.43 GiB/bank vs 1.39 with PF=0 (no ring). Verify 44.4 -> 35.0 ms at 1 live
                                   (106 -> 131 tok/s), 88 -> 79 ms at 4, plain lane 21.5 -> 25.9 tok/s at 8 live.
                                   -e QWEN36_PF=0 = the Stage T/V configuration.
  DF_MS_ROWS=0                     the row kernels' R=30 (6 x 5) verifier is broken: opt-in only

Streams: DF_STREAMS (default 8) slots; KV pool DF_POOL = 65,536 tokens per slot unless set; the plain lane's buckets are
those <= min(32, NS); the model's batched GDN state has the largest of them as rows (DF_MODEL_GDN_B_NROWS).
Spec verifiers: only the live counts that can speculate (1..DF_PLAIN_LIVE-1 when the lane is armed), not 1..NS:
in spec mode no live slot index is >= PLAIN_LIVE - 1 (slots are filled lowest first and a 7th live request moves
every request to the lane until it drains), so the 7x4 .. NSx1 verifiers were built and never run.

  python3 df_auto.py --keys   # the owned keys as an ERE alternation (launcher: filter the preset env file)
  python3 df_auto.py --env    # what apply() would set from an empty env
"""
import os
import sys

FIXED = (
    ("DF_STREAMS", "8"),
    ("DF_SPEC_KCAP", "8"), ("DF_SPEC_ROWS", "32"), ("DF_SPEC_K", "auto"),
    ("DF_MS_ROWS", "0"), ("DF_MS_MULTI", "1"), ("DF_MS_CHAIN", "0"), ("DF_MS_SUB", "none"),
    ("DF_MODEL_GDN_B", "1"), ("DF_MS_BEGIN_DIRECT", "1"), ("DF_MS_BDRAFT", "2"), ("DF_MS_GCB_BATCH", "1"),
    ("QWEN36_ATTN_FUSED_MAXB", "64"), ("QWEN36_ATTN_FUSED_SDPA_CHUNK", "64"),
    ("QWEN36_PF", "1"), ("QWEN36_PF_RING_ONLY", "1"),
    ("DF_LEN", "65536"), ("DF_WARM_LENS", "100,1000"),
    ("DF_PLAIN_LANE", "1"), ("DF_PLAIN_LIVE", "7"), ("DF_PLAIN_BUCKETS", "8,16,32"),
    # main 8 Nov: the DFlash2 prefix cache (dfpc, gated on two starts on1/on2 with DF_AUTO; snapshot slots default
    # 2 x NS = 16 at NS 8, 2.30 GiB/card). -e DF_PREFIX_CACHE=0 turns it off.
    ("DF_PREFIX_CACHE", "1"),
)
DERIVED = ("DF_POOL", "DF_MODEL_GDN_B_NROWS", "DF_SPEC_PLAN", "DF_MS_LANE_SLOTS")
KEYS = tuple(k for k, _ in FIXED) + DERIVED
PER_SLOT = 65536
POOL_MAX = 524288


def plan(env):
    ns = int(env["DF_STREAMS"])
    kcap, rows = int(env["DF_SPEC_KCAP"]), int(env["DF_SPEC_ROWS"])
    top = min(ns, int(env["DF_PLAIN_LIVE"]) - 1) if env.get("DF_PLAIN_LANE") == "1" else ns
    return ",".join(f"{L}:{max(1, min(kcap, rows // L))}" for L in range(1, max(1, top) + 1))


def compute(env):
    """The values apply() sets into `env` (a dict-like), setdefault semantics; returns {key: (value, explicit)}."""
    out = {}

    def sd(k, v):
        exp = k in env and env[k] != ""
        if not exp:
            env[k] = v
        out[k] = (env[k], exp)

    for k, v in FIXED:
        sd(k, v)
    ns = int(env["DF_STREAMS"])
    sd("DF_POOL", str(min(PER_SLOT * ns, POOL_MAX)))  # dfa: 524288 fits at NS 8 and NS 16 (n16a: 655360 OOM)
    b = [int(x) for x in env["DF_PLAIN_BUCKETS"].split(",") if x.strip() and int(x) <= min(32, ns)]
    sd("DF_MODEL_GDN_B_NROWS", str(max(b) if b else 1))
    sd("DF_SPEC_PLAN", plan(env))
    sd("DF_MS_LANE_SLOTS", "1" if ns > 8 else "0")  # dfa: NS > 8 only fits with lane-only slots (NS 8 = as gated)
    return out


def apply():
    if os.environ.get("DF_AUTO", "0") != "1" or os.environ.get("_DF_AUTO_DONE") == "1":
        return
    out = compute(os.environ)
    os.environ["_DF_AUTO_DONE"] = "1"
    exp = sorted(k for k, (_v, e) in out.items() if e)
    print("[DF_AUTO] k = max(1, min(8, 32 // live)); speculate <= 6 live; plain lane above; one 32-row tile row; "
          f"{len(out)} switches, explicit env wins for: {', '.join(exp) or 'none'}", flush=True)
    print("[DF_AUTO] " + " ".join(f"{k}={v}" for k, (v, _e) in sorted(out.items())), flush=True)


if __name__ == "__main__":
    if "--keys" in sys.argv:
        print("|".join(KEYS))
    else:
        e = {"DF_AUTO": "1"}
        for a in sys.argv[1:]:
            if "=" in a:
                k, v = a.split("=", 1)
                e[k] = v
        for k, (v, x) in compute(e).items():
            print(f"{k}={v}" + ("   # explicit" if x else ""))

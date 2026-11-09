# SPDX-License-Identifier: Apache-2.0
"""ep2 pcx QWEN36_PREFIX_CACHE: the GDN snapshot index shared by the model and vLLM's KV cache manager.

Both run in the EngineCore process. The model (Qwen36Model._pcx_*) mirrors its snapshot index here; the patched
vllm/v1/core/kv_cache_manager.py (get_computed_blocks) caps every prefix-cache hit at the largest 2048 boundary whose
GDN state is in KEYS and pins that key until the step's prefill has run. So a hit never needs a recompute of shared
KV blocks: the tokens past the cap get fresh blocks. The manager finds this module through sys.modules[ALIAS] (no
import of the model package from vLLM); when it is absent (model not loaded yet), the hit is capped at 0.
Pure Python, no ttnn."""
import hashlib
import sys

ALIAS = "ep2_pcx_registry"
CHUNK = 2048
KEYS = {}  # sha1 hex of int32 ids [0, k*CHUNK) -> slot (mirror of the model's LRU index)
PINS = set()  # keys a scheduled request will restore in the coming prefill: never evicted before it ran
STATS = {"capped": 0, "capped_tokens": 0, "uncapped": 0}


def boundary_keys(token_ids, kmax, chunk=CHUNK):
    """{k: sha1 of the little-endian int32 ids [0, k*chunk)} for k = 1..kmax, one incremental pass.
    token_ids: any sequence of ints (list, ConstantList, numpy array, torch tensor row)."""
    import array

    h = hashlib.sha1()
    keys = {}
    for k in range(1, kmax + 1):
        seg = token_ids[(k - 1) * chunk : k * chunk]
        if hasattr(seg, "numpy"):
            seg = seg.numpy()
        if hasattr(seg, "astype"):
            b = seg.astype("<i4").tobytes()
        else:
            b = array.array("i", [int(x) for x in seg]).tobytes()
        h.update(b)
        keys[k] = h.hexdigest()
    return keys


def cap_hit(token_ids, num_cached, chunk=CHUNK):
    """Largest k*chunk <= num_cached with a stored GDN state for ids [0, k*chunk); pins that key. Returns tokens."""
    kmax = int(num_cached) // chunk
    if kmax <= 0:
        return 0
    keys = boundary_keys(token_ids, kmax, chunk)
    for k in range(kmax, 0, -1):
        if keys[k] in KEYS:
            PINS.add(keys[k])
            if k < kmax or k * chunk < num_cached:
                STATS["capped"] += 1
                STATS["capped_tokens"] += int(num_cached) - k * chunk
            else:
                STATS["uncapped"] += 1
            return k * chunk
    STATS["capped"] += 1
    STATS["capped_tokens"] += int(num_cached)
    return 0


sys.modules.setdefault(ALIAS, sys.modules[__name__])

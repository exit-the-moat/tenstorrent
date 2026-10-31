p = "tt/backbone_paged.py"
s = open(p).read()
old = """        shape = [n_blocks * int(os.environ.get("DRAMA3_KV_STREAMS", "1")), args.n_kv_heads, BLOCK, args.head_dim]
"""
new = """        shape = [kv_pool_blocks(n_blocks), args.n_kv_heads, BLOCK, args.head_dim]
"""
assert old in s
s = s.replace(old, new, 1)
i = s.index("\nclass ")
s = s[:i] + '''

def kv_pool_blocks(n_blocks):
    """K/V pool size in blocks for DRAMA3_KV_STREAMS sequences. port10 DRAMA3_KV_COMPACT=1: the shared voice-prefix
    blocks (DRAMA3_KV_PREFIX_BLOCKS, default 10 = 640 rows) exist once (stream 0's), the other streams get
    n_blocks - prefix blocks each (host/batch.py maps them the same way)."""
    n = int(os.environ.get("DRAMA3_KV_STREAMS", "1"))
    if os.environ.get("DRAMA3_KV_COMPACT", "0") == "1":
        return n_blocks + (n - 1) * (n_blocks - int(os.environ.get("DRAMA3_KV_PREFIX_BLOCKS", "10")))
    return n_blocks * n
''' + s[i:]
open(p, "w").write(s)

p = "tt/tower_device.py"
s = open(p).read()
old = """        shape = [max_seq // BLOCK * int(os.environ.get("DRAMA3_KV_STREAMS", "1")), nkv, BLOCK, hd]  # batch mode: a pool
"""
new = """        from models.demos.blackhole.drama3.tt.backbone_paged import kv_pool_blocks

        shape = [kv_pool_blocks(max_seq // BLOCK), nkv, BLOCK, hd]  # batch mode: a pool (port10: compact option)
"""
assert old in s
s = s.replace(old, new, 1)
open(p, "w").write(s)

p = "host/batch.py"
s = open(p).read()
old = """        rows = [list(range(nP)) + [b * nb + j for j in range(nP, nb)] for b in range(B)]
"""
new = """        rows = [list(range(nP)) + [b * nb + j for j in range(nP, nb)] for b in range(B)]
        if os.environ.get("DRAMA3_KV_COMPACT", "0") == "1":  # port10: the pool holds the prefix blocks once
            kp = int(os.environ.get("DRAMA3_KV_PREFIX_BLOCKS", "10"))
            assert nP >= kp, ("the voice prefix is shorter than DRAMA3_KV_PREFIX_BLOCKS", nP, kp)
            rows = [list(range(nb))] + [list(range(nP)) + [nb + (b - 1) * (nb - kp) + (j - kp) for j in range(nP, nb)]
                                        for b in range(1, B)]
"""
assert old in s
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("ok")

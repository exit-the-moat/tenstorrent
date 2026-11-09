# SPDX-License-Identifier: Apache-2.0
"""ep2 t3b DF_CHAIN: the drafter greedy ids combined across the 2 cards on the card (exact vs the host rule;
test: not included in this package)."""
import ttnn


def combine_dev(mesh, ccl, idx, mx, tok, shard, row0):
    """idx [1,1,8] uint32 RM (per-device shard argmax), mx [1,1,8,1] bf16 RM (the logit there), tok [8] uint32 RM
    (row 0 = anchor), row0 [1,1,8,1] fp32 TILE (1 at row 0). Returns [8,1] uint32 RM = [anchor, ids 1..7] on every
    device: ids = device-1 candidate (+ shard) where its value is strictly larger, else device 0 (host rule).
    The ids cross the cards as three 6-bit digits (exact in bf16: concat / all-gather carry bf16), rebuilt in fp32."""
    from models.tt_transformers.tt.ccl import tt_all_gather

    f32 = ttnn.float32
    n = int(idx.shape[-1])
    it = ttnn.typecast(ttnn.to_layout(ttnn.reshape(idx, (1, 1, n, 1)), ttnn.TILE_LAYOUT), f32)
    if mesh.get_num_devices() == 1:  # df1c TP1: one vocab shard, the argmax id is the draft id
        tf = ttnn.typecast(ttnn.to_layout(ttnn.reshape(tok, (1, 1, n, 1)), ttnn.TILE_LAYOUT), f32)
        blk = ttnn.where(row0, tf, it)
        out = ttnn.to_layout(ttnn.typecast(blk, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(out, (n, 1))
    a = ttnn.floor(ttnn.multiply(it, 1.0 / 4096))
    r = ttnn.subtract(it, ttnn.multiply(a, 4096.0))
    b = ttnn.floor(ttnn.multiply(r, 1.0 / 64))
    c = ttnn.subtract(r, ttnn.multiply(b, 64.0))
    vt = ttnn.typecast(ttnn.to_layout(mx, ttnn.TILE_LAYOUT), f32)
    iv = ttnn.concat([a, b, c, vt], dim=-1)  # [1,1,8,4]
    g = tt_all_gather(iv, mesh, ccl, cluster_axis=None, dim=1, topology=ttnn.Topology.Linear)  # [1,2,8,4]
    part = lambda d, k: ttnn.typecast(ttnn.slice(g, (0, d, 0, k), (1, d + 1, n, k + 1)), f32)
    ids_ = []
    for d in range(2):
        ids_.append(ttnn.add(ttnn.add(ttnn.multiply(part(d, 0), 4096.0), ttnn.multiply(part(d, 1), 64.0)), part(d, 2)))
    better = ttnn.gt(part(1, 3), part(0, 3))
    ids = ttnn.where(better, ttnn.add(ids_[1], float(shard)), ids_[0])
    tf = ttnn.typecast(ttnn.to_layout(ttnn.reshape(tok, (1, 1, n, 1)), ttnn.TILE_LAYOUT), f32)
    blk = ttnn.where(row0, tf, ids)
    out = ttnn.to_layout(ttnn.typecast(blk, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(out, (n, 1))

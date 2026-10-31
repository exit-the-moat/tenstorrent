import torch

import ttnn

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    B, k = 2, 8
    src = torch.randint(0, 256, (1, 1, B, k), dtype=torch.int32)
    scores = ttnn.from_torch(torch.randn(1, 1, B, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    v, idx = ttnn.topk(scores, k=k, dim=-1)
    print("topk idx", idx.dtype, idx.layout, idx.shape, flush=True)
    ref = ttnn.to_torch(idx).reshape(-1)
    for name in ("u32_rm", "u32_tile"):
        try:
            if name == "u32_rm":
                t = ttnn.typecast(idx, ttnn.uint32)
                t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
                t = ttnn.reshape(t, (1, 1, 1, B * k))
            else:
                t = ttnn.typecast(idx, ttnn.uint32)
                t = ttnn.reshape(t, (1, 1, 1, B * k))
                t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
            print(name, "reshape ok", t.dtype, t.layout, t.shape, torch.equal(ttnn.to_torch(t).reshape(-1).long(), ref.long()), flush=True)
            for back in ("tile_typecast",):
                u = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
                u = ttnn.typecast(u, ttnn.uint16)
                u = ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT)
                print(name, back, u.dtype, u.layout, u.shape, torch.equal(ttnn.to_torch(u).reshape(-1).long(), ref.long()), flush=True)
        except Exception as e:
            print(name, "FAILED", str(e)[:300], flush=True)
finally:
    ttnn.close_device(mesh)
print("DONE")

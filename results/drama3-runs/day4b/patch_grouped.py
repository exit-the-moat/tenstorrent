p = "/opt/ep0/tt-metal/models/demos/blackhole/qwen36/tt/moe/prefill_grouped.py"
s = open(p).read()
s = s.replace("""import torch

import ttnn
""", """import os

import torch

import ttnn
""", 1)
s = s.replace('''def _u32(mesh_device, t):''', '''def _bmm_config(mesh_device, M, K, N, down):
    """Batched expert matmul [1,E,M,K] x [1,E,K,N] (one expert per core block, full N per core): the default
    program re-reads the weights per 32-row block and runs at ~37 GB/s (7.6 ms for gate_up at M = 32 on a P150).
    Measured (tests/sweep_grouped_mm.py, ms gate_up / down): M=32 1.36 / 1.26, 64 1.44 / 1.31, 128 1.66 / 2.57,
    640 8.1 / default 9.98 (the full-N output block of down caps per_core_M at 2 in L1). None = the default.
    QWEN36_GROUPED_BMM_CONFIG=0 restores the default everywhere."""
    if os.environ.get("QWEN36_GROUPED_BMM_CONFIG", "1") != "1":
        return None
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    if down:
        if Mt <= 2:
            pcm, ibw = Mt, 4
        elif Mt == 4:
            pcm, ibw = 2, 2
        else:
            return None
    else:
        pcm, ibw = next(m for m in (4, 3, 2, 1) if Mt % m == 0), 8
    g = mesh_device.compute_with_storage_grid_size()
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y),
        in0_block_w=min(ibw, Kt),
        out_subblock_h=1,
        out_subblock_w=4 if Nt % 4 == 0 else 1,
        per_core_M=pcm,
        per_core_N=Nt,
    )


def _u32(mesh_device, t):''')
a = "    ug = ttnn.matmul(xg, weights.gate_up_proj, compute_kernel_config=ckc, dtype=ttnn.bfloat16)  # [1, E, M, 2I]"
assert a in s
s = s.replace(a, """    gu, dp = weights.gate_up_proj, weights.down_proj
    cfg_gu = _bmm_config(mesh_device, M, gu.shape[-2], gu.shape[-1], down=False)
    cfg_dn = _bmm_config(mesh_device, M, dp.shape[-2], dp.shape[-1], down=True)
    ug = ttnn.matmul(xg, gu, program_config=cfg_gu, compute_kernel_config=ckc, dtype=ttnn.bfloat16)  # [1, E, M, 2I]""")
b = "    dn = ttnn.matmul(act, weights.down_proj, compute_kernel_config=ckc, dtype=ttnn.bfloat16)  # [1, E, M, H]"
assert b in s
s = s.replace(b, "    dn = ttnn.matmul(act, dp, program_config=cfg_dn, compute_kernel_config=ckc, dtype=ttnn.bfloat16)  # [1, E, M, H]")
open(p, "w").write(s)
print("patched")

p = "/opt/ep0/tt-metal/models/demos/blackhole/qwen36/tt/moe/prefill_grouped.py"
s = open(p).read()
old = '''    ug = ttnn.matmul(xg, gu, program_config=cfg_gu, compute_kernel_config=ckc, dtype=ttnn.bfloat16)  # [1, E, M, 2I]'''
assert s.count(old) == 1
s = s.replace(old, '''    # QWEN36_GROUPED_GU_GRID_MIN_M=256 (opt-in): gate_up on an 8x8 core grid from this M up: the program ttnn
    # then picks is bit-identical to the default one at M = 640 and 1.8x faster (16.8 -> 9.2 ms, sweep_grouped_grid.py)
    gu_kw = {}
    min_m = int(os.environ.get("QWEN36_GROUPED_GU_GRID_MIN_M", "0"))
    if cfg_gu is None and min_m and M >= min_m:
        gu_kw["core_grid"] = ttnn.CoreGrid(y=8, x=8)
    ug = ttnn.matmul(xg, gu, program_config=cfg_gu, compute_kernel_config=ckc, dtype=ttnn.bfloat16, **gu_kw)  # [1, E, M, 2I]''')
open(p, "w").write(s)
print("ok")

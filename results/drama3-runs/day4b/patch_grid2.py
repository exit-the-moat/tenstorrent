p = "/opt/ep0/tt-metal/models/demos/blackhole/qwen36/tt/moe/prefill_grouped.py"
s = open(p).read()
old = '''    if cfg_gu is None and min_m and M >= min_m:
        gu_kw["core_grid"] = ttnn.CoreGrid(y=8, x=8)'''
assert s.count(old) == 1
s = s.replace(old, '''    if cfg_gu is None and min_m and M >= min_m:
        gu_kw["core_grid"] = ttnn.CoreGrid(y=8, x=8)
    # QWEN36_GROUPED_SMALL_GRID=1 (opt-in): below M = 256 both matmuls on the full 11x10 grid (M = 32: gate_up
    # 7.6 -> 1.4 ms, down 2.4 -> 1.3 ms; NOT bit-identical to the default: max |diff| 0.125 / 0.03)
    dn_kw = {}
    if os.environ.get("QWEN36_GROUPED_SMALL_GRID", "0") == "1" and M < 256:
        if cfg_gu is None:
            gu_kw["core_grid"] = ttnn.CoreGrid(y=10, x=11)
        dn_kw["core_grid"] = ttnn.CoreGrid(y=10, x=11)''')
old = '''    dn = ttnn.matmul(act, dp, program_config=cfg_dn, compute_kernel_config=ckc, dtype=ttnn.bfloat16)  # [1, E, M, H]'''
assert s.count(old) == 1
s = s.replace(old, '''    if cfg_dn is not None:
        dn_kw = {}
    dn = ttnn.matmul(act, dp, program_config=cfg_dn, compute_kernel_config=ckc, dtype=ttnn.bfloat16, **dn_kw)  # [1, E, M, H]''')
open(p, "w").write(s)
print("ok")

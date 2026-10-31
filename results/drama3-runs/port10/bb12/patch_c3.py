p = "/opt/ep0/tt-metal/models/demos/blackhole/drama3/tests/check_bb_prefetch2.py"
s = open(p).read()
i = s.index("        res = {}\n        S0, S1")
j = s.index("        json.dump(res")
new = """        res = {}
        nsteps = int(os.environ.get("BB12_STEPS", "6"))
        steps = [torch.randn(args.dim, generator=g) * 0.05 for _ in range(nsteps)]

        def lg(on):
            bb.pf = pf if on else None
            bb.reset()
            bb.prefill(E)
            out = [bb.decode_eager(e, 128 + j)[1] for j, e in enumerate(steps)]
            bb.pf = pf
            return out

        def cmp(a, b):
            return [int(not torch.equal(x, y)) for x, y in zip(a, b)]

        A0 = lg(False)  # no global CB yet
        pf.pf.start()
        A1 = lg(False)  # global CB allocated, prefetcher running idle
        B = lg(True)
        A2 = lg(False)
        B2 = lg(True)
        res = dict(gcb_vs_nogcb=cmp(A0, A1), on_vs_off=cmp(A1, B), off_after_on=cmp(A1, A2), on_repeat=cmp(B, B2))
        print("ab", json.dumps(res), flush=True)
"""
s = s[:i] + new + s[j:]
open(p + ".tmp", "w").write(s)
import os

os.replace(p + ".tmp", p)
print("ok")

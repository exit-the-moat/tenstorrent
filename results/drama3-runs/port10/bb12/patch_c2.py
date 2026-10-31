p = "/opt/ep0/tt-metal/models/demos/blackhole/drama3/tests/check_bb_prefetch2.py"
s = open(p).read()
old = """        res = {}
        for collect in (False,):"""
new = """        def state_after_step0(on):
            bb.pf = pf if on else None
            bb.reset()
            bb.prefill(E)
            bb.decode_eager(steps[0], 128)
            bb.pf = pf
            st = {}
            for li, (dn, rec, conv) in enumerate(bb.gdn_state):
                st[f"gdn{li}.rec"] = ttnn.to_torch(rec).float()
                q = dn.split_conv_state
                rows = list(q) if q is not None else []
                for qi, t in enumerate(rows):
                    st[f"gdn{li}.conv{qi}"] = ttnn.to_torch(t).float()
            for ai, (k, v) in enumerate(bb.kv):
                st[f"kv{ai}.k"] = ttnn.to_torch(k).float()[:4]
                st[f"kv{ai}.v"] = ttnn.to_torch(v).float()[:4]
            return st

        res = {}
        S0, S1 = state_after_step0(False), state_after_step0(True)
        diff = [(k, int((S0[k] != S1[k]).sum()), float((S0[k] - S1[k]).abs().max())) for k in S0 if not torch.equal(S0[k], S1[k])]
        res["state_diff"] = diff
        print("state_diff", len(diff), "of", len(S0), diff[:12], flush=True)
        for collect in (False,):"""
assert s.count(old) == 1
s = s.replace(old, new)
open(p + ".tmp", "w").write(s)
import os

os.replace(p + ".tmp", p)
print("ok")

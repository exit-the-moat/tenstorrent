# SPDX-License-Identifier: Apache-2.0
"""ep2 pf4 (QWEN36_PF_MTRACE="128,256,..."): traced masked-bucket prefill (short prompts and long-prompt tails).

The eager masked-bucket forward builds its valid_len-dependent tensors on the host per request (the FIR one-hot state
select, the GDN chunk-kernel masks) and dispatches ~120 ops per layer from Python: a 26-token prompt costs ~350 ms,
a 12-token tail of an 8k prompt ~300 ms. Here each listed bucket's forward (embedding -> 64 layers -> hidden) is
captured once at warm-up with those tensors as persistent device buffers (CTX below) and replayed per request after
copying the request's values into them. The math and the programs are the eager masked forward's, so outputs are
bit-identical. Differences that do not touch real rows: the K/V fill writes the whole bucket (rows past the prompt
land in the request's last block, as eager does up to the block end, or in vLLM's null block 0), and the GDN masks
are applied also when valid_len == bucket (a multiply by 1).
"""
import os

CTX = None  # dict of the bucket being captured / replayed: fir_sel, m, mq (device tensors), else None


def buckets():
    v = os.environ.get("QWEN36_PF_MTRACE", "")
    return tuple(int(b) for b in v.split(",") if b.strip() and b.strip() != "0")

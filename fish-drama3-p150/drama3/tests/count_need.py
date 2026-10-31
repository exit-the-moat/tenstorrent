# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""port7 TTFA: per-layer row need of the grouped-MoE suffix prefill (the busiest expert's row count, which picks M)
and the number of active experts, over a normal host.generate_fast run (counter around _grouped_we)."""
import json
import sys

from models.demos.blackhole.qwen36.tt.moe import prefill_grouped as pg

REC = []
_orig = pg._grouped_we


def counted(hidden_states, w, e, rows, weights, config, mesh_device, force_M=None):
    import torch

    if force_M is None:
        c = torch.bincount(e.reshape(-1), minlength=config.num_experts)
        REC.append(dict(S=int(hidden_states.shape[-2]), n=int(rows.numel()), need=int(c.max()), active=int((c > 0).sum())))
    return _orig(hidden_states, w, e, rows, weights, config, mesh_device, force_M)


pg._grouped_we = counted

if __name__ == "__main__":
    i = sys.argv.index("--need-out")
    out = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    from models.demos.blackhole.drama3.host import generate_fast

    try:
        generate_fast.main()
    finally:
        sfx = [r for r in REC if r["S"] <= 128]
        res = dict(calls=len(REC), suffix_calls=len(sfx))
        if sfx:
            nd = [r["need"] for r in sfx]
            res.update(need_max=max(nd), over32=sum(x > 32 for x in nd), over64=sum(x > 64 for x in nd), over96=sum(x > 96 for x in nd),
                       active_mean=sum(r["active"] for r in sfx) / len(sfx), rows_mean=sum(r["n"] for r in sfx) / len(sfx))
        json.dump(dict(summary=res, calls=REC), open(out, "w"))
        print("NEED", json.dumps(res), flush=True)

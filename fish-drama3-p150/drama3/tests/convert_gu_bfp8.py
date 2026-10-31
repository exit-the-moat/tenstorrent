# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Day 5: write BFP8 gate/up expert cache files for the given layers (next to the BFP4 ones), one layer at a time
with an os.sync() and a pause after each (the server's SATA SSD has dropped writes under load).

    QWEN36_EXPERT_GU_BFP8_LAYERS=<list> python -m models.demos.blackhole.drama3.tests.convert_gu_bfp8 --layers <list>
"""
import argparse
import hashlib
import os
import time

import ttnn

for k, v in dict(HF_MODEL="/opt/ep0/drama3_shim", TT_CACHE_PATH="/opt/ep0/cache/drama3", MAX_PREFILL_CHUNK_SIZE="32").items():
    os.environ.setdefault(k, v)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--pause", type=float, default=5.0)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    assert os.environ.get("QWEN36_EXPERT_GU_BFP8_LAYERS"), "set QWEN36_EXPERT_GU_BFP8_LAYERS to the same layers"
    from models.demos.blackhole.drama3.tt.device_check import build_layer, free_layer
    from models.demos.blackhole.drama3.tt.weight_mapping import weight_map
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs

    wmap = weight_map("/models/drama-3")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE)
    try:
        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=4096, enable_mtp=False)
        cache = args.weight_cache_path()
        for i in layers:
            t0 = time.time()
            layer, t_read, t_build = build_layer(mesh, args, i, cache, wmap)
            free_layer(layer)
            del layer
            os.sync()
            # hash from the page cache (= the bytes written); convert_gu.sh re-hashes from disk after drop_caches
            with open("/opt/ep0/day5/cache_bfp8.written.sha256", "a") as f:
                for n in ("gate_proj", "up_proj"):
                    p = cache / f"layers.{i}" / f"moe.experts.{n}_dtype_BFLOAT8_B_layout_TILE.tensorbin"
                    h = hashlib.sha256()
                    with open(p, "rb") as fh:
                        for blk in iter(lambda: fh.read(1 << 24), b""):
                            h.update(blk)
                    f.write(f"{h.hexdigest()}  layers.{i}/{p.name}\n")
            log(f"layer {i}: read {t_read:.1f}s build {t_build:.1f}s total {time.time() - t0:.1f}s")
            time.sleep(a.pause)
    finally:
        ttnn.close_device(mesh)
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
        raise

"""Episode 3 harness helpers: open the mesh, map a split name to the tt_dit parallel configs."""
import os, time, json
import ttnn
from tests.scripts.common import get_updated_device_params

DTYPES = {"bf16": None, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b, "bf8m4": ttnn.bfloat8_b, "bf4x": ttnn.bfloat8_b}
if os.environ.get("EP3_WEIGHTS_HINT") == "bf8m4":
    os.environ.setdefault("TT_DIT_BIG_LINEAR_BF4_MIN", "50000000")
# split -> parallel config tuples; encoder/vae follow the DiT's tensor split.
SPLITS = {
    # (b) tensor split: one DiT over both cards, CFG branches in the batch
    "tp": dict(cfg=(1, 0), sp=(1, 0), tp=(2, 1), enc=(2, 1), vae=(2, 1)),
    # (a) guidance per card: each card holds a whole DiT and computes one CFG branch per step
    "cfg": dict(cfg=(2, 1), sp=(1, 0), tp=(1, 0), enc=(1, 0), vae=(1, 0)),
}


def open_mesh(shape=(1, 2), trace_region=200_000_000, fabric=True):
    params = {"trace_region_size": trace_region}
    if fabric:
        params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING if __import__("os").environ.get("EP3_FABRIC") == "ring" else ttnn.FabricConfig.FABRIC_1D
    params = get_updated_device_params(params)
    fc = params.pop("fabric_config", None)
    params.pop("fabric_tensix_config", None)
    if fc is not None:
        ttnn.set_fabric_config(fc, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED, ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), **params)
    return mesh, fc


def close_mesh(mesh, fc):
    for s in mesh.get_submeshes():
        ttnn.close_mesh_device(s)
    ttnn.close_mesh_device(mesh)
    if fc is not None:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def dram_mb(mesh):
    """Allocated DRAM per chip in MB, from the allocator (mesh view: per chip, all chips in lockstep)."""
    try:
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        return round(v.total_bytes_allocated_per_bank * v.num_banks / 1e6)
    except Exception as e:  # noqa: BLE001
        return f"n/a ({type(e).__name__}: {e})"


def stamp():
    return time.strftime("%H:%M:%S")


def log(path, **kw):
    with open(path, "a") as f:
        f.write(json.dumps(kw) + "\n")

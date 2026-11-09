# SPDX-License-Identifier: Apache-2.0
"""QWEN36_LAZY_LOAD=1: build the model from the tensor cache without reading the bf16 checkpoint.

The normal start loads the full HF model (52 GB bf16 for the 27B) and builds every torch weight (slice, transpose,
reorder) before ttnn.as_tensor, which then ignores it whenever its cache file exists. With 62 GB of RAM the bf16
pages and the tensor cache evict each other, so every start re-reads both from the SSD (~18 min of layer loading).

Lazy mode instead:
  - reads only the safetensors headers; every 2-D weight (projections, embedding, LM head) becomes a `meta` tensor
    (shape and dtype, no data); the rest (1-D norms, A_log, dt_bias; 3-D conv1d) is read for real (a few MB). 2-D
    is all or nothing because the TP loaders concatenate small 2-D weights (GDN in_proj_a/b) into large ones, and
    torch.cat refuses to mix meta and CPU tensors;
  - runs the unchanged model constructor: torch ops on meta tensors only compute shapes, and ttnn.as_tensor on a
    cache hit returns load_tensor_flatbuffer(<file>) without looking at the torch tensor, so the device tensors are
    the cached ones, byte for byte, exactly as in the normal path;
  - guards ttnn.as_tensor / ttnn.from_torch: a meta tensor that would need its data (no cache file, or a cache file
    that fails to load) raises LazyCacheMiss before anything is converted or written. create_tt_model then drops
    the partial model and does the normal full load (which also writes the missing cache files, as before).

Only plain BF16 safetensors checkpoints go lazy (the normal path loads them with dtype="auto", i.e. unchanged bf16);
anything else falls back to the normal load.
"""
import json
import os
import struct
from contextlib import contextmanager
from pathlib import Path

import torch
from loguru import logger

_DTYPES = {"BF16": torch.bfloat16}


class LazyCacheMiss(RuntimeError):
    """A lazy (meta) weight was needed as data: its cache file is missing or unreadable."""


def enabled():
    return os.environ.get("QWEN36_LAZY_LOAD", "0") == "1"


def _read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def _checkpoint_files(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    index = ckpt_dir / "model.safetensors.index.json"
    if index.is_file():
        with open(index) as f:
            return [ckpt_dir / n for n in sorted(set(json.load(f)["weight_map"].values()))]
    single = ckpt_dir / "model.safetensors"
    if single.is_file():
        return [single]
    raise LazyCacheMiss(f"no safetensors checkpoint in {ckpt_dir}")


def load_raw_state_dict(ckpt_dir):
    """Raw checkpoint keys -> meta tensors (2-D) or real tensors (1-D, 3-D). Vision and MTP keys are skipped (the
    remap drops them anyway)."""
    from safetensors import safe_open

    raw = {}
    small = {}
    for path in _checkpoint_files(ckpt_dir):
        for key, info in _read_header(path).items():
            if "visual" in key or key.startswith("model.visual") or key.startswith("mtp"):
                continue
            if info["dtype"] not in _DTYPES:
                raise LazyCacheMiss(f"{key}: dtype {info['dtype']} (lazy load handles plain BF16 checkpoints only)")
            shape = info["shape"]
            if len(shape) != 2:
                small.setdefault(path, []).append(key)
                raw[key] = None  # keep checkpoint order
            else:
                raw[key] = torch.empty(shape, dtype=_DTYPES[info["dtype"]], device="meta")
    for path, keys in small.items():
        with safe_open(str(path), framework="pt") as sf:
            for key in keys:
                raw[key] = sf.get_tensor(key)
    return raw


def load_lazy_state_dict(ckpt_dir):
    from models.demos.blackhole.qwen36.tt.weight_mapping import is_fp8_checkpoint, remap_qwen36_state_dict

    if is_fp8_checkpoint(ckpt_dir):
        raise LazyCacheMiss("FP8 checkpoint (lazy load handles plain BF16 checkpoints only)")
    return remap_qwen36_state_dict(load_raw_state_dict(ckpt_dir))


def _cache_file(cache_file_name, dtype, layout):
    # Same name ttnn.as_tensor builds (ttnn/operations/core.py).
    dtype_name = dtype.name if dtype is not None else "None"
    layout_name = layout.name if layout is not None else "None"
    return Path(f"{cache_file_name}_dtype_{dtype_name}_layout_{layout_name}.tensorbin")


@contextmanager
def guard(stats):
    """Patch ttnn.as_tensor / ttnn.from_torch so a meta tensor can only ever be served from an existing cache file."""
    import ttnn

    orig_as_tensor = ttnn.as_tensor
    orig_from_torch = ttnn.from_torch

    def as_tensor(tensor, dtype=None, *, layout=ttnn.ROW_MAJOR_LAYOUT, cache_file_name=None, **kwargs):
        if isinstance(tensor, torch.Tensor) and tensor.is_meta:
            if cache_file_name is None:
                raise LazyCacheMiss(f"as_tensor without a cache file for a lazy weight of shape {tuple(tensor.shape)}")
            path = _cache_file(cache_file_name, dtype, layout)
            if not path.is_file():
                raise LazyCacheMiss(f"no cache file {path}")
            stats["hits"] += 1
            stats["bytes"] += path.stat().st_size
        return orig_as_tensor(tensor, dtype, layout=layout, cache_file_name=cache_file_name, **kwargs)

    def from_torch(tensor, *args, **kwargs):
        if isinstance(tensor, torch.Tensor) and tensor.is_meta:
            # Reached when a cache file exists but fails to load (as_tensor then regenerates it from the tensor).
            raise LazyCacheMiss(f"from_torch on a lazy weight of shape {tuple(tensor.shape)}")
        return orig_from_torch(tensor, *args, **kwargs)

    ttnn.as_tensor = as_tensor
    ttnn.from_torch = from_torch
    try:
        yield
    finally:
        ttnn.as_tensor = orig_as_tensor
        ttnn.from_torch = orig_from_torch

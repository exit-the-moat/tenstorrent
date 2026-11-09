# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model construction entry point (gpt_oss/gemma4 `create_tt_model` convention)."""
import gc
import os
import time

from loguru import logger

from models.demos.blackhole.qwen36.tt import lazy_load
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


def create_tt_model(
    mesh_device,
    max_batch_size=1,
    max_seq_len=2048,
    n_layers=None,
    layer_indices=None,
    hf_model=None,
):
    """Build the Qwen3.5-9B model. Returns (args, model, state_dict).

    HF_MODEL (env var) is the single source of truth. `hf_model`, if given, sets it.
    `layer_indices` runs ONLY the listed checkpoint layers (profiling); it takes precedence
    over `n_layers` (first-N truncation). See Qwen36Model.from_pretrained for details.

    QWEN36_LAZY_LOAD=1 builds the model from the tensor cache only (tt/lazy_load.py); on any cache miss it
    falls back to the normal full load. The returned state_dict then holds meta tensors for the large weights.
    """
    if hf_model is not None:
        os.environ["HF_MODEL"] = hf_model

    if lazy_load.enabled():
        t0 = time.time()
        try:
            return _create(mesh_device, max_batch_size, max_seq_len, n_layers, layer_indices, lazy=True)
        except Exception as e:  # noqa: BLE001 - any failure: the normal load is always correct
            logger.warning(f"QWEN36_LAZY_LOAD: {e!r} after {time.time() - t0:.0f} s; falling back to the full load")
        gc.collect()
    return _create(mesh_device, max_batch_size, max_seq_len, n_layers, layer_indices, lazy=False)


def _create(mesh_device, max_batch_size, max_seq_len, n_layers, layer_indices, lazy):
    args = Qwen36ModelArgs(
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    if layer_indices is not None:
        layer_indices = list(layer_indices)
        assert layer_indices, "layer_indices must be non-empty"
        assert all(
            0 <= i < len(args.attention_type_list) for i in layer_indices
        ), f"layer_indices {layer_indices} out of range [0, {len(args.attention_type_list)})"
        args.layer_indices = layer_indices
        args.n_layers = len(layer_indices)
    elif n_layers is not None:
        args.n_layers = n_layers
        args.attention_type_list = args.attention_type_list[:n_layers]

    cache_path = args.weight_cache_path()
    if lazy:
        t0 = time.time()
        logger.info("QWEN36_LAZY_LOAD: safetensors headers + small weights only, large weights from the tensor cache")
        state_dict = lazy_load.load_lazy_state_dict(args.CKPT_DIR)
        stats = {"hits": 0, "bytes": 0}
        with lazy_load.guard(stats):
            model = Qwen36Model(mesh_device, args, state_dict, tensor_cache_path=cache_path)
        logger.info(
            f"QWEN36_LAZY_LOAD: built in {time.time() - t0:.0f} s, {stats['hits']} lazy weights from the cache "
            f"({stats['bytes'] / 1e9:.1f} GB), 0 misses"
        )
        return args, model, state_dict

    logger.info("Loading + remapping weights via Qwen36ModelArgs.load_state_dict()...")
    state_dict = args.load_state_dict()
    model = Qwen36Model(mesh_device, args, state_dict, tensor_cache_path=cache_path)
    return args, model, state_dict

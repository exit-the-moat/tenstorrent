"""Make the GDN depthwise conv1d prefill split its channels into QWEN36_CONV_SPLIT pieces (default 1 = unchanged).

Why (Episode 1, 2026-11-01): at TP=2 each card holds twice the conv channels of the validated TP=4 path; the
height-sharded conv's static circular buffers then reach ~1.09 MB of L1 and clash with a resident L1 buffer
("Statically allocated circular buffers ... clash with L1 buffers", warm-up, chunk 2048). A depthwise conv
(groups = C) is independent per channel, so running it on channel slices and concatenating is the same math.
Usage: patch_conv.py <orig gdn/tp.py> <out>
"""
import sys

src = open(sys.argv[1]).read()

# 1. weights: per-device channel pieces, built at load next to conv_w1d
old_w = '''    tw["conv_w1d"] = ttnn.from_torch(
        W1d,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    return tw'''
new_w = '''    tw["conv_w1d"] = ttnn.from_torch(
        W1d,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    # ep1 patch: QWEN36_CONV_SPLIT=n -> n channel pieces per device (each device's local rows cut into n slices)
    _n = int(os.environ.get("QWEN36_CONV_SPLIT", "1"))
    if _n > 1:
        _cl = W1d.shape[0] // tp
        assert _cl % (_n * 32) == 0, (_cl, _n)
        _cp = _cl // _n
        _Wr = W1d.reshape(tp, _cl, 1, W1d.shape[-1])
        tw["conv_w1d_parts"] = [
            ttnn.from_torch(
                _Wr[:, j * _cp:(j + 1) * _cp].reshape(tp * _cp, 1, W1d.shape[-1]).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            for j in range(_n)
        ]
    return tw'''
assert src.count(old_w) == 1
src = src.replace(old_w, new_w)

# 2. the conv: split path before the original single conv
old_c = '''        # Prepare conv weight once (warmup); avoids host reprocess + keeps traced replay device-only.
        if self._conv1d_wprep is None:'''
new_c = '''        if "conv_w1d_parts" in self.tw:  # ep1 patch: channel-split depthwise conv (see load_gdn weights)
            return self._conv1d_prefill_split(xin, T, Lin, C, cc, conv_cfg, _dram), new_state
        # Prepare conv weight once (warmup); avoids host reprocess + keeps traced replay device-only.
        if self._conv1d_wprep is None:'''
assert src.count(old_c) == 1
src = src.replace(old_c, new_c)

old_r = '''    def _row_proj(self, x, weight):'''
new_r = '''    def _conv1d_prefill_split(self, xin, T, Lin, C, cc, conv_cfg, _dram):
        """ep1 patch: the depthwise conv on n channel slices, concatenated (same math as one conv, groups=C)."""
        dev, K = self.mesh, self.K
        parts = self.tw["conv_w1d_parts"]
        n = len(parts)
        cp = C // n
        if getattr(self, "_conv1d_wprep_parts", None) is None:
            self._conv1d_wprep_parts = [
                ttnn.prepare_conv_weights(
                    weight_tensor=w, input_memory_config=_dram, input_layout=ttnn.ROW_MAJOR_LAYOUT,
                    weights_format="OIHW", in_channels=cp, out_channels=cp, batch_size=1, input_height=1,
                    input_width=Lin, kernel_size=(1, K), stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                    has_bias=False, groups=cp, device=dev, input_dtype=ttnn.bfloat16, conv_config=conv_cfg,
                    compute_config=cc,
                )
                for w in parts
            ]
        outs = []
        for j, wp in enumerate(self._conv1d_wprep_parts):
            xj = ttnn.slice(xin, (0, 0, 0, j * cp), (1, Lin, 1, (j + 1) * cp), memory_config=_dram)
            oj = ttnn.conv1d(
                input_tensor=xj, weight_tensor=wp, device=dev, in_channels=cp, out_channels=cp, batch_size=1,
                input_length=Lin, kernel_size=K, stride=1, padding=0, dilation=1, groups=cp, dtype=ttnn.bfloat16,
                conv_config=conv_cfg, compute_config=cc, slice_config=ttnn.Conv2dL1FullSliceConfig,
                return_output_dim=False, return_weights_and_bias=False,
            )
            ttnn.deallocate(xj)
            oj = ttnn.sharded_to_interleaved(oj, _dram)
            outs.append(ttnn.reshape(oj, (1, T, cp)))
        ttnn.deallocate(xin)
        out = ttnn.concat(outs, dim=-1, memory_config=_dram) if n > 1 else outs[0]
        for o in outs:
            if n > 1:
                ttnn.deallocate(o)
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=_dram)
        return ttnn.silu(out, memory_config=_dram)

    def _row_proj(self, x, weight):'''
assert src.count(old_r) == 1
src = src.replace(old_r, new_r)
if "\nimport os\n" not in src:
    src = "import os\n" + src
open(sys.argv[2], "w").write(src)
print("patched")

"""cc13: opt-in core restriction for tt/codec_device.py DeviceDecoder (default path untouched)."""
import sys

f = sys.argv[1]
s = open(f).read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old, s.count(old))
    s = s.replace(old, new)


rep('''KDIR = "models/demos/blackhole/drama3/kernels"
''', '''KDIR = "models/demos/blackhole/drama3/kernels"
# cc13 (opt-in): DRAMA3_CODEC_CORES="x0,y0,x1,y1" (or DeviceDecoder(cores=..., sub_device_id=...)): every program of
# the decoder on that core rectangle only (its own sub-device, so it can run beside the decode on the other cores):
# conv1d core_grid override, matmuls on the sub-device, sub_core_grids on the eltwise / layout ops, the snake kernel
# on those cores, and the channel-0 tail as slice + tanh + untilize (no transpose: it has no core option)
CORES = os.environ.get("DRAMA3_CODEC_CORES", "")


def core_rect(spec):
    """"x0,y0,x1,y1" -> CoreRangeSet of that rectangle (None for "")."""
    if not spec:
        return None
    x0, y0, x1, y1 = (int(v) for v in spec.split(","))
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})
''')

rep('''    def __init__(self, mesh, codec, length):
        """codec: the host codec (fold_weight_norm()'d); length: latent rows per call (window frames * 4)."""
        self.mesh, self.L0 = mesh, length
''', '''    def __init__(self, mesh, codec, length, cores=None, sub_device_id=None):
        """codec: the host codec (fold_weight_norm()'d); length: latent rows per call (window frames * 4).
        cores / sub_device_id (cc13, opt-in): the core rectangle (CoreRangeSet) of the codec sub-device and its id."""
        self.mesh, self.L0 = mesh, length
        self.crs = cores if cores is not None else core_rect(CORES)
        self.sd = sub_device_id
''')

# pointwise convs (matmuls) and the bf16 rounding typecasts
rep('''        if c.k == 1:
            if CONV_IN_BF16 and x.dtype != ttnn.bfloat16:
                x = ttnn.typecast(x, ttnn.bfloat16)
            return ttnn.linear(x, c.w, bias=c.b, compute_kernel_config=self.ckc, dtype=ACT)
''', '''        if c.k == 1:
            if CONV_IN_BF16 and x.dtype != ttnn.bfloat16:
                x = self._typecast(x, ttnn.bfloat16)
            if self.crs is not None:
                return ttnn.linear(x, c.w, bias=c.b, compute_kernel_config=self.ckc, dtype=ACT, sub_device_id=self.sd)
            return ttnn.linear(x, c.w, bias=c.b, compute_kernel_config=self.ckc, dtype=ACT)
''')
rep('''            elif o.startswith("abh="):
                cfg.act_block_h_override = int(o[4:])
''', '''            elif o.startswith("abh="):
                cfg.act_block_h_override = int(o[4:])
        if self.crs is not None:
            cfg.core_grid = self.crs
            cfg.override_sharding_config = True
''')
rep('''    def _conv_inner(self, x, L, c, cfg):
        if CONV_IN_BF16 and x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
''', '''    def _typecast(self, x, dt):
        if self.crs is not None:
            return ttnn.typecast(x, dt, sub_core_grids=self.crs)
        return ttnn.typecast(x, dt)

    def _layout(self, x, layout):
        if self.crs is not None:
            return ttnn.to_layout(x, layout, sub_core_grids=self.crs)
        return ttnn.to_layout(x, layout)

    def _add(self, x, y):
        if self.crs is not None:
            return ttnn.add(x, y, sub_core_grids=self.crs)
        return ttnn.add(x, y)

    def _conv_inner(self, x, L, c, cfg):
        if CONV_IN_BF16 and x.dtype != ttnn.bfloat16:
            x = self._typecast(x, ttnn.bfloat16)
''')
rep('''        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return out
''', '''        if out.layout != ttnn.TILE_LAYOUT:
            out = self._layout(out, ttnn.TILE_LAYOUT)
        return out
''')
# snake kernel cores
rep('''            g = self.mesh.compute_with_storage_grid_size()
            per = max(-(-items // (g.x * g.y)), min(8, items))  # >= 8 tiles per core (fewer runtime args to set)
            nc = -(-items // per)
            cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(nc)]
''', '''            g = self.mesh.compute_with_storage_grid_size()
            x0 = y0 = 0
            if self.crs is not None:  # cc13: the codec rectangle
                bb = self.crs.bounding_box()
                x0, y0 = bb.start.x, bb.start.y
                g = ttnn.CoreCoord(bb.end.x - x0 + 1, bb.end.y - y0 + 1)
            per = max(-(-items // (g.x * g.y)), min(8, items))  # >= 8 tiles per core (fewer runtime args to set)
            nc = -(-items // per)
            cores = [ttnn.CoreCoord(x0 + i % g.x, y0 + i // g.x) for i in range(nc)]
''')
# body
rep('''            if pend is not None and kind not in ("snake", "res"):
                x, pend = ttnn.add(x, pend), None
''', '''            if pend is not None and kind not in ("snake", "res"):
                x, pend = self._add(x, pend), None
''')
rep('''                y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
                y = ttnn.reshape(y, (1, L * s, cout))
                L = L * s
                x = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
''', '''                y = self._layout(y, ttnn.ROW_MAJOR_LAYOUT)
                y = ttnn.reshape(y, (1, L * s, cout))
                L = L * s
                x = self._layout(y, ttnn.TILE_LAYOUT)
''')
rep('''                else:
                    x = ttnn.add(x, y)
            elif kind == "tanh":
                if OUT_CH0:  # [1, L, 32] -> row 0 of [1, 32, L], tanh, row major [1, 1, L]
''', '''                else:
                    x = self._add(x, y)
            elif kind == "tanh":
                if OUT_CH0 and self.crs is not None:  # cc13: column 0 as [1, L, 1], tanh, row major (= L samples)
                    x = ttnn.slice(x, [0, 0, 0], [1, L, 1], sub_core_grids=self.crs)
                    x = ttnn.tanh(x, sub_core_grids=self.crs)
                    x = self._layout(x, ttnn.ROW_MAJOR_LAYOUT)
                elif OUT_CH0:  # [1, L, 32] -> row 0 of [1, 32, L], tanh, row major [1, 1, L]
''')
rep('''                else:
                    x = ttnn.tanh(x)
            if collect:''', '''                else:
                    x = ttnn.tanh(x, sub_core_grids=self.crs) if self.crs is not None else ttnn.tanh(x)
            if collect:''')
open(f + ".tmp", "w").write(s)
print("patched")

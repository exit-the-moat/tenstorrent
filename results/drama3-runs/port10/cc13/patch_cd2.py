"""cc13: layout_full (estimate mode) + linear core_grid without a sub-device."""
import sys

f = sys.argv[1]
s = open(f).read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old, s.count(old))
    s = s.replace(old, new)


rep('''    def __init__(self, mesh, codec, length, cores=None, sub_device_id=None):
        """codec: the host codec (fold_weight_norm()'d); length: latent rows per call (window frames * 4).
        cores / sub_device_id (cc13, opt-in): the core rectangle (CoreRangeSet) of the codec sub-device and its id."""
        self.mesh, self.L0 = mesh, length
        self.crs = cores if cores is not None else core_rect(CORES)
        self.sd = sub_device_id
''', '''    def __init__(self, mesh, codec, length, cores=None, sub_device_id=None, layout_full=False):
        """codec: the host codec (fold_weight_norm()'d); length: latent rows per call (window frames * 4).
        cores / sub_device_id (cc13, opt-in): the core rectangle (CoreRangeSet) of the codec sub-device and its id
        (without an id the matmuls run on a grid of the rectangle's size at (0, 0): timing only).
        layout_full (cc13, timing estimate only): the tile <-> row-major conversions and the channel-0 tail on the full
        grid (sub_core_grids untilize takes one tile row only), so no sub-device can be loaded."""
        self.mesh, self.L0 = mesh, length
        self.crs = cores if cores is not None else core_rect(CORES)
        self.sd = sub_device_id
        self.layout_full = layout_full
''')
rep('''            if self.crs is not None:
                return ttnn.linear(x, c.w, bias=c.b, compute_kernel_config=self.ckc, dtype=ACT, sub_device_id=self.sd)
''', '''            if self.crs is not None and self.sd is not None:
                return ttnn.linear(x, c.w, bias=c.b, compute_kernel_config=self.ckc, dtype=ACT, sub_device_id=self.sd)
            if self.crs is not None:
                g = self.crs.bounding_box().grid_size()
                return ttnn.linear(x, c.w, bias=c.b, compute_kernel_config=self.ckc, dtype=ACT, core_grid=ttnn.CoreGrid(y=g.y, x=g.x))
''')
rep('''    def _to_layout(self, x, layout):
        if self.crs is not None:
''', '''    def _to_layout(self, x, layout):
        if self.crs is not None and not self.layout_full:
''')
rep('''                if OUT_CH0 and self.crs is not None:  # cc13''', '''                if OUT_CH0 and self.crs is not None and not self.layout_full:  # cc13''')
open(f + ".tmp", "w").write(s)
print("patched")

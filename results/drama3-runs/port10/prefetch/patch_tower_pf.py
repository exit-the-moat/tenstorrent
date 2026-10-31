"""port10: patch tt/tower_device.py for DRAMA3_TOWER_PREFETCH (writes <file>.new; the caller mv's it)."""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:60], s.count(old))
    s = s.replace(old, new)


rep(
    """        attn_fused_decode.install([L.attn for L in self.shared + self.residual], mesh)
""",
    """        attn_fused_decode.install([L.attn for L in self.shared + self.residual], mesh)
        # DRAMA3_TOWER_PREFETCH=1 (port10, opt-in): the one-row decode (fused frame trace, step()) streams its 6 x 13
        # matmul weights through the DRAM-core tensor prefetcher (tt/tower_prefetch.py); after the merged qkv weight
        # exists, before any trace capture. A no-op with DRAMA3_TOWER_DS and in batch mode (host/batch, host/cbatch).
        from models.demos.blackhole.drama3.tt import tower_prefetch

        self.pf = None
        if tower_prefetch.ENABLED and not any(L.ds for L in self.shared + self.residual) and not _batch_main():
            self.pf = tower_prefetch.TowerPrefetch(mesh, self.shared + self.residual)
        self._pf_capture = False

    def _pf_frame(self):
        import contextlib

        return self.pf.frame(capture=self._pf_capture) if self.pf is not None else contextlib.nullcontext()
""",
)
rep(
    """    def _fused_body(self):
        \"\"\"All 9 segments""",
    """    def _fused_body(self):
        with self._pf_frame():
            return self._fused_body_inner()

    def _fused_body_inner(self):
        \"\"\"All 9 segments""",
)
rep(
    """        self.fused_tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.fused_codes = self._fused_body()
""",
    """        self.fused_tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._pf_capture = True  # the prefetch request is captured too: every replay re-sends it
        try:
            self.fused_codes = self._fused_body()
        finally:
            self._pf_capture = False
""",
)
rep(
    """    def step(
        self, cand_logits, h_dev, pos, input_frame, window, gen, temperature=1.0, top_p=0.9, top_k=None, collect=False,
        force_codes=None,
    ):
        \"\"\"Same contract""",
    """    def step(self, *a, **k):
        with self._pf_frame():
            return self._step(*a, **k)

    def _step(
        self, cand_logits, h_dev, pos, input_frame, window, gen, temperature=1.0, top_p=0.9, top_k=None, collect=False,
        force_codes=None,
    ):
        \"\"\"Same contract""",
)
rep(
    """class TowerLayerDevice:""",
    """def _batch_main():
    \"\"\"True when the running program is host/batch.py or host/cbatch.py (python -m ...).\"\"\"
    import sys

    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return (getattr(spec, "name", "") or "").rsplit(".", 1)[-1] in ("batch", "cbatch")


class TowerLayerDevice:""",
)
open(p + ".new", "w").write(s)
print("patched")

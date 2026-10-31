import os
p = "/opt/ep0/tt-metal/models/demos/blackhole/drama3/tt/device_sampler.py"
s = open(p).read()
old = """            kd("sampler_fused_reader.cpp", ct, rt, ttnn.ReaderConfigDescriptor()),
            kd("sampler_fused_reader.cpp", ct, rt, ttnn.WriterConfigDescriptor(), [("SAMPLER_B", "1")]),"""
new = """            kd("sampler_fused_reader.cpp", ct, rt, ttnn.ReaderConfigDescriptor(), self.defs),
            kd("sampler_fused_reader.cpp", ct, rt, ttnn.WriterConfigDescriptor(), [("SAMPLER_B", "1")] + self.defs),"""
assert s.count(old) == 1; s = s.replace(old, new)
old2 = """        self.ccfg.unpack_to_dest_mode = modes
"""
new2 = """        self.ccfg.unpack_to_dest_mode = modes
        # DRAMA3_SAMPLER_FUSED_PROFILE=1 (checks): wall-clock stamps in row 0 of the code tile (the code still at (0, 0))
        self.defs = [("SAMPLER_PROFILE", "1")] if os.environ.get("DRAMA3_SAMPLER_FUSED_PROFILE", "0") == "1" else []
"""
assert s.count(old2) == 1; s = s.replace(old2, new2)
open(p + ".tmp", "w").write(s); os.replace(p + ".tmp", p); print("ok")

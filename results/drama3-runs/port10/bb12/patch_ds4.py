import os
p = "/opt/ep0/tt-metal/models/demos/blackhole/drama3/tt/device_sampler.py"
s = open(p).read()
old = "        modes[0] = modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32\n"
new = "        for i in (0, 1, 6, 7, 8):\n            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32\n"
assert s.count(old) == 1; s = s.replace(old, new)
old = "cb(4, bf, 2048, 2), cb(5, f32, 4096, 1)]"
new = "cb(4, bf, 2048, 2), cb(5, f32, 4096, 1)] + [cb(i, f32, 4096, 1) for i in range(6, 11)]"
assert s.count(old) == 1; s = s.replace(old, new)
s = s.replace("""over all logits on the SFPU (the ops ttnn.subtract / ttnn.exp run), z = sum(E) as an exact fixed-point sum rounded
    once, then p / cum / keep / e / noise / argmax in fp32 in rank order.""", """over all logits on the SFPU (the ops ttnn.subtract / ttnn.exp run), z = sum(E) as a fixed-point sum rounded once,
    p = e / z and e / noise on the SFPU (div_binary_tile, as ttnn.divide), the cum as ttnn.cumsum's Kahan recurrence,
    then keep / argmax in rank order.""")
open(p + ".tmp", "w").write(s); os.replace(p + ".tmp", p); print("ok")

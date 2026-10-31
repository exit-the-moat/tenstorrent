import time, torch
from models.demos.blackhole.drama3.host import codec as C
codes = torch.load("/opt/ep0/day3/out/fix.frames.pt")[:, 1:].T.contiguous()[None]
c = C.load_codec(); c.fold_weight_norm(); torch.set_num_threads(8)
with torch.inference_mode():
    ref = c.from_indices(codes)
    z = c.quantizer.decode(codes)
    dec = torch.compile(c.decoder, dynamic=True)
    t0=time.time(); out = dec(z); print("compile+first", round(time.time()-t0,1), flush=True)
    for n in (254, 96, 42):
        zz = c.quantizer.decode(codes[:, :, :n])
        dec(zz)
        t0=time.time(); o = dec(zz); print(n, "frames:", round(time.time()-t0,3), "s", flush=True)
    print("max diff vs eager", float((out-ref).abs().max()))

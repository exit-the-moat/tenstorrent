import time, torch
from models.demos.blackhole.drama3.host import codec as C
codes = torch.load("/opt/ep0/day3/out/fix.frames.pt")[:, 1:].T.contiguous()[None]
c = C.load_codec(); c.fold_weight_norm(); torch.set_num_threads(8)
with torch.inference_mode():
    for _ in range(2):
        t0=time.time(); z=c.quantizer.decode(codes); t1=time.time()
        dec=c.decoder; x=z
        times=[]
        for i,m in enumerate(dec.model):
            s=time.time(); x=m(x); times.append((type(m).__name__, round(time.time()-s,3), tuple(x.shape)))
        print("quantizer.decode", round(t1-t0,3), tuple(z.shape)); print(times)

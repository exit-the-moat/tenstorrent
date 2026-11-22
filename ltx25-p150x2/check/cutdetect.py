# ltx1: find hard cuts in clips: frame-to-frame mean abs diff (quarter res) > 4x the clip median
import av, numpy as np, sys, glob, json
res = {}
for f in sorted(sum([glob.glob(g) for g in sys.argv[1:]], [])):
    c = av.open(f); fr = [x.to_ndarray(format="rgb24")[::8, ::8].astype(np.float32) for x in c.decode(video=0)]
    a = np.stack(fr); d = np.abs(a[1:] - a[:-1]).mean(axis=(1, 2, 3)); med = float(np.median(d))
    cuts = [int(i + 1) for i in np.where(d > max(4 * med, 8.0))[0]]
    res[f] = {"frames": len(a), "median_diff": round(med, 2), "max_diff": round(float(d.max()), 2), "cuts_at_frame": cuts}
    print(f.split("/")[-1], res[f], flush=True)

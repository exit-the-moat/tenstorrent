import av, numpy as np, sys
c = av.open(sys.argv[1]); fr = [f.to_ndarray(format="rgb24")[::4, ::4].astype(np.float32) for f in c.decode(video=0)]
a = np.stack(fr); d = np.abs(a[1:] - a[:-1]).mean(axis=(1, 2, 3))
joints = [int(x) for x in sys.argv[2].split(",")]
print("frames", len(a), "consecutive mean abs diff (0-255): median %.2f p95 %.2f max %.2f at %d" % (np.median(d), np.percentile(d, 95), d.max(), d.argmax()))
for k in joints:
    print("joint %d->%d: %.2f | neighbours" % (k - 1, k, d[k - 1]), ["%.2f" % x for x in d[k - 5:k + 4]])

import sys, torch, glob, os
a, b = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(a + '/*.frames.pt')):
    x = torch.load(f); y = torch.load(os.path.join(b, os.path.basename(f)))
    n = min(len(x), len(y)); d = (x[:n] != y[:n]).any(-1).nonzero()
    i = int(d[0]) if len(d) else None
    print(os.path.basename(f), len(x), len(y), 'first diff frame', i, '' if i is None else (x[i].tolist(), y[i].tolist()))

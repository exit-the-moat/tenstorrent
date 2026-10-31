import sys, glob, os, torch
a, b = sys.argv[1], sys.argv[2]
n = same = 0
for f in sorted(glob.glob(a + "/*.frames.pt")):
    g = os.path.join(b, os.path.basename(f))
    if not os.path.exists(g):
        continue
    n += 1
    same += torch.equal(torch.load(f), torch.load(g))
print(f"frames identical {same}/{n}")

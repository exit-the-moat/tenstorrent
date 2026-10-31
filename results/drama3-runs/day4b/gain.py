import sys, numpy as np, soundfile as sf
a, _ = sf.read(sys.argv[1]); b, _ = sf.read(sys.argv[2]); n = min(len(a), len(b)); a, b = a[:n], b[:n]
g = (a * b).sum() / (b * b).sum()
e = a - g * b
print("gain", round(g, 4), "snr_after_gain_db", round(10 * np.log10((a**2).sum() / (e**2).sum()), 2))
# per 4096-sample block gain spread
blk = 8192
gs = [(a[i:i+blk]*b[i:i+blk]).sum()/max((b[i:i+blk]**2).sum(),1e-9) for i in range(0, n - blk, blk) if (a[i:i+blk]**2).sum() > 1e-3]
print("block gains min/median/max", round(min(gs),3), round(float(np.median(gs)),3), round(max(gs),3))

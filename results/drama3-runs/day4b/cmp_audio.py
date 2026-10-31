import sys, torch, soundfile as sf, numpy as np
a_dir, b_dir = sys.argv[1], sys.argv[2]
for l in ["close", "fix"]:
    fa = torch.load(f"{a_dir}/{l}.frames.pt"); fb = torch.load(f"{b_dir}/{l}.frames.pt")
    a, _ = sf.read(f"{a_dir}/{l}.wav"); b, _ = sf.read(f"{b_dir}/{l}.wav")
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    e = a - b
    snr = 10 * np.log10((a ** 2).sum() / (e ** 2).sum())
    pcc = np.corrcoef(a, b)[0, 1]
    rms = lambda x: 20 * np.log10(np.sqrt((x ** 2).mean()))
    print(l, "frames_equal", torch.equal(fa, fb), "len", len(a), "snr_db", round(snr, 2), "pcc", round(pcc, 5), "rms_a", round(rms(a), 2), "rms_b", round(rms(b), 2), "max_abs_diff", round(float(np.abs(e).max()), 4))

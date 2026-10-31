from pathlib import Path
p = Path("/opt/ep0/tt-metal/models/demos/blackhole/drama3/host/generate_fast.py")
s = p.read_text()
a = '''        s.split_point([s.prompt(x, [(ref_text, codes)]) for x in ("A", "Z")] + list(prompts.values()))
        s.warmup(prompts[ids[0]])'''
b = '''        s.split_point([s.prompt(x, [(ref_text, codes)]) for x in ("A", "Z")] + list(prompts.values()))
        # DRAMA3_ONLY_BUCKET=<rows> (v20 episode render): keep only the lines whose suffix prefill bucket is <rows>, so a
        # lines file with mixed lengths runs as one pass per bucket (a parked trace can't compile a second bucket)
        bk = {lid: s.bb.bucket_for(r.shape[0] - s.P) for lid, r in prompts.items()}
        log(f"suffix prefill buckets {dict(sorted(__import__('collections').Counter(bk.values()).items()))}")
        if os.environ.get("DRAMA3_ONLY_BUCKET"):
            ids = [lid for lid in ids if bk[lid] == int(os.environ["DRAMA3_ONLY_BUCKET"])]
            if not ids:
                log("no lines in this bucket")
                return
            s._warm_T = max(prompts[lid].shape[0] for lid in ids)
        s.warmup(prompts[ids[0]])'''
assert s.count(a) == 1
p.write_text(s.replace(a, b))
print("patched")

#!/usr/bin/env python3
"""Verify a local HF snapshot against the repo's tree API: sha256 for LFS files, git blob sha1 for the rest.
Usage: verify_hf.py <repo_id> <dir>. Prints DONE or FAILED at the end."""
import hashlib, json, os, sys, urllib.request

repo, d = sys.argv[1], sys.argv[2]
tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"))
bad = missing = 0
files = [t for t in tree if t["type"] == "file"]
for i, t in enumerate(files, 1):
    p = os.path.join(d, t["path"])
    if not os.path.exists(p):
        missing += 1
        print(f"[{i}/{len(files)}] MISSING {t['path']}", flush=True)
        continue
    if "lfs" in t:
        h = hashlib.sha256()
        want = t["lfs"]["oid"]
    else:
        h = hashlib.sha1(b"blob %d\0" % os.path.getsize(p))
        want = t["oid"]
    with open(p, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    ok = h.hexdigest() == want
    bad += not ok
    print(f"[{i}/{len(files)}] {'ok ' if ok else 'BAD'} {t['path']}", flush=True)
extra = sorted(set(os.listdir(d)) - {t["path"].split("/")[0] for t in files})
print("local files not in the repo:", extra)
print("DONE all ok" if bad == missing == 0 else f"FAILED bad={bad} missing={missing}", flush=True)

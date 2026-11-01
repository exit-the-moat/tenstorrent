#!/usr/bin/env python3
"""Check every file listed in <dir>/crc32.txt ("<crc32hex>  <name>"). Prints DONE or FAILED at the end."""
import os, sys, zlib

d = sys.argv[1]
bad = 0
lines = [l.split(None, 1) for l in open(os.path.join(d, "crc32.txt")) if l.strip()]
for i, (want, name) in enumerate(lines, 1):
    name = name.strip()
    crc = 0
    with open(os.path.join(d, name), "rb") as f:
        while chunk := f.read(16 << 20):
            crc = zlib.crc32(chunk, crc)
    got = f"{crc:08x}"
    ok = got == want.lower()
    bad += not ok
    print(f"[{i}/{len(lines)}] {'ok ' if ok else 'BAD'} {name} {got} (want {want})", flush=True)
print("DONE all ok" if bad == 0 else f"FAILED {bad} bad files", flush=True)

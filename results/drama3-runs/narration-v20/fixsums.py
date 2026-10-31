import glob
OLD = "(cd $out && sha256sum *.wav > SHA256SUMS && sync && echo 3 > /proc/sys/vm/drop_caches && sha256sum -c --quiet SHA256SUMS && echo SUMS_OK)"
NEW = '/opt/ep0/verify_new.sh $out "*.wav"  # evicts only the new wavs; drop_caches evicted the model weights too (2026-10-31)'
for f in sorted(glob.glob("/opt/ep0/v20/render_v20*.sh")):
    s = open(f).read()
    assert s.count(OLD) == 1, f
    open(f, "w").write(s.replace(OLD, NEW))
    print("ok", f)

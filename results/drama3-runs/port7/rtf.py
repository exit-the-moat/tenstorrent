import json, sys
for d in sys.argv[1:]:
    t = json.load(open(d + "/timing.json"))
    au = sum(v["audio_s"] for v in t.values()); cs = sum(v["compute_s"] for v in t.values()); gs = sum(v["gen_s"] for v in t.values()); fr = sum(v["frames"] for v in t.values())
    tt = [v["ttfa_s"] for v in t.values()]
    print(d.split("/")[-1], "lines", len(t), "RTF %.3f" % (cs / au), "gen ms/frame %.2f" % (gs / fr * 1e3), "rtf/line %.3f-%.3f" % (min(v["rtf"] for v in t.values()), max(v["rtf"] for v in t.values())),
          "TTFA %.3f-%.3f" % (min(tt), max(tt)), "worst_late %.3f" % max(v.get("worst_late_s", 0) for v in t.values()), "frames", fr, "audio %.1f" % au)

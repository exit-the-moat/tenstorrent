import json, torch, sys
def agg(d):
    t=json.load(open(d+"/timing.json")); g=sum(v["gen_s"] for v in t.values()); a=sum(v["audio_s"] for v in t.values()); f=sum(v["frames"] for v in t.values())
    return round(g/a,4), f, round(1e3*g/f,2)
ref=sys.argv[1]; base=sys.argv[2]
for d in sys.argv[3:]:
    t=json.load(open(d+"/timing.json"))
    same=[torch.equal(torch.load(f"{d}/{k}.frames.pt"), torch.load(f"{ref}/{k}.frames.pt")) for k in t]
    same_b=[torch.equal(torch.load(f"{d}/{k}.frames.pt"), torch.load(f"{base}/{k}.frames.pt")) for k in t]
    print(d, "rtf,frames,ms/frame", agg(d), "frames==ref", sum(same), "/8", "==base", sum(same_b), "/8", flush=True)
print(ref, agg(ref))

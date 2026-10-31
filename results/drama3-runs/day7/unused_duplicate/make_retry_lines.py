# Day 7: lines file for the narration retries (id = <line id>_<variant>; text = spoken; written = the episode form)
import json
import re
src = {l["id"]: l for l in json.load(open("/opt/ep0/voice/card_lines.json"))}
out = []
for lid in ("s04-05", "s05-02", "s02-01", "s02-02", "s10-02"):
    l = src[lid]
    out.append(dict(id=f"{lid}_orig", line=lid, variant="orig", written=l["written"], spoken=l["spoken"]))
    if "moat" in l["spoken"].lower():
        sp = re.sub("moat", lambda m: "Mote" if m.group(0)[0] == "M" else "mote", l["spoken"], flags=re.I).replace("cloud bills", "cloud bill's")
        out.append(dict(id=f"{lid}_mote", line=lid, variant="mote", written=l["written"], spoken=sp))
json.dump(out, open("/opt/ep0/day7/retry_lines.json", "w"), indent=1)
for x in out:
    print(x["id"], "|", x["spoken"])

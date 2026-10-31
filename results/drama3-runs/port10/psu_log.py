"""port10: whole-system power from the Corsair HX1500i over USB (hwmon corsairpsu: power1 = total DC output,
the sum of the 12 / 5 / 3.3 V rails). psu_log.py <out.jsonl> <stop file>: one sample per second."""
import glob, json, os, sys, time
h = next(p for p in glob.glob("/sys/class/hwmon/hwmon*") if open(p + "/name").read().strip() == "corsairpsu")
out, stop = sys.argv[1], sys.argv[2]
with open(out, "a") as f:
    while not os.path.exists(stop):
        try:
            f.write(json.dumps(dict(t=time.time(), total_w=int(open(h + "/power1_input").read()) / 1e6,
                                    w12=int(open(h + "/power2_input").read()) / 1e6)) + "\n")
            f.flush()
        except Exception as e:
            f.write(json.dumps(dict(t=time.time(), error=str(e))) + "\n")
        time.sleep(1)

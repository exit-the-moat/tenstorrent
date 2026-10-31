"""port10: whole-system power while one card works. Every ~2 s: PSU total DC output (Corsair HX1500i, hwmon
corsairpsu power1) and both cards board + chip power (tt-smi -s). pw2.py log <out.jsonl> <stop>;
pw2.py sum <out.jsonl> <busy card> [t0 t1]: means, and system minus the idle card = "system with one card"."""
import glob, json, os, subprocess, sys, time
TT = "/opt/tenstorrent/venv/bin/tt-smi"
def psu():
    h = next(p for p in glob.glob("/sys/class/hwmon/hwmon*") if open(p + "/name").read().strip() == "corsairpsu")
    return int(open(h + "/power1_input").read()) / 1e6
def log(out, stop):
    with open(out, "a") as f:
        while not os.path.exists(stop):
            try:
                d = json.loads(subprocess.run([TT, "-s"], capture_output=True, text=True, timeout=30).stdout)
                r = dict(t=time.time(), psu_w=psu())
                for i, x in enumerate(d["device_info"]):
                    r[f"board{i}"] = float(x["telemetry"]["board_power"]); r[f"chip{i}"] = float(x["telemetry"]["power"])
                f.write(json.dumps(r) + "\n"); f.flush()
            except Exception as e:
                f.write(json.dumps(dict(t=time.time(), error=str(e))) + "\n")
            time.sleep(1)
def summ(path, busy, t0=None, t1=None):
    s = [json.loads(l) for l in open(path) if "psu_w" in l]
    if t0: s = [x for x in s if float(t0) <= x["t"] <= float(t1)]
    idle = 1 - int(busy)
    m = lambda k: round(sum(x[k] for x in s) / len(s), 1)
    out = dict(n=len(s), psu_total_w=m("psu_w"), busy_board_w=m(f"board{busy}"), busy_chip_w=m(f"chip{busy}"),
               idle_board_w=m(f"board{idle}"))
    out["system_one_card_w"] = round(out["psu_total_w"] - out["idle_board_w"], 1)
    print(json.dumps(out))
if __name__ == "__main__":
    {"log": log, "sum": summ}[sys.argv[1]](*sys.argv[2:])

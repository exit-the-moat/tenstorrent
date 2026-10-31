p = "/opt/ep0/tt-metal/models/demos/blackhole/drama3/tt/decode_linear.py"
s = open(p, "rb").read().decode()
nl = "\r\n" if "\r\n" in s else "\n"
o = "                and (args[0].shape[-2] == 1 or (args[0].shape[-2] <= ROWS and math.prod(args[0].shape[:-2]) == 1))" + nl
n = nl.join(
    [
        "                and (",
        "                    args[0].shape[-2] == 1",
        "                    if ROWS == 1",
        "                    else args[0].shape[-2] <= ROWS and math.prod(args[0].shape[:-2]) == 1",
        "                )",
        "",
    ]
)
assert s.count(o) == 1
open(p, "wb").write(s.replace(o, n).encode())
print("FIXED")

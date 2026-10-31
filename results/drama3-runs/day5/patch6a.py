import os

os.chdir("/opt/ep0/tt-metal/models/demos/blackhole/drama3")


def edit(p, pairs):
    raw = open(p, "rb").read()
    crlf = b"\r\n" in raw
    s = raw.decode().replace("\r\n", "\n")
    for o, n in pairs:
        assert s.count(o) == 1, (p, o)
        s = s.replace(o, n)
    if crlf:
        s = s.replace("\n", "\r\n")
    open(p, "wb").write(s.encode())


edit(
    "host/generate_fast.py",
    [
        (
            '    ap.add_argument("--lines", default="close")\n',
            '    ap.add_argument("--lines", default="close")\n'
            '    ap.add_argument("--lines-file", default="", help="JSON list of {id, spoken, written} (Day 6 card_lines.json): text = spoken")\n',
        ),
        (
            '    lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}\n',
            '    if a.lines_file:\n'
            '        lines = {l["id"]: dict(l, text=l["spoken"]) for l in json.load(open(a.lines_file))}\n'
            '    else:\n'
            '        lines = {l["id"]: l for l in yaml.safe_load(open(VOICE / "lines.yaml"))["lines"]}\n',
        ),
    ],
)
edit(
    "tests/summarize_day4.py",
    [
        (
            'LINES = yaml.safe_load(open("/opt/ep0/voice/lines.yaml"))["lines"]\n',
            '# DRAMA3_LINES_FILE: a JSON list of {id, spoken, written} (Day 6 card_lines.json) instead of lines.yaml\n'
            'if os.environ.get("DRAMA3_LINES_FILE"):\n'
            '    LINES = [dict(l, text=l["spoken"]) for l in json.load(open(os.environ["DRAMA3_LINES_FILE"]))]\n'
            'else:\n'
            '    LINES = yaml.safe_load(open("/opt/ep0/voice/lines.yaml"))["lines"]\n',
        ),
        ("import json\nimport re\n", "import json\nimport os\nimport re\n"),
    ],
)
print("PATCHED")

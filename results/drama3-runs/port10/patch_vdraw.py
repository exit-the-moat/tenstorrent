p = "host/batch.py"
s = open(p).read()
anchor = "def _pre(tower, frame, gen):"
add = '''def _sample_rows(L, temperature, top_k, top_p, N, m=64):
    """port10: tt/tower_device.sample_with_noise (keep_crossing=True) for n rows at once: L [n, C] logits, N [n, C]
    noise -> [n] chosen indices. Rows whose kept ranks may reach past the top m fall back to the one-row function."""
    from models.demos.blackhole.drama3.tt.tower_device import MIN_TEMPERATURE, sample_with_noise

    scaled = L.float() / max(temperature, MIN_TEMPERATURE)
    vals, ids = torch.topk(scaled, m, dim=-1)
    cum = torch.cumsum(torch.exp(vals - torch.logsumexp(scaled, -1, keepdim=True)), -1)
    drop = cum > top_p
    drop[:, 1:] = drop[:, :-1].clone()
    drop |= torch.arange(m) >= top_k
    drop[:, 0] = False
    probs = torch.softmax(vals.masked_fill(drop, float("-inf")), -1)
    out = ids.gather(1, torch.argmax(probs / N[:, :m], -1, keepdim=True))[:, 0]
    if top_k > m:
        for r in torch.nonzero(~(cum[:, -2] > top_p)).flatten().tolist():
            out[r] = sample_with_noise(L[r], temperature, top_k, top_p, N[r], keep_crossing=True, m=m)
    return out


def _draw_many(tower, logits, windows, pres, temperature, top_p, top_k):
    """port10 DRAMA3_CB_VDRAW=1: _draw for n streams at once (the same per-row math and noise) -> (mains, code0s,
    redrawn flags, host rows [5, n, H])."""
    from models.demos.blackhole.drama3.host.tower import RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P

    t = tower.host
    top_k = top_k or len(t.candidates)
    L = torch.stack(logits)
    first = t.candidates[_sample_rows(L, temperature, top_k, top_p, torch.stack([p["n1"] for p in pres]))]
    ras = t.candidates[_sample_rows(L, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, torch.stack([p["n2"] for p in pres]))]
    first_l, ras_l = first.tolist(), ras.tolist()
    mains = [r if (f in w and t.sem0 <= f < t.sem0 + t.cb_size) else f for f, r, w in zip(first_l, ras_l, windows)]
    m = torch.tensor(mains)
    c0 = (m - t.sem0).clamp(0, t.cb_size - 1)
    lab = t.is_semantic(m)
    si = torch.cat([p["si"] for p in pres])
    rows = [tower._emb_rows(layer, c0, lab, si if i == 0 else None) for i, layer in enumerate(tower.shared)]
    rows.append(tower._emb_rows(tower.residual[0], c0, lab))
    return mains, c0.tolist(), [f != mm for f, mm in zip(first_l, mains)], torch.stack(rows)  # [5, n, H]


'''
assert anchor in s
s = s.replace(anchor, add + anchor, 1)
open(p, "w").write(s)

p = "host/cbatch.py"
s = open(p).read()
old = """        t1 = time.time()
        for b in act:
            x = slot[b]
            main, code0, redrawn, rows5 = _draw(tw, x["logits"], x["window"], x["pre"], temperature, top_p, top_k)
            mains[b], code0s[b] = main, code0
            x["ras"] += int(redrawn)
            host_rows[:, b] = rows5
            noise[0, :, b] = x["pre"]["heads"][:, : st.K]
"""
new = """        t1 = time.time()
        if vdraw:  # port10: every active stream's draw in one batched pass (same math and noise per row)
            xs = [slot[b] for b in act]
            ms_, cs_, rd_, rows_ = _draw_many(tw, [x["logits"] for x in xs], [x["window"] for x in xs], [x["pre"] for x in xs],
                                              temperature, top_p, top_k)
            for i, b in enumerate(act):
                mains[b], code0s[b] = ms_[i], cs_[i]
                xs[i]["ras"] += int(rd_[i])
                noise[0, :, b] = xs[i]["pre"]["heads"][:, : st.K]
            host_rows[:, act] = rows_
        for b in act if not vdraw else ():
            x = slot[b]
            main, code0, redrawn, rows5 = _draw(tw, x["logits"], x["window"], x["pre"], temperature, top_p, top_k)
            mains[b], code0s[b] = main, code0
            x["ras"] += int(redrawn)
            host_rows[:, b] = rows5
            noise[0, :, b] = x["pre"]["heads"][:, : st.K]
"""
assert old in s
s = s.replace(old, new, 1)
old = "    from models.demos.blackhole.drama3.host.batch import _draw, _pre\n"
new = "    from models.demos.blackhole.drama3.host.batch import _draw, _draw_many, _pre\n"
assert old in s
s = s.replace(old, new, 1)
old = "    t_draw = t_codec = 0.0  # port10: parts of t_host (sampling draws, codec windows on the card)\n"
new = old + "    vdraw = os.environ.get(\"DRAMA3_CB_VDRAW\", \"0\") == \"1\"\n"
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("ok")

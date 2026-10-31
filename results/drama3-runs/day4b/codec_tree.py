import torch
from models.demos.blackhole.drama3.host.codec import load_codec
c = load_codec(); c.fold_weight_norm()
def show(m, name, depth=0, maxd=4):
    n = sum(p.numel() for p in m.parameters())
    extra = ""
    if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        extra = f" k={m.kernel_size} s={m.stride} d={m.dilation} {m.in_channels}->{m.out_channels} groups={m.groups} pad={m.padding}"
    print("  " * depth + f"{name}: {type(m).__name__} {n/1e6:.2f}M{extra}")
    if depth < maxd:
        for k, ch in m.named_children():
            show(ch, k, depth + 1, maxd)
show(c.quantizer, "quantizer", 0, 3)
show(c.decoder, "decoder", 0, 5)

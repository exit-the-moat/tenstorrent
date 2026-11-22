# ltx1 I2V: preprocess a still exactly like ltx_pipelines (H.264 re-compress at the model's CRF, resize + centre crop,
# normalise to [-1, 1]) for the stage-1 and stage-2 sizes; tt's generate() loads <dir>/<H>x<W>.pt.
import sys, os, torch
from ltx_pipelines.utils.media_io.decode import load_conditioning_still
path, H, W, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
crf = int(sys.argv[5]) if len(sys.argv) > 5 else 18  # LTX_2_4_PARAMS.default_image_crf (applies to 2.5)
os.makedirs(out, exist_ok=True)
st = load_conditioning_still(path, crf)
for h, w in ((H // 2, W // 2), (H, W)):
    t = st.tensor(h, w, torch.float32, torch.device("cpu"))
    torch.save(t.cpu(), f"{out}/{h}x{w}.pt")
    print(out, h, w, tuple(t.shape), float(t.min()), float(t.max()))

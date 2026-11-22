# ltx1: I2V final comparison (cards vs CPU ref i0) + frame-0 fidelity to the input still
import sys, types, torch
for n in ("models", "models.tt_dit", "models.tt_dit.utils", "models.tt_dit.utils.video"):
    sys.modules[n] = types.ModuleType(n)
class Audio: pass
sys.modules["models.tt_dit.utils.video"].Audio = Audio
from ltx_pipelines.utils.blocks.audio_decoder import AudioDecoder
torch.set_num_threads(16)
REF = "/work/ref/i0"
def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten(); a, b = a - a.mean(), b - b.mean(); return float(a @ b / (a.norm() * b.norm()))
def psnr(a, b): return float(-10 * torch.log10(((a - b) ** 2).mean()))
ro = torch.load(f"{REF}/video_decoder_out0.pt", weights_only=False)
r = ro if isinstance(ro, torch.Tensor) else ro[0]
r = r.float()  # F,H,W,3 [0,1] expected
print("ref decoded", tuple(r.shape), float(r.min()), float(r.max()))
tt = torch.load(sys.argv[1], weights_only=False)
v = (tt["video"][0].float().clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 3, 0)
img = torch.load(sys.argv[2]).float()[0, :, 0].permute(1, 2, 0) * 0.5 + 0.5  # input still (preprocessed, H,W,3)
print("final video: PCC %.5f PSNR %.2f dB; per-frame PSNR" % (pcc(v, r), psnr(v, r)), [round(psnr(v[i], r[i]), 1) for i in range(0, v.shape[0], 4)])
print("frame 0 vs input still: cards %.2f dB, CPU reference %.2f dB" % (psnr(v[0], img), psnr(r[0], img)))
print("frame 24 vs input still: cards %.2f dB, CPU ref %.2f dB" % (psnr(v[-1], img), psnr(r[-1], img)))
s2 = torch.load(f"{REF}/s2_state_out.pt", weights_only=False); al = s2["audio"]["latent"].float()
with torch.inference_mode():
    au = AudioDecoder("/models/ltx25/vae/ltx-2.5-audio-vae-bf16.safetensors".replace("/models", "/models"), torch.float32, torch.device("cpu"))(al.reshape(1, al.shape[1], 8, 16).permute(0, 2, 1, 3))
wr = au.waveform.float().reshape(2, -1); wt = torch.as_tensor(tt["audio"].waveform).float().reshape(2, -1); n = min(wr.shape[1], wt.shape[1])
print("audio PCC %.4f" % pcc(wt[:, :n], wr[:, :n]))
import numpy as np, PIL.Image as I
I.fromarray((torch.cat([img, r[0], v[0]], 1).numpy() * 255).clip(0, 255).astype(np.uint8)).save("/out/i2v_frame0_input_ref_cards.png")
I.fromarray((torch.cat([r[-1], v[-1]], 1).numpy() * 255).clip(0, 255).astype(np.uint8)).save("/out/i2v_lastframe_ref_cards.png")

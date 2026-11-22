# ltx1: finish the r0 CPU reference: decode the final latents (fp32 conv VAE, fp32 audio VAE+vocoder) -> frames + wav.
import torch, numpy as np
from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder
from ltx_pipelines.utils.blocks.audio_decoder import AudioDecoder
torch.set_num_threads(16)
import os
R = os.environ.get("LTX_WEIGHTS", "/models/ltx25"); REF = os.environ.get("REF", "/work/ref/r0")
lat = torch.load(f"{REF}/video_decoder_in.pt", weights_only=False)["args"][0].float()
s2 = torch.load(f"{REF}/s2_state_out.pt", weights_only=False)
alat = s2["audio"]["latent"].float()  # (1, 26, 128) patchified
with torch.inference_mode():
    vd = VideoDecoder(f"{R}/vae/ltx-2.5-video-vae-conv-bf16.safetensors", torch.float32, torch.device("cpu"))
    frames = torch.cat([c for c in vd(lat, None, torch.Generator().manual_seed(0))], dim=0)  # (F,H,W,3) [0,1]
    ad = AudioDecoder(f"{R}/vae/ltx-2.5-audio-vae-bf16.safetensors", torch.float32, torch.device("cpu"))
    au = ad(alat.reshape(1, alat.shape[1], 8, 16).permute(0, 2, 1, 3))
torch.save({"frames": frames, "wav": au.waveform if hasattr(au, "waveform") else au[0]}, f"{REF}/final_fp32decode.pt")
print("REFDECODE_DONE", tuple(frames.shape))

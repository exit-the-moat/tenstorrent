# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fish Audio's codec (``codec.pth``, the "modded DAC", 44.1 kHz, 10 codebooks per frame) on the host CPU.

The model code is Fish's own, loaded by file path from the SGLang-Omni clone (``fishaudio_s2_pro/fish_speech/
models/dac/{modded_dac,rvq}.py``) and built by hand from its ``modded_dac_vq.yaml`` (no hydra / omegaconf).
Those files import a few layers from Descript's ``dac`` package, installed ``--no-deps`` in /opt/ep0/pyext-codec.
``dac/__init__`` pulls in argbind, encodec and all of audiotools, so the needed submodules are loaded directly
behind namespace stubs, with a two-class ``audiotools`` stub (AudioSignal / ml.BaseModel are only base classes).
"""

import functools
import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import torch

FISH_DAC = Path(
    os.environ.get(
        "DRAMA3_FISH_DAC", "/opt/ep0/ref/sglang-omni/sglang_omni/models/fishaudio_s2_pro/fish_speech/models/dac"
    )
)
DAC_PKG = Path(os.environ.get("DRAMA3_DAC_PKG", "/opt/ep0/pyext-codec/dac"))
CODEC = os.environ.get("DRAMA3_CODEC", "/models/drama-3/codec.pth")


def _install_import_shims():
    if "dac.nn.layers" in sys.modules:
        return
    at = types.ModuleType("audiotools")
    at.AudioSignal = type("AudioSignal", (), {})
    at.STFTParams = type("STFTParams", (), {})
    at.ml = types.ModuleType("audiotools.ml")
    at.ml.BaseModel = torch.nn.Module
    sys.modules.setdefault("audiotools", at)
    sys.modules.setdefault("audiotools.ml", at.ml)
    for name, sub in (("dac", ""), ("dac.nn", "nn"), ("dac.model", "model")):
        m = types.ModuleType(name)
        m.__path__ = [str(DAC_PKG / sub)]
        sys.modules[name] = m
    import dac.model.base  # noqa: F401  (CodecMixin)
    import dac.nn.layers  # noqa: F401
    import dac.nn.quantize  # noqa: F401


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=1)
def load_codec(path=CODEC):
    """The codec as Fish builds it from modded_dac_vq.yaml, weights from codec.pth, eval mode on CPU."""
    _install_import_shims()
    md = _load_file("drama3_fish_modded_dac", FISH_DAC / "modded_dac.py")
    rvq = _load_file("drama3_fish_rvq", FISH_DAC / "rvq.py")
    common = dict(
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )

    def window_transformer():  # the yaml's *transformer_module (pre and post module)
        cfg = md.ModelArgs(block_size=2048, n_layer=8, n_head=16, dim=1024, intermediate_size=3072, **common)
        return md.WindowLimitedTransformer(causal=True, window_size=128, input_dim=1024, config=cfg)

    quantizer = rvq.DownsampleResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.5,
        downsample_factor=[2, 2],
        post_module=window_transformer(),
        pre_module=window_transformer(),
        semantic_codebook_size=4096,
    )
    codec = md.DAC(
        sample_rate=44100,
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        decoder_dim=1536,
        decoder_rates=[8, 8, 4, 2],
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=functools.partial(md.ModelArgs, block_size=8192, **common),
        quantizer=quantizer,
    )
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    res = codec.load_state_dict(state, strict=False, assign=True)  # Fish loads it the same way
    codec.load_report = dict(missing=list(res.missing_keys), unexpected=list(res.unexpected_keys))
    return codec.eval()


def load_wav(path, sample_rate):
    """Mono float waveform [1, samples] at sample_rate (torchaudio resample, as Fish's reference encoder)."""
    import soundfile as sf
    import torchaudio

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.T).mean(0, keepdim=True)
    return torchaudio.functional.resample(audio, sr, sample_rate)


@torch.no_grad()
def encode_reference(path):
    """Codes [10, frames] (int64) of a reference clip: Fish's encode_reference_waveform on the CPU (fp32)."""
    codec = load_codec()
    audio = load_wav(path, codec.sample_rate)
    codes, _ = codec.encode(audio, torch.tensor([audio.shape[1]], dtype=torch.long))
    return codes[0].long()


@torch.no_grad()
def decode_codes(codes):
    """[10, frames] codes -> float waveform [samples] at codec.sample_rate."""
    codec = load_codec()
    return codec.from_indices(codes[None].long())[0, 0]


def frames_to_seconds(frames):
    codec = load_codec()
    return frames * codec.frame_length / codec.sample_rate


def save_wav(path, audio, sample_rate):
    import soundfile as sf

    sf.write(str(path), audio.clamp(-1, 1).numpy(), sample_rate, subtype="PCM_16")


if __name__ == "__main__":
    # round trip: python -m models.demos.blackhole.drama3.host.codec <in.wav> <out.wav>
    codec = load_codec()
    print("codec", codec.sample_rate, "Hz, frame_length", codec.frame_length, codec.load_report)
    codes = encode_reference(sys.argv[1])
    print("codes", tuple(codes.shape), "min/max per codebook", codes.min(1).values.tolist(), codes.max(1).values.tolist())
    audio = decode_codes(codes)
    print("decoded", audio.shape[0], "samples =", round(audio.shape[0] / codec.sample_rate, 2), "s; peak", float(audio.abs().max()))
    save_wav(sys.argv[2], audio, codec.sample_rate)
    print("DONE", math.isfinite(float(audio.abs().max())))

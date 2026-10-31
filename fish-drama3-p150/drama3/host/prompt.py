# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drama 3 prompt rows ("reference-bracket-v1") and the backbone input embeddings, on the host.

A port of SGLang-Omni's ``fishaudio_drama3/prompt.py::build_prompt_rows`` and
``AudioDecoder.embed_backbone`` (their validation of caller text is kept in short form).
A row is ``[token id, code 0, ..., code 9]``; text rows carry zero codes, a reference audio frame is
``[semantic_start + code0, code0, ..., code9]``.
"""

import math
import re

import torch

TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = "convert the provided text to speech reference to the following:"
MESSAGE_END = "<|im_end|>\n"
GENERATION_START = "<|im_start|>assistant\n<|voice|>"
STRUCTURAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:")
SPEAKER_TAG = re.compile(r"<\|speaker:([0-9]+)\|>")
# the Unicode White_Space property (Fish strips with exactly this set)
WHITE_SPACE = "".join(
    map(chr, (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000))
)
NUM_CODEBOOKS = 10
CODEBOOK_SIZE = 4096


def _check_text(text, where):
    if not text.strip(WHITE_SPACE):
        raise ValueError(f"{where}: empty text")
    for t in STRUCTURAL_TOKENS:
        if t in text:
            raise ValueError(f"{where}: {t} is written by the prompt builder")


def with_speaker_tag(text, speaker=0):
    return text if SPEAKER_TAG.match(text.lstrip(WHITE_SPACE)) else f"<|speaker:{speaker}|> {text}"


def build_prompt_rows(text, references, tokenizer):
    """references: list of (transcript, codes [10, frames]). Returns int64 rows [T, 11]."""
    sem0 = tokenizer.token_to_id("<|semantic:0|>")
    segments = ["<|im_start|>system\n", REFERENCE_INSTRUCTION if references else TTS_INSTRUCTION]
    for i, (transcript, codes) in enumerate(references):
        _check_text(transcript, f"references.{i}")
        assert codes.ndim == 2 and codes.shape[0] == NUM_CODEBOOKS and 0 <= codes.min() and codes.max() < CODEBOOK_SIZE
        tagged = with_speaker_tag(transcript, 0)
        body = tagged.lstrip(WHITE_SPACE)
        tag = SPEAKER_TAG.match(body)
        lead = tagged[: len(tagged) - len(body)]
        transcript = tag.group(0) + lead + body[tag.end() :]  # the tag follows "Text: " directly
        segments += [f"\n\n[Reference {i + 1}]\nText: {transcript}\nSpeech: ", codes]
    _check_text(text, "input")
    segments += [MESSAGE_END, "<|im_start|>user\n", with_speaker_tag(text, 0), MESSAGE_END, GENERATION_START]

    rows = []
    for seg in segments:  # each literal on its own: tokenizing the joined string changes ids at the seams
        if isinstance(seg, str):
            ids = torch.tensor(tokenizer.encode(seg).ids, dtype=torch.long)
            r = torch.zeros(ids.numel(), NUM_CODEBOOKS + 1, dtype=torch.long)
            r[:, 0] = ids
            rows.append(r)
        else:
            c = seg.long()
            rows.append(torch.cat([c[:1] + sem0, c]).T)
    return torch.cat(rows)


class FrameEmbedder:
    """Backbone input embedding of frame rows (Fish's embed_backbone, vq_backbone_codebooks = 1):
    text rows embed_tokens[id]; semantic rows (embed_tokens[id] + codebook_embeddings[code1 + 4096]) / sqrt(2)."""

    def __init__(self, embed_tokens, codebook_embeddings, semantic_start, semantic_end, backbone_codebooks=1):
        self.embed = embed_tokens  # [vocab, H] bf16
        self.cb = codebook_embeddings  # [10 * 4096, H] bf16
        self.sem0, self.sem1 = semantic_start, semantic_end
        self.k = backbone_codebooks
        self.scale = 1.0 / math.sqrt(1.0 + backbone_codebooks)

    def is_semantic(self, ids):
        return (ids >= self.sem0) & (ids <= self.sem1)

    def __call__(self, frames):
        """frames [N, 11] int64 -> FP32 [N, H]. Fish sums in the embedding dtype (bf16) and scales;
        here everything is FP32 from the BF16 tables, the device gets it rounded to BF16."""
        text = self.embed[frames[:, 0]].float()
        rows = frames[:, 2 : 2 + self.k].clamp(0, CODEBOOK_SIZE - 1) + torch.arange(1, 1 + self.k) * CODEBOOK_SIZE
        mixed = (text + self.cb[rows].float().sum(1)) * self.scale
        return torch.where(self.is_semantic(frames[:, 0]).unsqueeze(-1), mixed, text)

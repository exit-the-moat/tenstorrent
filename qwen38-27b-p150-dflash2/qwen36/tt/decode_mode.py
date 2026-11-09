# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Modified for Exit the Moat Episode 2 (2026-11-06): new file.
"""Explicit decode-step flag (Episode 2, Stage A1 of the 64-row decode rewrite).

Every decode/prefill branch in this port used to be chosen from the activation height: ``x.shape[-2] > 32``
meant prefill. That holds while a decode step is one tile row (B <= 32 rows), but not for a DFlash2 verify
step at 8 / 12 / 16 parallel streams (64 / 96 / 128 rows): those rows fell into the prefill branches with
the decode tensor layouts and failed (SCOREBOARD-night-ms.md, "What breaks at 64 rows").

The flag replaces the implicit test:

  model level   Qwen36DecoderLayer.forward enters ``with step(max_rows=...)`` for mode == "decode"; a prefill
                or chunk call never enters it.
  branch sites  ask ``decode_rows(rows)`` / ``prefill_rows(rows)`` instead of comparing with TILE_SIZE.

With no step active (every prefill, and every direct call outside a layer forward) the answers are exactly the
old shape test, so the existing 1- and 32-user paths are unchanged by construction:
``decode_rows(r) == (r <= TILE_SIZE)`` and ``prefill_rows(r) == (r > TILE_SIZE)``.

Sizes are in ROWS. ``tile_rows(rows)`` is how many 32-row tiles a step spans (1 today, 2 / 3 / 4 for 64 / 96 /
128 rows); code that splits a step must take that number from here, never a literal 1 or 2.

max_rows: QWEN36_MAX_DECODE_ROWS (default 64) = the widest decode step the decode branches must serve; 128 is
the 16-stream target. Stage A1 only routes the rows; the decode programs themselves at more than one tile row
are Stage A2. A decode step wider than max_rows falls back to the prefill branches (logged once) rather than
silently pretending it is decode.
"""
import contextlib
import os
import threading

TILE_SIZE = 32  # ttnn.TILE_SIZE; duplicated so importing this module stays cheap (no ttnn)

MAX_DECODE_ROWS = int(os.environ.get("QWEN36_MAX_DECODE_ROWS", "64"))

_LOG = os.environ.get("QWEN36_DM_LOG", "1") == "1"
_MAX_DECISIONS = 64

_local = threading.local()


def _st():
    """Per-thread step state: depth (re-entrant), max_rows of the innermost step, the branch decisions seen."""
    st = getattr(_local, "st", None)
    if st is None:
        st = _local.st = {"depth": 0, "max_rows": None, "decisions": [], "seen": set()}
    return st


def active():
    return _st()["depth"] > 0


def max_rows():
    return _st()["max_rows"] if active() else 0


@contextlib.contextmanager
def step(max_rows=None):
    """Mark the enclosed work as one decode step. Re-entrant; the innermost max_rows wins."""
    st = _st()
    prev = st["max_rows"]
    st["depth"] += 1
    st["max_rows"] = int(max_rows) if max_rows is not None else MAX_DECODE_ROWS
    try:
        yield
    finally:
        st["depth"] -= 1
        st["max_rows"] = prev


def tile_rows(rows):
    """Number of 32-row tiles a step of `rows` rows spans (>= 1)."""
    return max(1, -(-int(rows) // TILE_SIZE))


def decode_rows(rows, site=None):
    """True when `rows` must run the decode branches: always for one tile row (the old test), and for up to
    max_rows rows while a decode step is active. False otherwise (prefill)."""
    ok = rows <= TILE_SIZE or (active() and rows <= _st()["max_rows"])
    if site is not None:
        _note(site, rows, ok)
    return ok


def prefill_rows(rows, site=None):
    """The complement of decode_rows: True when `rows` must run the prefill branches."""
    dec = decode_rows(rows)
    if site is not None:
        _note(site, rows, dec)
    return not dec


def _note(site, rows, is_decode):
    st = _st()
    entry = (site, int(rows), "decode" if is_decode else "prefill")
    if entry in st["seen"]:
        return
    st["seen"].add(entry)
    if len(st["decisions"]) < _MAX_DECISIONS:
        st["decisions"].append(entry)
    if _LOG:
        from loguru import logger

        step = f"decode step, max_rows={st['max_rows']}" if active() else "no decode step"
        logger.info(
            f"ep2 decode_mode: {site} rows={entry[1]} ({tile_rows(entry[1])} tile rows) -> {entry[2]} [{step}]"
        )


def decisions():
    """The (site, rows, branch) triples seen so far by this thread (for probes and checks)."""
    return list(_st()["decisions"])


def reset_decisions():
    _st()["decisions"] = []

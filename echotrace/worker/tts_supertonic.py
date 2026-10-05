"""EchoTrace voice: Supertonic 3, one voice for Arabic and English.

    voice = SupertonicVoice(model_dir=".../supertonic3", voice="F2")
    pcm = voice.synthesize("يستخدم الطلاب برنامج Microsoft Word ...")   # float32, mono, 44.1 kHz

Text goes through ``speech_text.prepare`` first (pronunciation fixes, Arabic and
English runs). Each run is sent with its own language tag but the SAME voice
style, so one speaker reads both languages. The model pads every clip with about
0.3 s of silence at both ends; that is cut (``trim_silence``) before the runs are
joined with a short gap, so the audio returned has only ~40 ms of margin at each end
and the player must add the pause between sentences itself.
English must never be sent under the Arabic tag: the model then reads the Latin
letters badly (a doubled "s" in "Microsoft" was heard that way).

The model starts from random noise and, rarely, a draw comes out wrong. The noise is
therefore seeded from the text: a sentence sounds the same every time, so a bad one can
be reproduced and fixed. ``variant`` picks another draw, which is what a repeat of a
sentence the listener did not understand should use.

This module needs ``supertonic`` (and onnxruntime), which live in the worker's
Python environment, not inside NVDA.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from collections import OrderedDict
from typing import List, Optional

import numpy as np

from .speech_text import Lexicon, Run, prepare

SAMPLE_RATE = 44100
MAX_RUN_CHARS = 280                  # the model works best in chunks of at most ~300 characters


def split_long(text: str, limit: int = MAX_RUN_CHARS) -> List[str]:
    """Break an over-long run at the nearest comma, full stop or space."""
    if len(text) <= limit:
        return [text]
    out, rest = [], text
    while len(rest) > limit:
        cut = max((rest.rfind(c, 0, limit) for c in ("،", ",", ".", "؟", "?", "!", ":")), default=-1)
        if cut < limit // 2:
            cut = rest.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        out.append(rest[:cut + 1].strip())
        rest = rest[cut + 1:].strip()
    if rest:
        out.append(rest)
    return [c for c in out if c]


_ENDS_WITH_PUNCTUATION = re.compile(r"[.!?؟:;,،…]$")


def trim_silence(x: np.ndarray, floor_db: float = -48.0, margin: float = 0.04,
                 hop: float = 0.01) -> np.ndarray:
    """Cut the silence the model leaves at both ends (about 0.3 s each), keeping a short margin.

    Without this, every Arabic/English switch inside a sentence has about a second of dead air."""
    h = int(SAMPLE_RATE * hop)
    n = len(x) // h
    if n < 2:
        return x
    rms = np.sqrt((x[:n * h].astype(np.float64).reshape(n, h) ** 2).mean(1))
    if rms.max() <= 0:
        return x
    loud = np.flatnonzero(rms > rms.max() * 10 ** (floor_db / 20))
    pad = int(SAMPLE_RATE * margin)
    return x[max(0, loud[0] * h - pad): min(len(x), (loud[-1] + 1) * h + pad)]


def seed_for(*parts) -> int:
    """A stable 32-bit seed from the text and settings (not Python's per-process hash())."""
    digest = hashlib.blake2b("\x1f".join(str(p) for p in parts).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


class SupertonicVoice:
    def __init__(self, model_dir: Optional[str] = None, voice: str = "F2", steps: int = 8,
                 speed: float = 1.05, lexicon: Optional[Lexicon] = None,
                 gap_seconds: float = 0.04, cache_items: int = 64, continuation: bool = True,
                 engine=None):
        if engine is None:
            if model_dir:
                os.environ["SUPERTONIC_CACHE_DIR"] = model_dir
            from supertonic import TTS                  # heavy import, only in the worker
            engine = TTS(model="supertonic-3", auto_download=False)
        self._tts = engine                              # tests pass a stand-in here
        self.voice = voice
        self._style = self._tts.get_voice_style(voice_name=voice)
        self.steps = steps
        self.speed = speed
        self.lexicon = lexicon or Lexicon()
        self.gap = np.zeros(int(gap_seconds * SAMPLE_RATE), np.float32)
        self.continuation = continuation                # a run that is not the last ends with a comma (chosen by ear)
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._cache_items = cache_items
        self._lock = threading.Lock()                   # the seeded noise is process-wide state

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    # -- one chunk ---------------------------------------------------------------
    def _synth_chunk(self, chunk: str, lang: str, steps: int, speed: float, variant: int) -> np.ndarray:
        seed = seed_for(lang, chunk, self.voice, steps, round(speed, 3), variant)
        with self._lock:
            state = np.random.get_state()               # the library draws from numpy's global noise
            np.random.seed(seed)
            try:
                wav, _ = self._tts.synthesize(chunk, voice_style=self._style, lang=lang,
                                              total_steps=steps, speed=speed, silence_duration=0.0)
            finally:
                np.random.set_state(state)
        return np.squeeze(wav).astype(np.float32)

    def _unreadable_dropped(self, chunk: str) -> str:
        ok = self._tts.model.text_processor.supported_character_set
        return "".join(c for c in chunk if c in ok or c.isspace() or c.isascii())

    # -- one run -----------------------------------------------------------------
    def _speak_run(self, run: Run, steps: int, speed: float, variant: int) -> np.ndarray:
        parts = []
        for chunk in split_long(run.text):
            try:
                parts.append(self._synth_chunk(chunk, run.lang, steps, speed, variant))
            except Exception:
                # A character the model cannot read must never stop the reading: drop it and retry once.
                safe = self._unreadable_dropped(chunk)
                if not safe.strip():
                    continue
                parts.append(self._synth_chunk(safe, run.lang, steps, speed, variant))
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    # -- a sentence ----------------------------------------------------------------
    def synthesize(self, text: str, speed: Optional[float] = None, steps: Optional[int] = None,
                   variant: int = 0) -> np.ndarray:
        steps = steps or self.steps
        speed = speed or self.speed
        key = (text, self.voice, steps, round(speed, 3), variant, self.continuation)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        pieces: List[np.ndarray] = []
        runs = prepare(text, self.lexicon)
        for i, run in enumerate(runs):
            if self.continuation and i < len(runs) - 1 and not _ENDS_WITH_PUNCTUATION.search(run.text):
                # the model ends a bare run like a finished sentence (falling voice); a comma says "more to come"
                run = Run(run.lang, run.text + ("،" if run.lang == "ar" else ","))
            audio = trim_silence(self._speak_run(run, steps, speed, variant))
            if len(audio):
                if pieces:
                    pieces.append(self.gap)
                pieces.append(audio)
        out = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
        self._cache[key] = out
        while len(self._cache) > self._cache_items:
            self._cache.popitem(last=False)
        return out

    def synthesize_int16(self, text: str, **kw) -> np.ndarray:
        wav = np.clip(self.synthesize(text, **kw), -1.0, 1.0)
        return (wav * 32767).astype(np.int16)

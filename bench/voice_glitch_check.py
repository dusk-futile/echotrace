#!/usr/bin/env python3
"""Does the voice repeat a sound? Counts renderings with two "s"-like noise bursts close together
(the "microso soft" glitch) over many random draws.

It looks at the spectrogram, so it catches that one kind of glitch (a repeated "s", "sh" or "x"
sound) and nothing else; naturalness is for human ears. Run in the TTS environment from the
project root:

    export SUPERTONIC_CACHE_DIR="$PWD/bench/data/supertonic3"
    PY=bench/.venv-tts/bin/python

    # the whole pipeline, as the learner hears it (draw = variant 0, 1, 2 ...)
    $PY bench/voice_glitch_check.py pipeline "يستخدم الطلاب برنامج Microsoft Word ..." --draws 100

    # one piece sent straight to the model under a chosen tag (the old path was: the whole sentence, --tag ar)
    $PY bench/voice_glitch_check.py raw "Microsoft Word" --tag en --draws 400

Measured on 5 Oct 2026 (voice F2, 8 steps): whole mixed sentence under the Arabic tag 14 of 190 draws had
the glitch; "Microsoft Word" under the English tag 0 of 500. See RESULTS.md.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from echotrace.worker.tts_supertonic import SAMPLE_RATE, SupertonicVoice  # noqa: E402


def bursts(x: np.ndarray, hi=(4500, 10500), ratio=0.45, loud_db=-38.0, win=0.025, hop=0.005, min_len=0.03):
    """Time spans (start, end) where most of the energy is above 4.5 kHz: "s", "sh", "x"."""
    w, h = int(SAMPLE_RATE * win), int(SAMPLE_RATE * hop)
    window = np.hanning(w)
    power = np.abs(np.fft.rfft(np.array([x[i:i + w] * window for i in range(0, len(x) - w, h)]), axis=1)) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(w, 1 / SAMPLE_RATE)
    total = power.sum(1)
    high = power[:, (freqs >= hi[0]) & (freqs <= hi[1])].sum(1)
    on = (high / total > ratio) & (10 * np.log10(total / total.max()) > loud_db)
    spans, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if not v and start is not None:
            spans.append((start * hop, i * hop))
            start = None
    if start is not None:
        spans.append((start * hop, len(on) * hop))
    return [(a, b) for a, b in spans if b - a >= min_len]


def close_pairs(spans, gap=0.35):
    return [(a[0], b[1]) for a, b in zip(spans, spans[1:]) if b[0] - a[1] < gap]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("pipeline", "raw"))
    ap.add_argument("text")
    ap.add_argument("--tag", default="en", help="language tag for raw mode (ar, en, na)")
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--voice", default="F2")
    ap.add_argument("--first-seed", type=int, default=0, help="raw mode: seeds first_seed ...")
    ap.add_argument("--save-bad", metavar="DIR", help="write the glitchy draws here as WAV files")
    a = ap.parse_args()

    voice = SupertonicVoice(model_dir=os.path.join(HERE, "data", "supertonic3"), voice=a.voice, steps=a.steps,
                            cache_items=1)
    if a.save_bad:
        os.makedirs(a.save_bad, exist_ok=True)
    bad = []
    for i in range(a.draws):
        if a.mode == "pipeline":
            x = voice.synthesize(a.text, variant=i)
        else:
            np.random.seed(a.first_seed + i)              # the model draws its noise from numpy's global state
            wav, _ = voice._tts.synthesize(a.text, voice_style=voice._style, lang=a.tag, total_steps=a.steps,
                                           speed=voice.speed)
            x = np.squeeze(wav)
        pairs = close_pairs(bursts(x.astype(np.float64)))
        if pairs:
            bad.append(i)
            if a.save_bad:
                sf.write(os.path.join(a.save_bad, f"draw{i:04d}.wav"), x, SAMPLE_RATE, subtype="PCM_16")
            print(f"draw {i}: repeated sound at {[(round(s, 2), round(e, 2)) for s, e in pairs]}", flush=True)
    print(f"\n{len(bad)} of {a.draws} draws had a repeated sound ({100 * len(bad) / a.draws:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

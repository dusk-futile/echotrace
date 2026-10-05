#!/usr/bin/env python3
"""Voice listening test, round 2: Supertonic 3 (one offline model, Arabic and English, 10 voices).

Run inside the TTS environment, with the model cache pointing at the project data folder:

    SUPERTONIC_CACHE_DIR="$PWD/bench/data/supertonic3" \
      bench/.venv-tts/bin/python bench/voice_bakeoff2.py

Writes WAV files and results/voices2/index.html. Piper (round 1) files are linked
for side-by-side comparison, so run voice_bakeoff.py first.

What it answers
  * Which of the 10 preset voices sounds best for Arabic textbook text?
  * Arabic with English words: does the same voice cope when the whole sentence is
    sent as Arabic, or is it better to send the English words as English (same voice,
    so one speaker reads both)?
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import time

import numpy as np
import soundfile as sf
from supertonic import TTS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "voices2")
RATE = 44100

VOICES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
FULL_VOICES = ["M1", "M2", "F1", "F2"]          # also get the long paragraph

S1 = ("التمثيل الضوئي هو العملية التي تحول بها النباتات الخضراء ضوء الشمس والماء "
      "وثاني أكسيد الكربون إلى غذاء وأكسجين.")
S3 = "يستخدم الطلاب برنامج Microsoft Word لكتابة الواجب، ثم يحفظونه باسم Homework.docx."
S7 = ("تعد مصر من أقدم الحضارات في التاريخ، فقد نشأت على ضفاف نهر النيل قبل آلاف السنين، "
      "وبنى المصريون القدماء الأهرامات والمعابد التي لا تزال شاهدة على عبقريتهم في الهندسة والفلك والطب. "
      "وتحتفظ المتاحف المصرية اليوم بآلاف القطع الأثرية التي تجذب الزوار من كل أنحاء العالم.")
S8 = "Press the space bar to pause, and the left arrow to go back one sentence."

_ARABIC_RUN = re.compile(r"[؀-ۿݐ-ݿ]+")


def prep(text: str) -> str:
    """Two characters the model cannot read: Arabic semicolon and alef wasla."""
    return text.replace("؛", "،").replace("ٱ", "ا").replace("ـ", "")


def split_runs(text: str):
    out, pos = [], 0
    for m in _ARABIC_RUN.finditer(text):
        if m.start() > pos:
            out.append(("en", text[pos:m.start()]))
        out.append(("ar", m.group()))
        pos = m.end()
    if pos < len(text):
        out.append(("en", text[pos:]))
    merged = []
    for lang, seg in out:               # punctuation-only fragments join their Arabic neighbour
        if lang == "en" and not re.search(r"[A-Za-z0-9]", seg) and merged:
            merged[-1] = (merged[-1][0], merged[-1][1] + seg)
        else:
            merged.append((lang, seg))
    return merged


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    tts = TTS(model="supertonic-3", auto_download=False)
    styles = {v: tts.get_voice_style(voice_name=v) for v in VOICES}
    tts.synthesize("مرحبا", voice_style=styles["M1"], lang="ar")           # warm-up

    results = []

    def render(sid, vname, label, text, lang, parts=None):
        t0 = time.perf_counter()
        if parts is None:
            wav, _ = tts.synthesize(prep(text), voice_style=styles[vname], lang=lang)
            samples = np.squeeze(wav)
        else:                                                              # split by language
            chunks = []
            for rl, seg in parts:
                w, _ = tts.synthesize(prep(seg), voice_style=styles[vname], lang=rl)
                chunks += [np.squeeze(w), np.zeros(int(0.10 * RATE), np.float32)]
            samples = np.concatenate(chunks)
        took = time.perf_counter() - t0
        fn = f"{sid}__{vname}__{label}.wav"
        sf.write(os.path.join(OUT, fn), samples, RATE, subtype="PCM_16")
        dur = len(samples) / RATE
        results.append(dict(sid=sid, voice=vname, label=label, file=fn,
                            audio_s=round(dur, 2), took_s=round(took, 2), rtf=round(took / dur, 2)))
        print(f"{sid} {vname} {label:12s} audio {dur:5.1f}s took {took:5.2f}s RTF {took / dur:4.2f}", flush=True)

    for v in VOICES:
        render("s1", v, "ar", S1, "ar")
        render("s3", v, "whole_ar", S3, "ar")
        render("s3", v, "split", S3, None, parts=split_runs(S3))
    for v in FULL_VOICES:
        render("s7", v, "ar", S7, "ar")
    render("s8", "M1", "en", S8, "en")
    render("s8", "F1", "en", S8, "en")

    meta = dict(results=results)
    json.dump(meta, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1)
    write_html(results)
    print("\nOpen", os.path.join(OUT, "index.html"))
    return 0


def player(fn: str, caption: str, small: str = "") -> str:
    return (f"<div class=v><b>{html.escape(caption)}</b><br>"
            f"<audio controls preload=none src='{html.escape(fn)}'></audio>"
            f"{('<br><small>' + html.escape(small) + '</small>') if small else ''}</div>")


def write_html(results) -> None:
    by = {}
    for r in results:
        by.setdefault((r["sid"], r["label"]), []).append(r)

    def block(title, text, sid, label, rtl=True, ref=None):
        cells = [player(r["file"], f"Supertonic {r['voice']}", f"RTF {r['rtf']}") for r in by.get((sid, label), [])]
        if ref:
            cells = [player(ref[0], ref[1], "Piper (round 1)")] + cells
        d = "rtl" if rtl else "ltr"
        return f"<section><h3>{html.escape(title)}</h3><p dir={d} class=t>{html.escape(text)}</p>{''.join(cells)}</section>"

    rows = [
        block("1. Textbook sentence: all 10 voices", S1, "s1", "ar",
              ref=("../voices/s1__kareem_medium.wav", "Piper kareem_medium")),
        block("2a. Arabic with English words: sent as ONE Arabic sentence", S3, "s3", "whole_ar",
              ref=("../voices/s3__kareem_medium.wav", "Piper kareem_medium")),
        block("2b. Same sentence, English words sent as English (the same voice reads both)", S3, "s3", "split",
              ref=("../voices/s3__split_kareem_plus_lessac.wav", "Piper: two voices")),
        block("3. Long paragraph (stamina and rhythm), 4 voices", S7, "s7", "ar",
              ref=("../voices/s7__kareem_medium.wav", "Piper kareem_medium")),
        block("4. English instructions", S8, "s8", "en", rtl=False,
              ref=("../voices/s8__lessac_en.wav", "Piper lessac")),
    ]
    page = f"""<!doctype html><meta charset=utf-8><title>EchoTrace voices, round 2: Supertonic 3</title>
<style>body{{font:16px system-ui;max-width:960px;margin:2em auto;padding:0 1em}}
.t{{font-size:1.35em;line-height:1.7}} .v{{display:inline-block;margin:.4em 1em .4em 0;vertical-align:top}}
section{{border-top:1px solid #ccc;padding:.6em 0}} small{{color:#555}}</style>
<h1>EchoTrace voices, round 2: Supertonic 3</h1>
<p>One small offline model (about 99M parameters) that reads Arabic and English, with 10 preset voices
(F = female, M = male). Piper from round 1 is in each section for comparison.</p>
<p><b>What to listen for:</b> (1) which voice sounds most natural for Arabic; (2a vs 2b) whether the English
words come out better when sent as English; (3) whether a long paragraph stays pleasant.
Then tell us the voice names you like (for example "F2 and M3").</p>
{''.join(rows)}
<p><small>RTF = synthesis seconds per audio second on this Mac (below 1 is faster than real time); an old laptop will be slower.</small></p>"""
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(page)


if __name__ == "__main__":
    sys.exit(main())

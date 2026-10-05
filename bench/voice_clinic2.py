#!/usr/bin/env python3
"""Voice clinic 2: how Arabic and English are joined inside one sentence (voice F2).

Run in the TTS environment from the project root:

    SUPERTONIC_CACHE_DIR="$PWD/bench/data/supertonic3" \
      bench/.venv-tts/bin/python bench/voice_clinic2.py

Writes results/voices4/index.html. For each sentence:
  before  the clinic-1 version (about a second of dead air at every Arabic/English switch)
  A       silence trimmed at every join
  B       silence trimmed, and a run that is not the last ends with a comma ("more is coming")
"""
from __future__ import annotations

import html
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from echotrace.worker.speech_text import prepare                      # noqa: E402
from echotrace.worker.tts_supertonic import SAMPLE_RATE, SupertonicVoice  # noqa: E402

OUT = os.path.join(HERE, "results", "voices4")

S3 = "يستخدم الطلاب برنامج Microsoft Word لكتابة الواجب، ثم يحفظونه باسم Homework.docx."
S9 = "افتح ملف PDF من برنامج Excel ثم اضغط Ctrl+S لحفظ الملف على USB."
S10 = "اكتب العنوان www.school.edu.eg في متصفح Chrome ثم أرسل الملف إلى info@school.edu.eg."
S11 = "اضغط على زر Start ثم اختر Settings ثم Accessibility لتغيير حجم الخط."
S12 = "ابحث في Google عن كلمة Photosynthesis ثم افتح موقع Wikipedia لقراءة الشرح."
# (id, sentence, the clinic-1 file it replaces, or None)
SENTENCES = [("s3", S3, "../voices3/d_s3_new.wav"), ("s9", S9, "../voices3/e_s9_acronyms_keys.wav"),
             ("s10", S10, "../voices3/e_s10_web.wav"), ("s11", S11, None), ("s12", S12, None)]


def longest_silence(x: np.ndarray, thr_db: float = -45.0, hop: float = 0.01) -> float:
    """Longest quiet stretch strictly inside the audio, in seconds."""
    h = int(SAMPLE_RATE * hop)
    n = len(x) // h
    rms = np.sqrt((x[:n * h].astype(np.float64).reshape(n, h) ** 2).mean(1) + 1e-12)
    quiet = 20 * np.log10(rms / rms.max()) < thr_db
    idx = np.flatnonzero(~quiet)
    best = run = 0
    for q in quiet[idx[0]:idx[-1] + 1]:
        run = run + 1 if q else 0
        best = max(best, run)
    return best * hop


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    plain = SupertonicVoice(model_dir=os.path.join(HERE, "data", "supertonic3"), voice="F2", steps=8,
                            continuation=False)
    flow = SupertonicVoice(voice="F2", steps=8, continuation=True, engine=plain._tts)
    plain.synthesize("مرحبا")                                           # warm-up
    log = []
    for sid, text, before in SENTENCES:
        for tag, v in (("A", plain), ("B", flow)):
            t = time.perf_counter()
            x = v.synthesize(text)
            took = time.perf_counter() - t
            fn = f"{sid}_{tag}.wav"
            sf.write(os.path.join(OUT, fn), x, SAMPLE_RATE, subtype="PCM_16")
            log.append(dict(id=sid, tag=tag, file=fn, audio_s=round(len(x) / SAMPLE_RATE, 2), took_s=round(took, 2),
                            longest_pause_s=round(longest_silence(x), 2),
                            runs=[(r.lang, r.text) for r in prepare(text)]))
            print(f"{fn:10s} {len(x) / SAMPLE_RATE:5.1f}s  longest pause {longest_silence(x):.2f}s", flush=True)
        if before:
            old, sr = sf.read(os.path.join(HERE, "results", "voices3", os.path.basename(before)))
            log.append(dict(id=sid, tag="before", file=before, audio_s=round(len(old) / sr, 2),
                            longest_pause_s=round(longest_silence(np.squeeze(old)), 2)))
    json.dump(log, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    write_html(log)
    print("\nOpen", os.path.join(OUT, "index.html"))
    return 0


def write_html(log) -> None:
    def clip(r, caption):
        return (f"<div class=v><b>{html.escape(caption)}</b><br><audio controls preload=none src='{r['file']}'></audio>"
                f"<br><small>longest pause inside: {r['longest_pause_s']} s</small></div>")

    blocks = []
    for sid, text, before in SENTENCES:
        rows = {r["tag"]: r for r in log if r["id"] == sid}
        parts = []
        if "before" in rows:
            parts.append(clip(rows["before"], "before (clinic 1)"))
        parts.append(clip(rows["A"], "A: silence trimmed"))
        parts.append(clip(rows["B"], "B: trimmed, with commas"))
        blocks.append(f"<div class=blk><p dir=rtl class=t>{html.escape(text)}</p>{''.join(parts)}</div>")
    page = f"""<!doctype html><meta charset=utf-8><title>EchoTrace voice clinic 2: joins</title>
<style>body{{font:16px system-ui;max-width:980px;margin:2em auto;padding:0 1em}}
.t{{font-size:1.3em;line-height:1.7;margin:.3em 0}} .v{{display:inline-block;margin:.4em 1em .4em 0;vertical-align:top;max-width:300px}}
.blk{{margin:1em 0;border-top:1px solid #ccc;padding-top:.6em}} small{{color:#555}}</style>
<h1>EchoTrace voice clinic 2: Arabic and English in one sentence</h1>
<p>Voice F2. English words are read by the English side of the same voice. The model leaves about a third of a second of
silence at both ends of every piece, so the first version had close to a second of dead air at every switch.</p>
<ul><li><b>A</b>: that silence is cut, so the switch sounds like a normal short pause.</li>
<li><b>B</b>: same, and every piece that is not the last ends with a comma, so the voice sounds like “more is coming”
instead of finishing a sentence.</li></ul>
<p>For each sentence, tell us: does <b>A</b> or <b>B</b> flow better, and is any word still misread?</p>
{''.join(blocks)}"""
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(page)


if __name__ == "__main__":
    if "--html-only" in sys.argv:
        write_html(json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8")))
        sys.exit(0)
    sys.exit(main())

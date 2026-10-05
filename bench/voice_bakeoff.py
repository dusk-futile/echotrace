#!/usr/bin/env python3
"""Voice listening test: the same sentences in every voice we can run here.

Run inside the TTS environment (it needs the `piper` package):

    bench/.venv-tts/bin/python bench/voice_bakeoff.py

Writes WAV files and an HTML page to bench/results/voices/. Open
index.html, listen, and tell the project which voice names sound best.

Also records, per voice, the time to the FIRST audio chunk (what a listener
waits for after pressing a key) and the real-time factor (synthesis seconds per
audio second; below 1.0 means faster than real time). These are numbers from
THIS Mac (Intel i3-8100); an old laptop will be slower.
"""
from __future__ import annotations

import html
import json
import os
import re
import resource
import sys
import time
import wave

import numpy as np
from piper import PiperVoice, SynthesisConfig

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(HERE, "data", "voices")
OUT = os.path.join(HERE, "results", "voices")

# name -> (model file, diacritize Arabic?)
VOICES = {
    "kareem_low": ("ar_JO-kareem-low.onnx", True),
    "kareem_medium": ("ar_JO-kareem-medium.onnx", True),
    "kareem_medium_no_tashkeel": ("ar_JO-kareem-medium.onnx", False),
    "lessac_en": ("en_US-lessac-medium.onnx", True),
}

ARABIC_VOICES = ["kareem_low", "kareem_medium", "kareem_medium_no_tashkeel"]

# id, language, label, text, voices
SENTENCES = [
    ("s1", "ar", "MSA textbook (science)",
     "التمثيل الضوئي هو العملية التي تحول بها النباتات الخضراء ضوء الشمس والماء وثاني أكسيد الكربون إلى غذاء وأكسجين.",
     ARABIC_VOICES),
    ("s2", "ar", "MSA with numbers",
     "تأسست جامعة القاهرة عام 1908، ويدرس بها اليوم أكثر من 250 ألف طالب وطالبة.",
     ARABIC_VOICES),
    ("s3", "mixed", "Mixed Arabic and English",
     "يستخدم الطلاب برنامج Microsoft Word لكتابة الواجب، ثم يحفظونه باسم Homework.docx.",
     ARABIC_VOICES),
    ("s4", "ar", "Literary MSA",
     "كان الليل طويلا، وكانت النجوم تتلألأ فوق الصحراء كأنها عيون ساهرة لا تنام.",
     ARABIC_VOICES),
    ("s5", "ar", "Instructions (what EchoTrace would say)",
     "اضغط على مفتاح المسافة للإيقاف المؤقت، وعلى السهم الأيسر للرجوع إلى الجملة السابقة.",
     ARABIC_VOICES),
    ("s6", "ar", "Egyptian dialect (assistant greeting; expect an MSA voice to sound off)",
     "أهلاً بيك! أنا هنا عشان أقرأ لك أي حاجة على الشاشة. قول لي تحب أقرأ لك إيه؟",
     ARABIC_VOICES),
    ("s7", "ar", "Long paragraph (stamina and rhythm)",
     "تعد مصر من أقدم الحضارات في التاريخ، فقد نشأت على ضفاف نهر النيل قبل آلاف السنين، "
     "وبنى المصريون القدماء الأهرامات والمعابد التي لا تزال شاهدة على عبقريتهم في الهندسة والفلك والطب. "
     "وتحتفظ المتاحف المصرية اليوم بآلاف القطع الأثرية التي تجذب الزوار من كل أنحاء العالم.",
     ARABIC_VOICES),
    ("s8", "en", "English instructions",
     "Press the space bar to pause, and the left arrow to go back one sentence.",
     ["lessac_en"]),
]

NOT_GENERATED = [
    ("Windows OneCore Hoda / Naayf", "needs a Windows 10/11 PC with the Arabic voice pack; sample it on a lab laptop"),
    ("Azure ar-EG-SalmaNeural / ShakirNeural", "online only, best Egyptian Arabic; sample from Microsoft's voice gallery"),
    ("SILMA TTS (150M, Apache-2.0)", "server tier: needs a GPU, too heavy for the old laptops"),
]

_ARABIC_RUN = re.compile(r"[؀-ۿݐ-ݿ]+")


def rss_mb() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round((r / 1e6) if sys.platform == "darwin" else (r / 1e3))


def write_wav(path: str, samples: np.ndarray, rate: int) -> float:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype(np.int16).tobytes())
    return len(samples) / rate


def synth(voice: PiperVoice, text: str):
    """Returns (samples, rate, secs_to_first_chunk, secs_total)."""
    t0 = time.perf_counter()
    first = None
    chunks = []
    rate = voice.config.sample_rate
    for ch in voice.synthesize(text, syn_config=SynthesisConfig()):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(ch.audio_int16_array)
        rate = ch.sample_rate
    total = time.perf_counter() - t0
    samples = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
    return samples, rate, first or total, total


def split_runs(text: str):
    """Alternate Arabic and non-Arabic runs: ('ar', text) / ('en', text)."""
    out, pos = [], 0
    for m in _ARABIC_RUN.finditer(text):
        if m.start() > pos:
            out.append(("en", text[pos:m.start()]))
        out.append(("ar", m.group()))
        pos = m.end()
    if pos < len(text):
        out.append(("en", text[pos:]))
    # attach punctuation/space-only fragments to the Arabic neighbours
    merged = []
    for lang, seg in out:
        if lang == "en" and not re.search(r"[A-Za-z0-9]", seg) and merged:
            merged[-1] = (merged[-1][0], merged[-1][1] + seg)
        else:
            merged.append((lang, seg))
    return merged


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    loaded, load_secs = {}, {}
    for name, (model, tashkeel) in VOICES.items():
        path = os.path.join(VOICE_DIR, model)
        if not os.path.exists(path):
            print(f"missing {path}; run fetch_data.py voices first")
            return 1
        t = time.perf_counter()
        v = PiperVoice.load(path)
        v.use_tashkeel = tashkeel
        loaded[name] = v
        load_secs[name] = round(time.perf_counter() - t, 2)
    # Warm-up: the diacritizer model loads on first use and must not be charged to sentence 1.
    for name in ("kareem_medium", "lessac_en"):
        synth(loaded[name], "مرحبا" if name != "lessac_en" else "Hello.")

    results = []
    for sid, lang, label, text, voices in SENTENCES:
        for vn in voices:
            samples, rate, first, total = synth(loaded[vn], text)
            fn = f"{sid}__{vn}.wav"
            dur = write_wav(os.path.join(OUT, fn), samples, rate)
            results.append(dict(sid=sid, voice=vn, file=fn, audio_s=round(dur, 2),
                                first_chunk_s=round(first, 2), total_s=round(total, 2),
                                rtf=round(total / dur, 2) if dur else None))
            print(f"{sid} {vn:28s} audio {dur:5.1f}s  first chunk {first:4.2f}s  RTF {total / dur:4.2f}")
        if lang == "mixed":
            # Plan idea for mixed text: Arabic runs in the Arabic voice, Latin runs in the English voice.
            parts, t0 = [], time.perf_counter()
            for rl, seg in split_runs(text):
                v = loaded["kareem_medium"] if rl == "ar" else loaded["lessac_en"]
                s, r_, _, _ = synth(v, seg)
                if r_ != 22050:
                    raise RuntimeError("unexpected sample rate")
                parts.append(s)
                parts.append(np.zeros(int(0.12 * r_), np.int16))
            total = time.perf_counter() - t0
            samples = np.concatenate(parts)
            fn = f"{sid}__split_kareem_plus_lessac.wav"
            dur = write_wav(os.path.join(OUT, fn), samples, 22050)
            results.append(dict(sid=sid, voice="split_kareem_plus_lessac", file=fn,
                                audio_s=round(dur, 2), first_chunk_s=None,
                                total_s=round(total, 2), rtf=round(total / dur, 2)))
            print(f"{sid} split_kareem_plus_lessac       audio {dur:5.1f}s  RTF {total / dur:4.2f}")

    meta = dict(load_seconds=load_secs, peak_rss_mb=rss_mb(), results=results)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    write_html(meta)
    print(f"\nOpen {os.path.join(OUT, 'index.html')}")
    print(f"voice load seconds: {load_secs}; peak RSS {rss_mb()} MB")
    return 0


def write_html(meta) -> None:
    by_sid = {}
    for r in meta["results"]:
        by_sid.setdefault(r["sid"], []).append(r)
    rows = []
    for sid, lang, label, text, _ in SENTENCES:
        dirn = "ltr" if lang == "en" else "rtl"
        cells = []
        for r in by_sid.get(sid, []):
            fc = f"first audio {r['first_chunk_s']}s, " if r["first_chunk_s"] is not None else ""
            cells.append(
                f"<div class=v><b>{html.escape(r['voice'])}</b><br>"
                f"<audio controls preload=none src='{r['file']}'></audio><br>"
                f"<small>{fc}RTF {r['rtf']}</small></div>")
        rows.append(f"<section><h3>{html.escape(label)}</h3>"
                    f"<p dir={dirn} class=t>{html.escape(text)}</p>{''.join(cells)}</section>")
    todo = "".join(f"<li><b>{html.escape(a)}</b>: {html.escape(b)}</li>" for a, b in NOT_GENERATED)
    page = f"""<!doctype html><meta charset=utf-8><title>EchoTrace voice listening test</title>
<style>body{{font:16px system-ui;max-width:900px;margin:2em auto;padding:0 1em}}
.t{{font-size:1.4em;line-height:1.7}} .v{{display:inline-block;margin:.4em 1em .4em 0;vertical-align:top}}
section{{border-top:1px solid #ccc;padding:.6em 0}} small{{color:#555}}</style>
<h1>EchoTrace voice listening test</h1>
<p>Same sentences, every voice we can run offline. Tell us which voice names sound best for textbooks.
RTF = synthesis seconds per audio second on this Mac (below 1 is faster than real time); old laptops are slower.</p>
<p><b>kareem_medium_no_tashkeel</b> turns off the automatic Arabic vowel-marking, so you can hear what it adds.
<b>split_kareem_plus_lessac</b> reads the Arabic parts with the Arabic voice and the English parts with the English voice.</p>
{''.join(rows)}
<h3>Not generated here</h3><ul>{todo}</ul>
<p><small>Voice load seconds: {html.escape(json.dumps(meta['load_seconds']))}; peak memory {meta['peak_rss_mb']} MB.</small></p>"""
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)


if __name__ == "__main__":
    sys.exit(main())

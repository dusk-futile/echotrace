#!/usr/bin/env python3
"""Voice clinic: Supertonic F2, the "docx" problem, and speed settings.

Run in the TTS environment from the project root:

    SUPERTONIC_CACHE_DIR="$PWD/bench/data/supertonic3" \
      bench/.venv-tts/bin/python bench/voice_clinic.py

Writes results/voices3/index.html. Sections:
  A. the sentence that went wrong, as heard in round 2, and its two English fragments alone
  B. candidate spoken forms for the file name "Homework.docx"
  C. candidate spoken forms for "Microsoft Word"
  D. the new pipeline (pronunciation layer + one voice for both languages) on the same sentence
  E. the other test sentences through the new pipeline (numbers, acronyms, keys, a web address)
  F. speed against quality: fewer synthesis steps are faster on an old laptop; how do they sound?
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

from echotrace.worker.speech_text import Lexicon, prepare            # noqa: E402
from echotrace.worker.tts_supertonic import SAMPLE_RATE, SupertonicVoice  # noqa: E402

OUT = os.path.join(HERE, "results", "voices3")
VOICE = "F2"

S1 = ("التمثيل الضوئي هو العملية التي تحول بها النباتات الخضراء ضوء الشمس والماء "
      "وثاني أكسيد الكربون إلى غذاء وأكسجين.")
S2 = "تأسست جامعة القاهرة عام 1908، ويدرس بها اليوم أكثر من 250 ألف طالب وطالبة."
S3 = "يستخدم الطلاب برنامج Microsoft Word لكتابة الواجب، ثم يحفظونه باسم Homework.docx."
S4 = "كان الليل طويلا، وكانت النجوم تتلألأ فوق الصحراء كأنها عيون ساهرة لا تنام."
S5 = "اضغط على مفتاح المسافة للإيقاف المؤقت، وعلى السهم الأيسر للرجوع إلى الجملة السابقة."
S6 = "أهلاً بيك! أنا هنا عشان أقرأ لك أي حاجة على الشاشة. قول لي تحب أقرأ لك إيه؟"
S7 = ("تعد مصر من أقدم الحضارات في التاريخ، فقد نشأت على ضفاف نهر النيل قبل آلاف السنين، "
      "وبنى المصريون القدماء الأهرامات والمعابد التي لا تزال شاهدة على عبقريتهم في الهندسة والفلك والطب.")
S8 = "Press the space bar to pause, and the left arrow to go back one sentence."
S9 = "افتح ملف PDF من برنامج Excel ثم اضغط Ctrl+S لحفظ الملف على USB."
B = [("b1_as_is", "Homework.docx."), ("b2_dot_docx", "Homework dot docx."),
     ("b3_dot_doc_x", "Homework dot doc x."), ("b4_letters", "Homework dot D O C X."),
     ("b5_dock_x", "Homework dot dock x."), ("b6_docs", "Homework dot docs.")]
C = [("c1_as_is", "Microsoft Word", "en"), ("c2_period", "Microsoft Word.", "en"),
     ("c3_micro_soft", "Micro soft Word.", "en"), ("c4_na", "Microsoft Word", "na"),
     ("c5_arabic_tag", "Microsoft Word", "ar")]
S10 = "اكتب العنوان www.school.edu.eg في متصفح Chrome ثم أرسل الملف إلى info@school.edu.eg."


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    if "--html-only" in sys.argv:
        write_html(json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8")))
        print("rebuilt", os.path.join(OUT, "index.html"))
        return 0
    voice = SupertonicVoice(model_dir=os.path.join(HERE, "data", "supertonic3"), voice=VOICE, steps=8)
    tts, style = voice._tts, voice._style
    voice.synthesize("مرحبا")                                           # warm-up
    log = []

    def save(name, samples, took, note=""):
        fn = f"{name}.wav"
        sf.write(os.path.join(OUT, fn), samples, SAMPLE_RATE, subtype="PCM_16")
        dur = len(samples) / SAMPLE_RATE
        log.append(dict(name=name, file=fn, audio_s=round(dur, 2), took_s=round(took, 2),
                        rtf=round(took / dur, 2) if dur else None, note=note))
        print(f"{name:34s} audio {dur:5.1f}s RTF {took / dur if dur else 0:4.2f}", flush=True)
        return fn

    def raw(name, text, lang, steps=8, speed=1.05, note=""):
        t = time.perf_counter()
        wav, _ = tts.synthesize(text, voice_style=style, lang=lang, total_steps=steps, speed=speed)
        return save(name, np.squeeze(wav), time.perf_counter() - t, note)

    def pipe(name, text, steps=8, speed=1.05, lex=None, note=""):
        v = voice if lex is None else SupertonicVoice(model_dir=os.path.join(HERE, "data", "supertonic3"),
                                                      voice=VOICE, steps=steps, lexicon=lex)
        t = time.perf_counter()
        samples = v.synthesize(text, speed=speed, steps=steps)
        return save(name, samples, time.perf_counter() - t, note)

    # B. file name candidates (English language tag, same voice F2)
    for n, t in B:
        raw(n, t, "en")
    # C. "Microsoft Word" candidates
    for n, t, lg in C:
        raw(n, t, lg)
    # D / E. the new pipeline
    pipe("d_s3_new", S3, note=str([(r.lang, r.text) for r in prepare(S3)]))
    for sid, text in (("e_s2_numbers", S2), ("e_s4_literary", S4), ("e_s5_instructions", S5),
                      ("e_s6_egyptian", S6), ("e_s7_long", S7), ("e_s8_english", S8),
                      ("e_s9_acronyms_keys", S9), ("e_s10_web", S10)):
        pipe(sid, text, note=str([(r.lang, r.text) for r in prepare(text)]))
    # F. speed against quality
    for steps in (4, 6, 8):
        pipe(f"f_s1_steps{steps}", S1, steps=steps)
    for sp in (1.0, 1.3, 1.6):
        pipe(f"f_s1_speed{sp}", S1, steps=6, speed=sp)

    json.dump(log, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    write_html(log)
    print("\nOpen", os.path.join(OUT, "index.html"))
    return 0


def write_html(log) -> None:
    by = {r["name"]: r for r in log}

    def p(name, caption, extra=""):
        r = by[name]
        note = f"<br><small>{html.escape(extra)}</small>" if extra else ""
        return (f"<div class=v><b>{html.escape(caption)}</b><br>"
                f"<audio controls preload=none src='{r['file']}'></audio>"
                f"<br><small>RTF {r['rtf']}</small>{note}</div>")

    def ref(path, caption):
        return (f"<div class=v><b>{html.escape(caption)}</b><br>"
                f"<audio controls preload=none src='{html.escape(path)}'></audio></div>")

    sections = []
    sections.append("<section><h3>A. The sentence that went wrong (round 2, voice F2)</h3>"
                    f"<p dir=rtl class=t>{html.escape(S3)}</p>"
                    + ref("../voices2/s3__F2__whole_ar.wav", "2a: whole sentence as Arabic")
                    + ref("../voices2/s3__F2__split.wav", "2b: English words as English")
                    + p("c1_as_is", "just “Microsoft Word” (English)")
                    + p("b1_as_is", "just “Homework.docx.” (English)")
                    + "<p>Which of these has the doubled “s”, and on which word?</p></section>")
    sections.append("<section><h3>B. How should the file name be spoken?</h3><p dir=ltr class=t>Homework.docx</p>"
                    + "".join(p(n, t) for n, t in B) + "</section>")
    sections.append("<section><h3>C. How should “Microsoft Word” be spoken?</h3>"
                    + "".join(p(n, f"{t}  [{lg}]") for n, t, lg in C) + "</section>")
    sections.append("<section><h3>D. The new pipeline on the same sentence</h3>"
                    f"<p dir=rtl class=t>{html.escape(S3)}</p>"
                    + p("d_s3_new", "F2, pronunciation layer, English as English", by['d_s3_new']['note']) + "</section>")
    es = [("e_s2_numbers", "Numbers", S2), ("e_s4_literary", "Literary", S4),
          ("e_s5_instructions", "Instructions", S5), ("e_s6_egyptian", "Egyptian greeting", S6),
          ("e_s7_long", "Long paragraph", S7), ("e_s8_english", "English", S8),
          ("e_s9_acronyms_keys", "Acronyms and keys", S9), ("e_s10_web", "Web address and e-mail", S10)]
    sections.append("<section><h3>E. Other sentences through the new pipeline</h3>"
                    + "".join(f"<div class=blk><p dir={'ltr' if n == 'e_s8_english' else 'rtl'} class=t>{html.escape(t)}</p>"
                              + p(n, c, by[n]['note']) + "</div>" for n, c, t in es) + "</section>")
    sections.append("<section><h3>F. Speed against quality (an old laptop may need fewer steps)</h3>"
                    f"<p dir=rtl class=t>{html.escape(S1)}</p>"
                    + "".join(p(f"f_s1_steps{s}", f"{s} steps (8 is the default)") for s in (4, 6, 8))
                    + "<br>" + "".join(p(f"f_s1_speed{sp}", f"speed {sp}x (6 steps)") for sp in (1.0, 1.3, 1.6))
                    + "</section>")
    page = f"""<!doctype html><meta charset=utf-8><title>EchoTrace voice clinic: F2</title>
<style>body{{font:16px system-ui;max-width:980px;margin:2em auto;padding:0 1em}}
.t{{font-size:1.3em;line-height:1.7;margin:.3em 0}} .v{{display:inline-block;margin:.4em 1em .4em 0;vertical-align:top;max-width:360px}}
.blk{{margin:.8em 0}} section{{border-top:1px solid #ccc;padding:.6em 0}} small{{color:#555;word-break:break-word}}</style>
<h1>EchoTrace voice clinic: Supertonic F2</h1>
<p>Please tell us: (A) where the doubled “s” is, (B and C) which spoken form sounds right, (E) anything misread,
and (F) the lowest number of steps that still sounds good to you.</p>
{''.join(sections)}"""
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(page)


if __name__ == "__main__":
    sys.exit(main())

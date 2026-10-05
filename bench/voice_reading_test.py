#!/usr/bin/env python3
"""Reading test: textbook-style text through the whole pipeline (voice F2).

Run in the TTS environment from the project root:

    SUPERTONIC_CACHE_DIR="$PWD/bench/data/supertonic3" \
      bench/.venv-tts/bin/python bench/voice_reading_test.py

Writes results/voices5/index.html. Each group is ONE audio file (half a second between sentences) with the
sentence being spoken highlighted, so it reads like the real player: numbers, dates and money, math, titles and
abbreviations, English words inside Arabic, Egyptian speech, then whole paragraphs split by the sentence
splitter, then the same sentences at four reading speeds.
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

from echotrace.worker.sentences import split_sentences                  # noqa: E402
from echotrace.worker.speech_text import prepare                        # noqa: E402
from echotrace.worker.tts_supertonic import SAMPLE_RATE, SupertonicVoice  # noqa: E402

OUT = os.path.join(HERE, "results", "voices5")
PAUSE = 0.5                                      # seconds between sentences

NUMBERS = [
    ("A1", "بلغ عدد سكان مصر في عام 2023 نحو 105 ملايين نسمة."),
    ("A2", "تبعد القاهرة عن الإسكندرية حوالي 220 كم، وتستغرق الرحلة ساعتين ونصف."),
    ("A3", "ارتفعت درجة الحرارة إلى 38.5 درجة مئوية يوم 15 يوليو 2024."),
    ("A4", "حصل الطالب على 85% من الدرجات، وجاء ترتيبه الثالث على الفصل."),
    ("A5", "اشترى أحمد كتابين بسعر 45 جنيهاً للكتاب الواحد، فدفع 90 جنيهاً."),
    ("A6", "تبدأ المحاضرة في الساعة ٩:٣٠ صباحاً في القاعة رقم ١٢."),
    ("A7", "للتواصل مع المدرسة اتصل على الرقم 0123456789."),
    ("A8", "إذا كان 2 + 3 = 5 فإن 5 − 3 = 2، وإذا كان 4 × 3 = 12 فإن 12 ÷ 3 = 4."),
    ("A9", "The temperature reached 38.5 degrees on July 15, 2024, and 85% of students passed."),
    ("A10", "The exam is on 3/5/2025 at 9:30 a.m. in room 12."),
]
TITLES = [
    ("B1", "قال د. أحمد حسن إن الدراسة أُجريت في جامعة عين شمس."),
    ("B2", "ولد نجيب محفوظ عام 1911م وتوفي عام 2006م."),
    ("B3", "راجع الصفحة 45 من الفصل الثالث، ثم أجب عن الأسئلة 1 و2 و3."),
    ("B4", "التمثيل الضوئي (Photosynthesis) عملية تحدث في الأوراق الخضراء."),
    ("B5", "يتكون الماء من ذرتين من الهيدروجين وذرة من الأكسجين، ورمزه H2O."),
    ("B6", "اسم الملف report.pdf وحجمه 2.5 MB، وقد أرسلته إلى info@school.edu.eg."),
    ("B7", "افتح الإعدادات ثم اختر Accessibility ثم فعّل خيار Narrator."),
    ("B8", "Dr. Smith said that the lab opens at 9 a.m. on Sunday, e.g. for new students."),
]
EGYPTIAN = [
    ("C1", "إزيك يا أستاذ؟ عامل إيه النهارده؟ ماتنساش الواجب بتاع بكرة."),
    ("C2", "هو الدرس ده هيبدأ امتى؟ وأنا مش فاهم الجزء بتاع المعادلات."),
]
WATER = ("دورة الماء في الطبيعة هي الحركة المستمرة للماء بين سطح الأرض والغلاف الجوي. تبدأ الدورة عندما تسخن الشمس "
         "مياه البحار والأنهار فيتبخر جزء منها. يصعد بخار الماء إلى الأعلى، وهناك يبرد ويتكاثف مكوناً السحب. "
         "وعندما تثقل السحب بقطرات الماء تسقط على الأرض على هيئة مطر أو ثلج أو برد. يتجمع جزء من هذه المياه في "
         "الأنهار والبحيرات، ويتسرب جزء آخر إلى باطن الأرض ليكوّن المياه الجوفية. وهكذا تتكرر الدورة من جديد دون توقف.")
LAB = ("Welcome to the computer lab. Please save your work often, because the power may go out. To save a file, press "
       "Control and S. Your teacher will show you how to open Microsoft Word. If you need help, raise your hand and wait.")
MIXED = ("لحفظ الملف اضغط Ctrl+S ثم اكتب اسمه، مثل Homework.docx، واختر المجلد Documents. بعد ذلك أغلق برنامج Word "
         "واضغط على زر Shutdown عند الانتهاء. تذكر أن تحفظ عملك كل خمس دقائق، فقد ينقطع التيار الكهربائي في أي لحظة.")
SPEEDS = (1.05, 1.3, 1.6, 2.0)

GROUPS = [
    ("numbers", "Numbers, dates, money, math", "rtl", NUMBERS),
    ("titles", "Titles, abbreviations, English words inside Arabic", "rtl", TITLES),
    ("egyptian", "Egyptian speech", "rtl", EGYPTIAN),
]
PARAGRAPHS = [
    ("water", "Paragraph: the water cycle (Arabic)", "rtl", WATER, "P1"),
    ("lab", "Paragraph: the computer lab (English)", "ltr", LAB, "P2"),
    ("mixed", "Paragraph: saving a file (Arabic with English words)", "rtl", MIXED, "P3"),
]


def render(voice: SupertonicVoice, sentences, speed=None):
    """One audio array for a list of sentences, with start/end times for each."""
    pieces, spans, t = [], [], 0.0
    gap = np.zeros(int(PAUSE * SAMPLE_RATE), np.float32)
    took = 0.0
    for sid, text in sentences:
        t0 = time.perf_counter()
        x = voice.synthesize(text, speed=speed)
        took += time.perf_counter() - t0
        start = t
        pieces.append(x)
        t += len(x) / SAMPLE_RATE
        spans.append(dict(id=sid, text=text, start=round(start, 2), end=round(t, 2)))
        pieces.append(gap)
        t += PAUSE
    return np.concatenate(pieces[:-1]), spans, took


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    voice = SupertonicVoice(model_dir=os.path.join(HERE, "data", "supertonic3"), voice="F2", steps=8)
    voice.synthesize("مرحبا")                                              # warm-up
    log = []

    def save(key, title, direction, sentences, speed=None):
        x, spans, took = render(voice, sentences, speed)
        fn = f"{key}.wav"
        sf.write(os.path.join(OUT, fn), x, SAMPLE_RATE, subtype="PCM_16")
        audio = len(x) / SAMPLE_RATE
        log.append(dict(key=key, title=title, dir=direction, file=fn, audio_s=round(audio, 1), took_s=round(took, 1),
                        speed=speed, spans=spans,
                        spoken=[[(r.lang, r.text) for r in prepare(t)] for _, t in sentences]))
        print(f"{key:10s} {len(sentences):2d} sentences  audio {audio:5.1f}s  synthesis {took:5.1f}s  RTF {took / audio:.2f}",
              flush=True)

    for key, title, direction, items in GROUPS:
        save(key, title, direction, items)
    para_sentences = {}
    for key, title, direction, text, label in PARAGRAPHS:
        sents = [(f"{label}.{i + 1}", s.text) for i, s in enumerate(split_sentences(text))]
        para_sentences[key] = sents
        save(key, title, direction, sents)
    first_two = para_sentences["water"][:2]
    for sp in SPEEDS:
        save(f"speed_{sp}", f"Speed {sp}x", "rtl", first_two, speed=sp)

    json.dump(log, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    write_html(log)
    print("\nOpen", os.path.join(OUT, "index.html"))
    return 0


def write_html(log) -> None:
    def group(g):
        items = "".join(
            f"<li data-s='{s['start']}' data-e='{s['end'] + PAUSE}'><b>{html.escape(s['id'])}</b> {html.escape(s['text'])}</li>"
            for s in g["spans"])
        return (f"<section class=grp><h3>{html.escape(g['title'])}</h3>"
                f"<audio controls preload=none src='{g['file']}'></audio>"
                f"<small> {g['audio_s']} s</small><ol dir={g['dir']}>{items}</ol></section>")

    plain = [g for g in log if not g["key"].startswith("speed_")]
    speeds = [g for g in log if g["key"].startswith("speed_")]
    speed_html = "".join(
        f"<div class=v><b>{g['speed']}x</b><br><audio controls preload=none src='{g['file']}'></audio></div>" for g in speeds)
    first_two = "".join(f"<p dir=rtl class=t>{html.escape(s['text'])}</p>" for s in speeds[0]["spans"]) if speeds else ""
    page = f"""<!doctype html><meta charset=utf-8><title>EchoTrace reading test</title>
<style>body{{font:16px system-ui;max-width:900px;margin:2em auto;padding:0 1em}}
section{{border-top:1px solid #ccc;padding:.6em 0 1em}} ol{{list-style:none;padding:0;line-height:1.9;font-size:1.15em}}
li{{padding:.1em .4em;cursor:pointer;border-radius:4px}} li b{{color:#777;font-size:.8em;margin:0 .4em}} li.on{{background:#ffe9a8}}
small{{color:#555}} .v{{display:inline-block;margin:.4em 1em .4em 0}} .t{{font-size:1.15em;margin:.2em 0}}</style>
<h1>EchoTrace reading test (voice F2)</h1>
<p>Every group is one recording with half a second between sentences. The sentence being spoken is highlighted; click a
sentence to jump to it. Please tell us:</p>
<ul><li><b>Which sentences are misread</b> (by their number, for example A3 or B5) and what you heard instead.
Numbers, dates, money, % and the maths signs are the main thing to check.</li>
<li>Is half a second between sentences <b>too long, too short, or right</b>?</li>
<li>In the last section, <b>which speed is still easy to understand?</b></li>
<li>In the paragraphs: does it sound natural for long reading, or tiring?</li></ul>
{''.join(group(g) for g in plain)}
<section><h3>Reading speed</h3>{first_two}{speed_html}</section>
<script>
document.querySelectorAll('.grp').forEach(g => {{
  const a = g.querySelector('audio'), items = [...g.querySelectorAll('li')];
  items.forEach(li => li.addEventListener('click', () => {{ a.currentTime = +li.dataset.s; a.play(); }}));
  a.addEventListener('timeupdate', () => {{
    const t = a.currentTime;
    items.forEach(li => li.classList.toggle('on', t >= +li.dataset.s && t < +li.dataset.e));
  }});
}});
</script>"""
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(page)


if __name__ == "__main__":
    if "--html-only" in sys.argv:
        write_html(json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8")))
        sys.exit(0)
    sys.exit(main())

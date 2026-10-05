# EchoTrace Phase 0: OCR and voice evidence

Written 5 Oct 2026 on the dev Mac (Intel i3-8100, shared with other work). Every number
below is reproducible from `bench/` (commands at the end). Lower error is better.

## Bottom line

1. **No single offline OCR engine wins.** Tesseract 5 is better on clean, normal-size text;
   PP-OCRv5 is better on small, JPEG-compressed crops. Each fails differently, so the worker
   should run **Tesseract first and PP-OCRv5 as the fallback** (5.1% error on screen-like
   lines against 8.6% for Tesseract alone and 19.6% for PP-OCRv5 alone). A smarter choice
   between them (by confidence) has room to go lower: the per-image best of the two scores 4.1%
   and 10.3% on the two sets.
2. **mubsir's hybrid engine is not worth keeping for live reading.** On independent data it
   is no better than plain Tesseract, about 3 to 9 times slower, and far hungrier for memory.
   mubsir's *per-word splitting* idea, however, is confirmed by independent data and is essential.
3. **Four failure modes were found and each has a cheap fix** (section 4): light-on-dark text,
   word order in mixed Arabic/English lines, Tesseract silently returning nothing on ~6% of
   lines, and cutting words in the middle.
4. **Piper's Arabic voice is fast enough for any lab PC**: about 11 to 14 times faster than
   real time on this Mac, first audio in under a second. In the listening tests Supertonic 3
   (voice F2) sounded better, so Piper stays as the fast fallback (sections 6 and 6b).
5. **Vision Assistant Pro can be used without forking**, by pointing its OpenAI-compatible
   "custom" provider at our proxy. One security fix is needed in its source (section 7).

## 1. What was measured

| Set | What it is | Size |
|---|---|---|
| **KITAB sample** | Printed-text subsets of KITAB-Bench (MBZUAI, ACL 2025; independent): PATS and SynthesizeAR (line images, large and stylised fonts), ISI-PPT (tiny slide-screenshot lines, JPEG-blocky, some white-on-orange), ArabicOCR (short blocks), Hindawi (full book pages scanned at only ~75 dpi) | 185 images |
| **Screen set** | `make_screen_set.py`: tight single-line crops like a screenshot region. Arabic (real Wikipedia prose), mixed (Arabic with 1-2 English terms inside), English; light and dark themes; 11, 13, 16, 20 px; Tahoma and Arial. **Synthetic**: anti-aliased grayscale, no ClearType | 144 images |

Scoring: character error rate (CER) and word error rate (WER), micro-averaged (total edits over
total reference characters). "Plain" drops diacritics and tatweel and unifies digits, because none of
the offline engines emit tashkeel and our voice adds its own. Engines were compared on the images
they all ran on.

## 2. Independent check of mubsir's claims

| mubsir's claim | Independent result |
|---|---|
| PP-OCR collapses on long Arabic lines unless split into words first | **Confirmed, strongly.** On 111 line images without splitting, PP-OCRv4 scores 78.9% and PP-OCRv5 54.2%; with splitting 29.9% and 21.6%. On the widest lines (PATS) v5 goes 70.2% to 24.7%; on short lines (ISI-PPT) splitting makes no difference. This also explains why KITAB's authors got 0.77-0.79 for stock PaddleOCR. |
| The hybrid (DBNet lines + stacked Tesseract) is 4.8x better than either engine alone | **Not reproduced.** 75 images: hybrid 29.8% against plain Tesseract 12.6% and Tesseract with upscaling 9.4%; 1.74 s against 0.19 s per image; 1,145 MB against 79 MB. |
| (fairness check) the hybrid was built for 300 dpi book pages, not small crops | With upscaling on the 30 page and block images: hybrid 6.3% against Tesseract with upscaling 5.5%, but 4.2 s against 1.3 s per image and 922 MB against 121 MB. Level on accuracy, much heavier. Keep it only for batch book conversion, where its two-column logic matters (not tested here). |
| PP-OCR assumes dark text on a light background (not claimed, but true) | The splitter and recogniser both fail on light-on-dark lines: 91% error on dark screen lines until the image is inverted first. |

## 3. Results

### KITAB sample (CER %, plain; 185 images)

| engine | PATS | SynthesizeAR | ISI-PPT | ArabicOCR | Hindawi pages | average | s/image (median) | peak MB |
|---|---|---|---|---|---|---|---|---|
| PP-OCRv4, per-word split | 34.6 | 41.3 | 13.8 | 15.9 | 41.5 | 29.4 | 0.08 | 843 |
| **PP-OCRv5**, per-word split | 24.7 | 31.6 | 8.4 | 13.0 | 36.0 | 22.8 | 0.09 | 850 |
| PP-OCRv5 with its own newer detector | 24.7 | 31.6 | 8.4 | 54.4 | 34.9 | 30.8 | 0.10 | 858 |
| **Tesseract 5** `ara`, as is | 5.5 | 20.8 | 23.0 | 0.6 | 32.3 | 16.4 | 0.19 | 79 |
| Tesseract 5 `ara`, upscaled | 5.5 | 20.8 | 29.8 | 1.3 | 13.2 | 14.1 | 0.21 | 121 |

The PP-OCRv5 detector variant is clearly worse (keep the standard detector). The 843-858 MB
peaks are dominated by running the line detector on page-size images: the recogniser alone peaks near
120 MB, so this can be tuned down for small regions (Phase 2).

### Text height decides the winner (KITAB line images, CER %)

| line height | Tesseract | Tesseract upscaled | PP-OCRv5 |
|---|---|---|---|
| under 25 px | 24.2 | 37.9 | **10.3** |
| 25-34 px | 14.3 | 14.3 | **7.4** |
| 35-59 px | 7.9 | 7.6 | **1.6** |
| 60 px or more | **11.9** | 11.9 | 29.7 |

(First-version harness, before the fixes in section 4. Tesseract here is Arabic-only, single-line mode.)

### Screen set (CER %, plain; 144 images; corrected harness)

| engine | Arabic | English | Mixed | light | dark | **all** |
|---|---|---|---|---|---|---|
| Tesseract `ara+eng`, upscaled, polarity, single-line mode | 19.7 | 1.6 | 6.2 | 5.9 | 11.4 | **8.6** |
| same, raw-line mode (psm 13) | 30.6 | 1.6 | 17.2 | - | - | 15.8 |
| PP-OCRv5 per word, polarity, bidi order | 18.8 | 20.7 | 19.2 | 18.3 | 20.9 | 19.6 |
| PP-OCRv5 in chunks of up to 6:1 | 49.7 | 51.8 | 56.5 | 52.0 | 53.7 | 52.8 |
| PP-OCRv5 in chunks of up to 8:1 | 57.1 | 61.3 | 61.7 | 59.7 | 60.6 | 60.2 |
| PP-OCRv5 in chunks of up to 12:1 | 83.9 | 72.1 | 81.8 | 79.9 | 78.3 | 79.1 |

By font size (CER %):

| engine | 11 px | 13 px | 16 px | 20 px |
|---|---|---|---|---|
| Tesseract `ara+eng`, upscaled, polarity | 11.1 | 7.9 | 6.0 | 8.7 |
| PP-OCRv5 per word | 37.1 | 19.8 | 8.4 | **2.9** |

**Chunking several words together is much worse than splitting per word** (my own idea, tested and
rejected): the recogniser degrades fast with multi-word crops.

### Two engines together

| rule | screen set | KITAB lines |
|---|---|---|
| Tesseract alone (single-line mode) | 8.6 | 21.8 |
| PP-OCRv5 alone | 19.6 | 15.4 |
| Tesseract, retry in raw-line mode when empty | 6.0 | 17.7 |
| **Tesseract, PP-OCRv5 when empty or under half the length** | **5.1** | **16.8** |
| best of the two per image (upper bound, unknowable at run time) | 4.1 | 10.3 |

## 4. Failure modes found, each with its fix

| failure | evidence | fix |
|---|---|---|
| **Light text on dark background** | PP-OCRv5 91% error on dark lines; Tesseract also weaker | Invert when the (minority) ink is the bright class: right on all 144 screen images and the KITAB crops. Dark and light now score alike (20.9% / 18.3%) |
| **Word order in mixed or English lines** | English lines came out as "sublative. untall oh polaxis ..." (words recognised, order reversed) | Recognised pieces arrive in visual order. Put them back in logical order by the line's direction, keeping runs of English words in their own order (the approach NVDA uses for Windows OCR). Unit-tested |
| **Tesseract returns nothing** | Single-line mode (psm 7) returned empty on 7 of 144 screen lines and 9 of 111 KITAB lines, including clean black-on-white text. Silence is the worst failure for a blind listener | Raw-line mode (psm 13) recovered **16 of 16**; PP-OCRv5 as a second engine also covers it |
| **Cutting words** | Small mixed lines lost middle letters; chunking was worse still | Keep per-word splitting; it is accurate from 16 px up (8.4%, 2.9% at 20 px) but weak at 11 px (37%), where Tesseract is the better engine |

Also learned: upscaling helps low-resolution pages a lot (Hindawi 32% to 13%) but hurt the tiniest
lines (under 25 px: 24% to 38%); the right upscale factor per size still needs tuning on real screenshots.

## 5. Speed and memory (this Mac; an old laptop will be slower)

| | median s per image | peak memory |
|---|---|---|
| Tesseract (process start included) | 0.14-0.21 (lines), 0.4-1.7 (blocks, pages) | 67-146 MB (child process) |
| PP-OCRv5 per word, lines | 0.04-0.09 | ~120 MB |
| PP-OCRv5 with line detection, page-size input | 1.5-1.7 | ~850 MB (tunable) |
| mubsir hybrid | 1.7-4.2 | 920-1,145 MB |

Environment note: this Mac froze for roughly 15 minutes at a time, several times (one-off
single images took 925 s and 1,785 s; re-running them took 1.8 to 3.2 s). The stalls came from outside
the programs, so timings use medians. Lesson for the worker anyway: it needs a per-region watchdog.

## 6. Arabic voice (Piper `ar_JO-kareem`, this Mac)

| voice | median real-time factor | median first audio | slowest first audio |
|---|---|---|---|
| kareem low | 0.07 (14x faster than real time) | 0.64 s | 1.48 s |
| kareem medium | 0.09 (11x) | 0.82 s | 1.98 s |
| English lessac medium | 0.20 (5x) | 0.83 s | 0.83 s |

Arabic diacritization (libtashkeel) is built into Piper and on by default for Arabic voices.
Loading a voice takes about 1.5-2 s; four voices at once peaked at 1.5 GB, so measure one voice alone
before promising a memory figure. The listening page is built by `voice_bakeoff.py` (`results/voices/index.html`, generated, not committed). It has eight sentences
(textbook, numbers, mixed Arabic/English, Egyptian greeting, a long paragraph), each in low, medium, and
medium with diacritization switched off, plus a mixed sentence read by two voices. (Supertonic won this listening test; see the next subsection.)

### 6b. Supertonic 3, voice F2 (chosen by listening, 5 Oct 2026): the doubled "s" and the joins

Measured from spectrograms and a detector for a repeated "s"-like sound
(`voice_glitch_check.py`), because the tester cannot hear audio. It catches that one glitch only; whether
the voice sounds natural is decided by human ears.

| test | result |
|---|---|
| Whole mixed sentence sent under the **Arabic tag** (what a listener heard as "microso soft") | repeated sound in **14 of 190** draws (7%): on "Microsoft" in 9 of the 13 draws saved, on "docx" in 4 |
| "Microsoft Word" under the **English tag** | **0 of 500** draws |
| "Homework.docx." and "Homework dot doc x." under the English tag | 0 of 200 each; the final "x" sound was missing in 27 of 200 raw draws against 3 of 200 rewritten |
| The whole new pipeline (Arabic runs under `ar`, English runs under `en`, the pronunciation layer) | **0 of 100** draws of the same sentence |
| Duration of one phrase over 400 random draws | identical (1.46 s every time): only the acoustics are random |
| Silence the model adds to every clip | about 0.3 s at each end, so joining the pieces left **0.8-1.1 s of dead air** at every Arabic/English switch; after `trim_silence` the longest pause inside a sentence is 0.2-0.4 s (mostly commas) |

What follows from this: Latin text is never sent under the Arabic tag; every piece is trimmed before joining; the
noise is seeded from the text so a bad sentence can be reproduced, and a repeat of a sentence asks for another
draw (`variant`). Pitch measurements of how a phrase ends (comma against full stop) varied too much between
draws to choose by; a listener compared the two styles by ear and chose the comma style, which is now the default.

## 7. Vision Assistant Pro (the AI add-on), from reading its source

- Its `custom` provider speaks the **OpenAI chat protocol** with separate endpoint and model settings
  for chat, vision, OCR, speech-to-text and text-to-speech, and sends images as data URLs. So **our
  proxy can be an OpenAI-compatible gateway and the add-on works unmodified.**
- The add-on sets no token limit and uses temperature 0.7 for OCR. The proxy must therefore inject
  `max_tokens`, force temperature 0 for transcription, downscale images, cap per device, and choose the
  cheap model by alias (for example a model name `echo-ocr`).
- **Security fix needed:** its non-AI "fast" OCR engines send screenshot pixels to an undocumented
  Google endpoint (the one Chrome uses) and to a random third-party server. For student screens these
  must be removed from our copy, the only code change we expect to need.
- It bundles PyMuPDF inside the add-on, so shipping native libraries inside an NVDA add-on works.

### NVDA add-on store check (official add-on data, 297 add-ons, 5 Oct 2026)

- Present: Vision Assistant Pro (`VisionAssistant`, up to version 7.0.0), AIContentDescriber, several
  other AI helpers, KittenTTS (English neural voice), InflectMicroTTS, RHVoice, NativeSpeechGeneration.
- **Absent: any Piper/Sonata voice add-on and any Tesseract OCR add-on.** Sonata is only an engine; its
  one NVDA add-on is Welsh-only (techiaith, beta, voice about 77 MB downloaded on first use).
- So an Arabic neural voice inside NVDA has to be built by us, following that add-on's pattern. Without
  it, Arabic in NVDA means eSpeak NG (robotic) or Windows voices that need a language pack (admin).

## 8. Caveats

- The screen set is synthetic and clean; real ClearType screens, UI fonts and photographed pages will
  differ. **Real lab screenshots are the missing piece and could change the Tesseract versus PP-OCRv5 balance.**
- KITAB sample subsets are partly synthetic too; 185 images is enough to rank engines, not to quote
  to one decimal place.
- No cloud model (Gemini, Qwen) was scored yet: it needs an API key (see `.env.example`).
- Windows OCR was not scored: it needs a Windows PC with the Arabic OCR pack.
- Harness bugs found and fixed along the way (first runs discarded): word order always right-to-left,
  polarity checked after a white border was added.

## 9. Reproduce

```bash
python bench/fetch_data.py models && python bench/fetch_data.py kitab --rows 40
python bench/make_screen_set.py
python bench/ocr_bench.py run --engine tess_up           # one engine, own process
python bench/ocr_bench.py summary --common               # tables; add --tag screen3 / v3
python bench/breakdown.py --tag screen3 --by size        # by size, kind, theme, height
python bench/fallback_analysis.py --tag screen3 --primary tess_ara_eng_up_pol --secondary v2_v5_word
bench/.venv-tts/bin/python bench/voice_bakeoff.py
```

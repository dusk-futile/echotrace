# EchoTrace

A reading companion for **totally blind learners**. Press a key and what is on the screen, or in a PDF, is read
aloud in Arabic or English, with pause, "say that sentence again" and next line. Built for community labs with old
donated laptops: **offline first**, no administrator rights, with an optional cheap cloud AI for the hard cases.

> **Status: early.** The evidence phase is done and the voice pipeline is built and tested. The NVDA add-on and the
> lab bundle are not built yet, so nothing here can be handed to learners today.

## The idea

- **NVDA is the foundation**: it is open source and what the learners already use. EchoTrace becomes a small NVDA
  add-on (one-key commands: read the page left to right or right to left, read under the mouse, pause, next line,
  repeat the same sentence, stop) plus a lab bundle (portable NVDA, no admin rights), instead of a second program
  fighting NVDA for the keyboard and the speakers.
- **An engine worker** next to NVDA does the heavy parts (OCR, PDF reading, the voice) in its own process, so a
  crash there can never silence NVDA.
- **One natural voice for both languages.** [Supertonic 3](https://github.com/supertone-inc/supertonic) reads Arabic
  and English in the same voice, offline. English words inside Arabic text are read by the English side of that voice.
- **Optional AI** (Gemini-class, a few tokens per read) goes through a small proxy that holds the key; it is off
  by default.

## What exists today

| Part | Where | Notes |
|---|---|---|
| Pronunciation layer | `echotrace/worker/speech_text.py` | Cleans text, splits Arabic and English runs, spells out file names, acronyms, URLs, key combinations, chemistry formulas; rules live in a user-editable lexicon |
| Voice | `echotrace/worker/tts_supertonic.py` | Supertonic 3 wrapper: trims the silence the model adds at each end, seeds the model's randomness from the text (reproducible; `variant` gives another take), caches sentences |
| Sentence splitter | `echotrace/worker/sentences.py` | Arabic and English punctuation, abbreviations (`د. أحمد`, `Dr. Smith`), decimals, file names, headings, bullets; remembers lines for "next line" |
| Tests | `tests/` | 74 tests; the voice is tested against a stand-in for the model |
| Phase 0 evidence | `bench/` | OCR and voice bake-offs with the numbers in [`bench/RESULTS.md`](bench/RESULTS.md) |
| PC survey | `tools/EchoTrace-Check.bat` | Read-only check of a lab laptop (Windows version, RAM, CPU, NVDA, voices). **Untested on Windows so far** |

## What the evidence says (details in [`bench/RESULTS.md`](bench/RESULTS.md))

- **OCR:** no single offline engine wins. Tesseract 5 first, PP-OCRv5 as the fallback, gave 5.1% character error on
  synthetic screen text, against 8.6% (Tesseract alone) and 19.6% (PP-OCRv5 alone). Four failure modes were found,
  each with a cheap fix (light-on-dark text, mixed-direction order, Tesseract returning nothing, cutting words).
- **Voice:** Supertonic 3 (voice F2) runs about four times faster than real time on an Intel i3-8100 with about
  520 MB of memory; Piper's Arabic voice is the fast fallback. Sending English under the Arabic language tag gave a
  repeated "s" in 7% of renderings ("Microsoft" read as "microso soft"); through this pipeline 0 of 100.
- **NVDA ecosystem:** the official add-on store (297 add-ons, October 2026) has no Arabic neural voice, so this
  project packages its own.

## Run the tests

```bash
pip install -e ".[test]"
python -m pytest
```

## Try the voice

Needs the Supertonic 3 model (about 380 MB, downloaded from its own repository under its own licence) and a
Python environment with the `voice` extra (`pip install -e ".[voice]"`). The scripts in `bench/` build listening
pages:

```bash
SUPERTONIC_CACHE_DIR="$PWD/bench/data/supertonic3" python bench/voice_reading_test.py
```

`bench/voice_glitch_check.py` re-measures the repeated-sound glitch after any change to the voice.

The OCR bake-off in `bench/` reuses code, models and corpora from a checkout of
[mubsir](https://github.com/dusk-futile/mubsir); point `MUBSIR_ROOT` at it (see `bench/RESULTS.md`, "Reproduce").

## Roadmap

1. Phase 0, evidence: **done** except real lab screenshots, the lab PC survey and the Windows OCR and cloud-model rows.
2. Phase 1: portable NVDA with the natural voice as an NVDA speech synthesizer (no Arabic neural voice exists
   there yet) and a short spoken guide. Test with learners before building more.
3. Phase 2: the add-on (key layer, sentence player) and the OCR worker.
4. Phase 3: file reading from Explorer, the lab zip, the AI proxy.

## Licence

The code in this repository is MIT licensed (see `LICENSE`). Third-party parts keep their own licences: NVDA is
GPL-2, the Piper engine is GPL-3, PyMuPDF is AGPL-3, and the Supertonic 3 weights use the OpenRAIL-M licence (they are
not included here). A lab bundle that ships NVDA is distributed under the GPL.

## Credits

[NVDA](https://www.nvaccess.org/), [Supertonic](https://github.com/supertone-inc/supertonic),
[Piper](https://github.com/OHF-Voice/piper1-gpl), [Tesseract](https://github.com/tesseract-ocr/tesseract),
[PaddleOCR / PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR),
[KITAB-Bench](https://github.com/mbzuai-oryx/KITAB-Bench) (independent Arabic OCR benchmark), and
[Vision Assistant Pro](https://github.com/mahmoodhozhabri/VisionAssistantPro) (the pattern for the key layer).

## History

This repository began as the "Idea Universe" brainstorming prompt. That prompt is still in the git history
(commit `27e5ec1`).

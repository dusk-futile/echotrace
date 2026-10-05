"""The voice wrapper, with a stand-in for the model: language tags, seeding, caching, retries."""
import re
import types

import numpy as np
import pytest

from echotrace.worker.tts_supertonic import (SAMPLE_RATE, SupertonicVoice, seed_for, split_long,
                                              trim_silence)

S3 = "يستخدم الطلاب برنامج Microsoft Word لكتابة الواجب، ثم يحفظونه باسم Homework.docx."
RUN = 4410                                      # samples the stand-in returns per call (0.1 s)


class FakeEngine:
    """Stands in for supertonic.TTS: records every call and the random draw it saw."""

    def __init__(self, unreadable="", pad=0):
        self.calls = []
        self.unreadable = unreadable
        self.pad = pad                          # samples of silence the model leaves at each end
        ok = {chr(c) for c in range(32, 127)} | {chr(c) for c in range(0x0621, 0x064B)}
        self.model = types.SimpleNamespace(text_processor=types.SimpleNamespace(supported_character_set=ok))

    def get_voice_style(self, voice_name):
        return f"style:{voice_name}"

    def synthesize(self, text, voice_style, lang, total_steps, speed, silence_duration):
        if self.unreadable and self.unreadable in text:
            raise ValueError("unsupported character")
        self.calls.append(dict(text=text, lang=lang, style=voice_style, steps=total_steps, speed=speed,
                               silence=silence_duration, draw=float(np.random.rand())))
        clip = np.concatenate([np.zeros(self.pad), np.full(RUN, 0.1), np.zeros(self.pad)]).astype(np.float32)
        return clip[None, :], 0.1


@pytest.fixture
def engine():
    return FakeEngine()


def voice(engine, **kw):
    return SupertonicVoice(engine=engine, **kw)


# ----------------------------------------------------------------------------- language tags
def test_english_is_never_sent_with_the_arabic_tag(engine):
    sentences = [S3,
                 "افتح ملف PDF من برنامج Excel ثم اضغط Ctrl+S لحفظ الملف على USB.",
                 "اكتب العنوان www.school.edu.eg في متصفح Chrome ثم أرسل الملف إلى info@school.edu.eg.",
                 "Press the space bar, ثم اضغط Enter",
                 "الإصدار Windows 10 يدعم Wi-Fi"]
    v = voice(engine)
    for s in sentences:
        v.synthesize(s)
    for call in engine.calls:
        has_latin = re.search(r"[A-Za-z]", call["text"]) is not None
        has_arabic = re.search("[ء-ي]", call["text"]) is not None
        assert not (has_latin and has_arabic), call           # one language per call
        assert call["lang"] == ("en" if has_latin else "ar"), call


def test_the_sentence_that_went_wrong_is_read_as_four_runs(engine):
    voice(engine, continuation=False).synthesize(S3)
    assert [(c["lang"], c["text"]) for c in engine.calls] == [
        ("ar", "يستخدم الطلاب برنامج"),
        ("en", "Microsoft Word"),
        ("ar", "لكتابة الواجب، ثم يحفظونه باسم"),
        ("en", "Homework dot doc x."),
    ]


def test_one_voice_reads_both_languages(engine):
    voice(engine, voice="F2").synthesize(S3)
    assert {c["style"] for c in engine.calls} == {"style:F2"}


def test_runs_are_joined_with_a_short_gap(engine):
    out = voice(engine, gap_seconds=0.08).synthesize(S3)
    assert len(out) == 4 * RUN + 3 * int(0.08 * SAMPLE_RATE)
    assert out.dtype == np.float32


def test_silence_the_model_leaves_at_each_end_is_cut_before_joining():
    eng = FakeEngine(pad=int(0.3 * SAMPLE_RATE))              # about what the real model does
    out = voice(eng, gap_seconds=0.04).synthesize(S3)
    margin = int(0.04 * SAMPLE_RATE)
    assert len(out) == 4 * (RUN + 2 * margin) + 3 * int(0.04 * SAMPLE_RATE)
    boundary = 2 * margin + int(0.04 * SAMPLE_RATE)           # the dead air at each Arabic/English switch
    assert boundary / SAMPLE_RATE < 0.2


def test_continuation_ends_runs_that_are_not_last_with_a_comma(engine):
    voice(engine).synthesize(S3)                              # on by default: chosen by ear in a listening test
    assert [c["text"] for c in engine.calls] == [
        "يستخدم الطلاب برنامج،", "Microsoft Word,", "لكتابة الواجب، ثم يحفظونه باسم،", "Homework dot doc x."]


def test_continuation_can_be_switched_off_and_never_doubles_punctuation(engine):
    voice(engine, continuation=False).synthesize("اقرأ، ثم Word")
    assert [c["text"] for c in engine.calls] == ["اقرأ، ثم", "Word"]
    engine.calls.clear()
    voice(engine).synthesize("اقرأ، ثم Word. ثم اكتب")
    assert [c["text"] for c in engine.calls] == ["اقرأ، ثم،", "Word.", "ثم اكتب"]


def test_settings_reach_the_model(engine):
    voice(engine, steps=5, speed=1.2).synthesize("مرحبا")
    assert (engine.calls[0]["steps"], engine.calls[0]["speed"], engine.calls[0]["silence"]) == (5, 1.2, 0.0)


# ----------------------------------------------------------------------------- the random draw
def test_a_sentence_sounds_the_same_every_time():
    a, b = FakeEngine(), FakeEngine()
    voice(a).synthesize(S3)
    voice(b).synthesize(S3)
    assert [c["draw"] for c in a.calls] == [c["draw"] for c in b.calls]


def test_variant_picks_another_draw():
    a, b = FakeEngine(), FakeEngine()
    voice(a).synthesize(S3)
    voice(b).synthesize(S3, variant=1)
    assert [c["draw"] for c in a.calls] != [c["draw"] for c in b.calls]


def test_each_run_has_its_own_draw(engine):
    voice(engine).synthesize(S3)
    assert len({c["draw"] for c in engine.calls}) == len(engine.calls)


def test_the_processs_random_state_is_left_alone(engine):
    np.random.seed(123)
    expected = np.random.rand(3)
    np.random.seed(123)
    voice(engine).synthesize(S3)
    assert np.array_equal(np.random.rand(3), expected)


def test_seed_is_stable_and_fits_numpy():
    assert seed_for("ar", "نص", "F2", 8, 1.05, 0) == seed_for("ar", "نص", "F2", 8, 1.05, 0)
    assert seed_for("ar", "نص", "F2", 8, 1.05, 0) != seed_for("ar", "نص", "F2", 8, 1.05, 1)
    assert 0 <= seed_for("anything") < 2 ** 32
    assert seed_for("en", "Microsoft Word", "F2", 8, 1.05, 0) == 1388525846    # pinned: changing it changes every draw


# ----------------------------------------------------------------------------- cache
def test_a_repeat_is_served_from_the_cache(engine):
    v = voice(engine)
    first = v.synthesize(S3)
    n = len(engine.calls)
    assert v.synthesize(S3) is first
    assert len(engine.calls) == n


def test_a_variant_or_another_speed_is_a_new_cache_entry(engine):
    v = voice(engine)
    v.synthesize("مرحبا")
    n = len(engine.calls)
    v.synthesize("مرحبا", variant=1)
    v.synthesize("مرحبا", speed=1.3)
    assert len(engine.calls) == n + 2


def test_the_cache_is_bounded(engine):
    v = voice(engine, cache_items=2)
    for word in ("واحد", "اثنان", "ثلاثة"):
        v.synthesize(word)
    n = len(engine.calls)
    v.synthesize("ثلاثة")
    assert len(engine.calls) == n                 # newest still cached
    v.synthesize("واحد")
    assert len(engine.calls) == n + 1             # oldest was dropped


# ----------------------------------------------------------------------------- robustness
def test_an_unreadable_character_is_dropped_and_the_rest_is_spoken():
    eng = FakeEngine(unreadable="☃")
    out = voice(eng).synthesize("مرحبا ☃ بالعالم")
    assert len(out) == RUN
    assert eng.calls[-1]["text"] == "مرحبا  بالعالم"


def test_a_chunk_of_only_unreadable_characters_is_skipped():
    eng = FakeEngine(unreadable="☃")
    assert len(voice(eng).synthesize("☃")) == 0


def test_empty_and_punctuation_only_text_gives_no_audio(engine):
    v = voice(engine)
    assert len(v.synthesize("")) == 0
    assert len(v.synthesize("... !!")) == 0
    assert engine.calls == []


def test_a_long_run_is_split_at_natural_breaks(engine):
    long_text = "، ".join(["هذه جملة قصيرة من الكتاب"] * 30)
    voice(engine).synthesize(long_text)
    assert len(engine.calls) > 1
    assert all(len(c["text"]) <= 281 for c in engine.calls)


def test_split_long_keeps_every_word():
    text = " ".join(f"كلمة{i}" for i in range(200))
    assert " ".join(split_long(text)).split() == text.split()


def test_int16_output_is_clipped_and_scaled(engine):
    out = voice(engine).synthesize_int16("مرحبا")
    assert out.dtype == np.int16
    assert out.max() == int(0.1 * 32767)


# ----------------------------------------------------------------------------- trim_silence
def _tone(seconds, amp=1.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _noise(seconds, amp):
    return (amp * np.random.RandomState(0).randn(int(seconds * SAMPLE_RATE))).astype(np.float32)


def test_trim_cuts_both_ends_and_keeps_a_margin():
    x = np.concatenate([np.zeros(int(0.3 * SAMPLE_RATE)), _tone(0.5), np.zeros(int(0.3 * SAMPLE_RATE))]).astype(np.float32)
    seconds = len(trim_silence(x)) / SAMPLE_RATE
    assert 0.5 + 0.06 < seconds < 0.5 + 0.12


def test_trim_keeps_a_soft_consonant_before_the_vowel():
    soft = _noise(0.1, 0.004)                                  # about -48 dB: an "s" or "f" at the start of a word
    x = np.concatenate([np.zeros(int(0.3 * SAMPLE_RATE)), soft, _tone(0.4), np.zeros(int(0.3 * SAMPLE_RATE))]).astype(np.float32)
    assert len(trim_silence(x)) / SAMPLE_RATE > 0.5


def test_trim_leaves_silence_and_tiny_clips_alone():
    silent = np.zeros(SAMPLE_RATE, np.float32)
    assert trim_silence(silent) is silent
    tiny = np.ones(100, np.float32)
    assert trim_silence(tiny) is tiny


def test_trim_is_a_noop_on_audio_with_no_padding():
    x = _tone(0.5)
    assert len(trim_silence(x)) == len(x)

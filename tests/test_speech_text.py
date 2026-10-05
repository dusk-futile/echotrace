"""Pronunciation layer: cleaning, Arabic/English run splitting, token rewrites, user lexicon."""
import json

from echotrace.worker.speech_text import (Lexicon, Run, clean, load_lexicon, prepare,
                                           rewrite_arabic, rewrite_latin, split_runs)

S3 = "يستخدم الطلاب برنامج Microsoft Word لكتابة الواجب، ثم يحفظونه باسم Homework.docx."


# ----------------------------------------------------------------------------- clean
def test_clean_fixes_characters_the_voice_cannot_read():
    assert clean("نعم؛ ٱلحمد") == "نعم، الحمد"           # Arabic semicolon, alef wasla
    assert clean("كـتـاب") == "كتاب"                      # tatweel
    assert clean("a‏b‪c") == "abc"              # bidi marks
    assert clean("  two   spaces\nand\tnewline ") == "two spaces and newline"


def test_clean_drops_list_marks():
    assert clean("• أول بند") == "أول بند"
    assert clean("one ● two ▪ three") == "one two three"


def test_clean_keeps_diacritics():
    assert clean("التَّمْثِيلُ") == "التَّمْثِيلُ"


def test_clean_repairs_presentation_forms():
    assert clean("ﻻ") == "لا"                         # lam-alef ligature from a PDF/OCR


# ----------------------------------------------------------------------------- runs
def test_split_runs_alternates_languages():
    runs = split_runs(S3)
    assert [r.lang for r in runs] == ["ar", "en", "ar", "en"]
    assert runs[1].text.strip() == "Microsoft Word"
    assert runs[3].text == "Homework.docx."


def test_digits_stay_with_the_run_before_them():
    runs = split_runs("عام 1908 ثم Windows 10 هو")
    assert [(r.lang, r.text.strip()) for r in runs] == [
        ("ar", "عام 1908 ثم"), ("en", "Windows 10"), ("ar", "هو")]


def test_digits_only_text_is_still_spoken():
    assert split_runs("1908") == [Run("ar", "1908")]
    assert prepare("1908") == [Run("ar", "1908")]
    assert prepare("١٩٠٨") == [Run("ar", "١٩٠٨")]


def test_punctuation_only_input_produces_nothing():
    assert prepare("...") == []
    assert prepare("") == []


def test_arabic_comma_after_english_word_becomes_a_latin_comma():
    runs = prepare("استخدم Word، ثم Excel")
    assert [r.lang for r in runs] == ["ar", "en", "ar", "en"]
    assert runs[1].text == "Word,"


# ----------------------------------------------------------------------------- rewrites
def test_file_extension_is_spoken_not_guessed():
    lex = Lexicon()
    assert rewrite_latin("Homework.docx.", lex) == "Homework dot doc x."
    assert rewrite_latin("report.pdf", lex) == "report dot P D F"
    assert rewrite_latin("my_file-2.xlsx", lex) == "my_file-2 dot X L S X"


def test_unknown_extension_is_left_alone():
    assert rewrite_latin("notes.abc", Lexicon()) == "notes.abc"


def test_acronyms_are_spelled_but_word_acronyms_are_not():
    lex = Lexicon()
    assert rewrite_latin("Open a PDF in Word", lex) == "Open a P D F in Word"
    assert rewrite_latin("NASA and RAM", lex) == "NASA and RAM"
    assert rewrite_latin("a CPU and an MP3", lex) == "a C P U and an M P 3"


def test_all_caps_heading_is_not_spelled_out():
    assert rewrite_latin("WELCOME TO SCHOOL", Lexicon()) == "WELCOME TO SCHOOL"
    # ...but known acronyms still are, even inside a heading
    assert rewrite_latin("USB AND PDF", Lexicon()) == "U S B AND P D F"


def test_urls_and_emails():
    out = rewrite_latin("Write to info@school.edu.eg or visit www.school.edu.eg", Lexicon())
    assert out == "Write to info at school dot edu dot eg or visit w w w school dot edu dot eg"


def test_key_combinations():
    lex = Lexicon()
    assert rewrite_latin("Press Ctrl+S to save", lex) == "Press control plus S to save"
    assert rewrite_latin("Ctrl+Shift+Esc", lex) == "control plus Shift plus escape"


def test_brand_names_and_camel_case():
    lex = Lexicon()
    assert rewrite_latin("WhatsApp and YouTube", lex) == "Whats App and You Tube"
    assert rewrite_latin("SharePoint", lex) == "Share Point"
    assert rewrite_latin("McDonald", lex) == "McDonald"           # short parts: leave it


def test_symbols():
    assert rewrite_latin("R&D 50%", Lexicon()) == "R and D 50 percent"
    assert rewrite_arabic("نسبة 50% من") == "نسبة 50 في المئة من"


# ----------------------------------------------------------------------------- whole pipeline
def test_prepare_the_sentence_that_went_wrong():
    runs = prepare(S3)
    assert [(r.lang, r.text) for r in runs] == [
        ("ar", "يستخدم الطلاب برنامج"),
        ("en", "Microsoft Word"),
        ("ar", "لكتابة الواجب، ثم يحفظونه باسم"),
        ("en", "Homework dot doc x."),
    ]


# ----------------------------------------------------------------------------- user lexicon
def test_user_lexicon_overrides_defaults():
    lex = Lexicon().merged({"extensions": {"DOCX": "dock ex"}, "words": {"Microsoft": "Micro soft"}})
    assert rewrite_latin("Microsoft Homework.docx", lex) == "Micro soft Homework dot dock ex"


def test_user_can_add_a_word_acronym_and_a_spelled_acronym():
    lex = Lexicon().merged({"word_acronyms": ["radar"], "spelled": ["efe"]})
    assert rewrite_latin("RADAR", lex) == "RADAR"
    assert rewrite_latin("EFE", lex) == "E F E"


def test_broken_user_file_never_stops_speech(tmp_path):
    bad = tmp_path / "pronunciations.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    lex = load_lexicon(str(bad))
    assert rewrite_latin("report.pdf", lex) == "report dot P D F"
    assert load_lexicon(str(tmp_path / "missing.json")).extensions["docx"] == "doc x"


def test_good_user_file_is_merged(tmp_path):
    f = tmp_path / "pronunciations.json"
    f.write_text(json.dumps({"words": {"Resala": "Re sala"}}), encoding="utf-8")
    assert rewrite_latin("Resala", load_lexicon(str(f))) == "Re sala"


# ----------------------------------------------------------------------------- brackets, formulas
def test_an_opening_bracket_goes_with_the_word_after_it():
    runs = split_runs("التمثيل الضوئي (Photosynthesis) عملية")
    assert [(r.lang, r.text.strip()) for r in runs] == [
        ("ar", "التمثيل الضوئي"), ("en", "(Photosynthesis)"), ("ar", "عملية")]
    assert [(r.lang, r.text) for r in prepare("التمثيل الضوئي (Photosynthesis) عملية")] == [
        ("ar", "التمثيل الضوئي"), ("en", "(Photosynthesis)"), ("ar", "عملية")]


def test_an_opening_quote_goes_with_the_word_after_it_but_a_closing_one_stays():
    runs = split_runs('قال "Hello" ثم رحل')
    assert [(r.lang, r.text.strip()) for r in runs] == [("ar", "قال"), ("en", '"Hello"'), ("ar", "ثم رحل")]


def test_chemistry_formulas_and_paper_sizes_are_spelled():
    lex = Lexicon()
    assert rewrite_latin("H2O", lex) == "H 2 O"
    assert rewrite_latin("CO2 and O2", lex) == "C O 2 and O 2"
    assert rewrite_latin("H2SO4", lex) == "H 2 S O 4"
    assert rewrite_latin("NaCl and HCl", lex) == "N a C l and H C l"
    assert rewrite_latin("Use A4 paper", lex) == "Use A 4 paper"


def test_ordinary_words_with_digits_are_left_alone():
    lex = Lexicon()
    assert rewrite_latin("Windows10 and iPhone15", lex) == "Windows10 and iPhone15"
    assert rewrite_latin("COVID19", lex) == "COVID19"

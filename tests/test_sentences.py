"""Sentence splitter: punctuation, abbreviations, line joining, headings, long sentences."""
from echotrace.worker.sentences import Sentence, split_sentences


def texts(source, **kw):
    return [s.text for s in split_sentences(source, **kw)]


# ----------------------------------------------------------------------------- basics
def test_english_sentences():
    assert texts("Hello world. How are you? Fine!") == ["Hello world.", "How are you?", "Fine!"]


def test_arabic_sentences_with_the_arabic_question_mark_and_comma():
    assert texts("ما اسمك؟ اسمي أحمد، وأنا طالب. أحب القراءة.") == [
        "ما اسمك؟", "اسمي أحمد، وأنا طالب.", "أحب القراءة."]


def test_empty_and_symbol_only_input():
    assert split_sentences("") == []
    assert split_sentences(["", "  ", "..."]) == []
    assert split_sentences(["•"]) == []


def test_digits_only_line_is_kept():
    assert texts(["45"]) == ["45"]


def test_a_string_and_its_lines_give_the_same_result():
    text = "First line\n\nSecond line."
    assert split_sentences(text) == split_sentences(text.splitlines())


# ----------------------------------------------------------------------------- where not to cut
def test_decimals_and_versions_are_not_cut():
    assert texts("The value is 3.14. Version 2.0.1 is out.") == ["The value is 3.14.", "Version 2.0.1 is out."]
    assert texts("بلغت الحرارة 38.5 درجة. ثم انخفضت.") == ["بلغت الحرارة 38.5 درجة.", "ثم انخفضت."]


def test_english_titles_and_abbreviations():
    assert texts("Dr. Smith met Mr. Jones at 9 a.m. on Monday. They talked.") == [
        "Dr. Smith met Mr. Jones at 9 a.m. on Monday.", "They talked."]
    assert texts("Use a tool, e.g. a hammer. Then rest.") == ["Use a tool, e.g. a hammer.", "Then rest."]
    assert texts("See Fig. 3 and No. 5. Done.") == ["See Fig. 3 and No. 5.", "Done."]


def test_a_sentence_that_ends_with_pm_still_ends():
    assert texts("The lab closes at 5 p.m. The teacher leaves.") == ["The lab closes at 5 p.m.", "The teacher leaves."]


def test_arabic_titles_and_initials():
    assert texts("قال د. أحمد حسن إن الدراسة نجحت. ثم شكر الجميع.") == [
        "قال د. أحمد حسن إن الدراسة نجحت.", "ثم شكر الجميع."]
    assert texts("حضر أ.د. سارة علي المؤتمر. وألقت كلمة.") == ["حضر أ.د. سارة علي المؤتمر.", "وألقت كلمة."]


def test_a_year_followed_by_the_ad_letter_still_ends_the_sentence():
    assert texts("ولد عام 1911م. وتوفي عام 2006م.") == ["ولد عام 1911م.", "وتوفي عام 2006م."]


def test_file_names_and_web_addresses():
    assert texts("Save it as Homework.docx. Then close Word.") == ["Save it as Homework.docx.", "Then close Word."]
    assert texts("Visit www.school.edu.eg. It has lessons.") == ["Visit www.school.edu.eg.", "It has lessons."]


def test_initials_stay_with_the_name_but_a_vitamin_letter_ends_the_sentence():
    assert texts("Books by J. K. Rowling are popular. Many read them.") == [
        "Books by J. K. Rowling are popular.", "Many read them."]
    assert texts("It contains Vitamin C. It helps the body.") == ["It contains Vitamin C.", "It helps the body."]


def test_ellipsis_and_quotes():
    assert texts("Wait... what? Yes.") == ["Wait... what?", "Yes."]
    assert texts('He said "Stop!" and left. Then it rained.') == ['He said "Stop!" and left.', "Then it rained."]
    assert texts('"Stop!" He left.') == ['"Stop!"', "He left."]
    assert texts("Really?! Yes.") == ["Really?!", "Yes."]


def test_numbered_items_keep_their_number():
    assert texts(["1. Introduction", "The course begins."]) == ["1. Introduction", "The course begins."]
    assert texts(["١. المقدمة", "تبدأ الدراسة."]) == ["١. المقدمة", "تبدأ الدراسة."]


# ----------------------------------------------------------------------------- lines
def test_lines_of_one_paragraph_are_joined_and_remember_their_lines():
    got = split_sentences(["The plant takes in water from the soil and", "light from the sun to make food", "for itself. It grows."])
    assert got == [Sentence("The plant takes in water from the soil and light from the sun to make food for itself.", 0, 2),
                   Sentence("It grows.", 2, 2)]


def test_a_short_heading_is_its_own_sentence():
    got = split_sentences(["Chapter 1", "The plant takes in water from the soil and light from the sun."])
    assert [(s.text, s.first_line, s.last_line) for s in got] == [
        ("Chapter 1", 0, 0), ("The plant takes in water from the soil and light from the sun.", 1, 1)]
    assert texts(["الفصل الأول", "يتناول هذا الفصل دورة الماء في الطبيعة بالتفصيل."]) == [
        "الفصل الأول", "يتناول هذا الفصل دورة الماء في الطبيعة بالتفصيل."]


def test_bullets_are_separate_sentences():
    assert texts(["• First item to read here", "• Second item to read here"]) == [
        "• First item to read here", "• Second item to read here"]
    assert texts(["- أول بند في القائمة هنا", "- ثاني بند في القائمة هنا"]) == [
        "- أول بند في القائمة هنا", "- ثاني بند في القائمة هنا"]


def test_a_blank_line_ends_a_paragraph_and_indices_count_blank_lines():
    got = split_sentences(["The first paragraph has no full stop", "", "The second one starts here and goes on"])
    assert [(s.first_line, s.last_line) for s in got] == [(0, 0), (2, 2)]


def test_forced_breaks_from_the_layout():
    lines = ["The first paragraph is long enough to be one", "line of a page and has no stop", "The second paragraph begins here"]
    assert len(split_sentences(lines)) == 1
    assert len(split_sentences(lines, breaks=[1])) == 2


def test_hyphenated_english_words_are_mended():
    assert texts(["The compu-", "ter is on in the lab today."]) == ["The computer is on in the lab today."]


# ----------------------------------------------------------------------------- long sentences
def test_a_long_sentence_is_cut_at_a_comma_without_losing_words():
    clause = "هذه جملة طويلة من كتاب مدرسي عن دورة الماء في الطبيعة"
    long_sentence = "، ".join([clause] * 8) + "."
    got = texts(long_sentence, max_chars=150)
    assert len(got) > 1
    assert all(len(t) <= 150 for t in got)
    assert all(t.endswith("،") for t in got[:-1])
    assert " ".join(got).split() == long_sentence.split()


def test_a_long_sentence_without_commas_is_cut_at_a_space():
    sentence = " ".join(f"word{i}" for i in range(80)) + "."
    got = texts(sentence, max_chars=100)
    assert all(len(t) <= 100 for t in got)
    assert " ".join(got).split() == sentence.split()

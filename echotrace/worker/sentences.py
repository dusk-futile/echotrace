"""Split text into the sentences the player reads, repeats and steps through.

The unit a learner moves by is the sentence: "rewind" says the same sentence again, "next" goes to the next
one. Text arrives as lines (OCR of a window, a PDF page) or as running text. This module

  1. joins the lines of a paragraph (and mends English words hyphenated at a line end),
  2. keeps a short heading, a bullet or a numbered item as a sentence of its own,
  3. cuts at . ! ? ؟ … but not inside "3.14", "Dr. Smith", "9 a.m. on", "د. أحمد", "Homework.docx",
  4. cuts a very long sentence at a comma so a rewind never repeats half a minute of speech.

Each sentence remembers the lines it covers, because "next line" moves by line, not by sentence.
Pure Python with no heavy imports, so it can also run inside NVDA.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

MAX_CHARS = 180                     # about 12 seconds of speech: the most a rewind ever repeats
SHORT_HEADING_WORDS = 3             # a line this short, without punctuation, is a heading

_TERMINATORS = ".!?؟…"
_CLOSERS = "\"'”’»)]}"
_ENDS_WITH_PUNCTUATION = re.compile(r"[.!?؟…،,;؛:]$")
_CLAUSE_MARKS = "،,؛;:"
_AR = "ء-يٱ"

# a bullet or a list number at the start of a line: "•", "- ", "1.", "(2)", "a)", "أ-"
_BULLET = re.compile(rf"^\s*(?:[•●▪■◦*\-–—]|\(?(?:\d{{1,3}}|[٠-٩]{{1,3}}|[A-Za-z]|[{_AR}])[.)\-–])\s+")
_LIST_MARKER = re.compile(rf"^\(?(?:\d{{1,3}}|[٠-٩]{{1,3}}|[A-Za-z]|[{_AR}])$")

_TITLES = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "mt", "st"}
_ALWAYS = {"vs", "cf", "fig", "figs", "pp", "p", "vol", "ch", "eds", "approx", "dept", "inc", "ltd", "co"}
_BEFORE_DIGIT = {"no", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "art"}
_BEFORE_LOWER = {"etc"}
_LATIN_DOTTED = re.compile(r"^(?:[A-Za-z]\.)+[A-Za-z]$")
_ARABIC_DOTTED = re.compile(rf"^(?:[{_AR}]\.)+[{_AR}]$")
_ARABIC_LETTER = re.compile(rf"^[{_AR}]$")


@dataclass(frozen=True)
class Sentence:
    text: str
    first_line: int         # index into the lines that were given (blank lines count)
    last_line: int


# ----------------------------------------------------------------------------
# Lines -> paragraphs
# ----------------------------------------------------------------------------
def _is_heading_like(line: str) -> bool:
    return (len(line.split()) <= SHORT_HEADING_WORDS and not _ENDS_WITH_PUNCTUATION.search(line)
            and not line.endswith("-"))                             # "compu-" goes on in the next line


def _paragraphs(lines: Sequence[str], breaks: Iterable[int]) -> List[List[Tuple[int, str]]]:
    """Group non-blank lines (kept with their index) into flowing paragraphs."""
    forced = set(breaks)
    out: List[List[Tuple[int, str]]] = []
    cur: List[Tuple[int, str]] = []
    prev_idx: Optional[int] = None
    for idx, raw in enumerate(lines):
        line = " ".join(raw.split())
        if not line:
            if cur:
                out.append(cur)
            cur, prev_idx = [], None
            continue
        if cur:
            prev_line = cur[-1][1]
            if (_BULLET.match(line) or _is_heading_like(prev_line) or prev_idx in forced):
                out.append(cur)
                cur = []
        cur.append((idx, line))
        prev_idx = idx
    if cur:
        out.append(cur)
    return out


def _flow(para: Sequence[Tuple[int, str]]) -> Tuple[str, List[int]]:
    """One string for the paragraph and the line index of every character."""
    text = ""
    owner: List[int] = []
    for idx, line in para:
        if text:
            if text.endswith("-") and len(text) > 1 and text[-2].isalpha() and line[:1].islower():
                text, owner = text[:-1], owner[:-1]                 # compu- / ter -> computer
            else:
                text += " "
                owner.append(owner[-1])
        text += line
        owner += [idx] * len(line)
    return text, owner


# ----------------------------------------------------------------------------
# Where a sentence ends
# ----------------------------------------------------------------------------
def _next_char(text: str, pos: int) -> str:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return text[pos] if pos < len(text) else ""


def _last_token(text: str, end: int) -> str:
    token = text[:end].split(" ")[-1] if end else ""
    return token.lstrip("(\"'“‘«[")


def _is_boundary(text: str, start: int, i: int, j: int, k: int) -> bool:
    """text[i:j+1] is a run of . ! ? ؟ …, text[j+1:k+1] closing quotes or brackets after it."""
    group, closers, nxt = text[i:j + 1], text[j + 1:k + 1], _next_char(text, k + 1)
    if any(c in group for c in "!?؟"):
        # '"Stop!" he said' goes on; a closing quote followed by a lowercase word is a dialogue tag
        return not (closers and nxt.islower() and nxt.isascii())
    if len(group) >= 3 or "…" in group:                       # an ellipsis
        return not (nxt.islower() and nxt.isascii())
    if group != ".":
        return True
    so_far = text[start:i].strip()
    if _LIST_MARKER.match(so_far):                              # "1." / "أ." in front of an item
        return False
    token = _last_token(text, i)
    low = token.lower()
    if low in _TITLES or low in _ALWAYS:
        return False
    if low in _BEFORE_DIGIT and nxt.isdigit():
        return False
    if low in _BEFORE_LOWER and nxt.islower() and nxt.isascii():
        return False
    if _LATIN_DOTTED.match(token):
        return not (low in ("e.g", "i.e") or nxt.islower() or nxt.isdigit())
    if _ARABIC_LETTER.match(token) or _ARABIC_DOTTED.match(token):    # د. أحمد   أ.د. سارة   ق.م.
        return False
    if re.fullmatch(r"[A-Z]", token):                           # J. K. Rowling, but not "Vitamin C. It"
        before = text[:i - len(token)].rstrip().split(" ")[-1] if i - len(token) > 0 else ""
        after = re.match(r"\s*([A-Z])\.", text[k + 1:])
        if re.fullmatch(r"[A-Z]\.", before) or after:
            return False
    return True


def _cut_long(text: str, a: int, b: int, max_chars: int) -> List[Tuple[int, int]]:
    """Cut text[a:b] into parts of at most max_chars, preferring a comma, then a space."""
    parts: List[Tuple[int, int]] = []
    while b - a > max_chars:
        window = text[a:a + max_chars + 1]
        cut = max((window.rfind(c + " ") + 1 for c in _CLAUSE_MARKS if window.rfind(c + " ") >= 0), default=0)
        if cut < max_chars // 2:
            cut = window.rfind(" ")
            if cut < max_chars // 2:
                cut = max_chars
        parts.append((a, a + cut))
        a += cut
        while a < b and text[a].isspace():
            a += 1
    parts.append((a, b))
    return parts


def _split_paragraph(para: Sequence[Tuple[int, str]], max_chars: int) -> List[Sentence]:
    text, owner = _flow(para)
    spans: List[Tuple[int, int]] = []
    start, i, n = 0, 0, len(text)
    while i < n:
        if text[i] not in _TERMINATORS:
            i += 1
            continue
        j = i
        while j + 1 < n and text[j + 1] in _TERMINATORS:
            j += 1
        k = j
        while k + 1 < n and text[k + 1] in _CLOSERS:
            k += 1
        if (k + 1 == n or text[k + 1].isspace()) and _is_boundary(text, start, i, j, k):
            spans.append((start, k + 1))
            start = k + 1
        i = k + 1
    spans.append((start, n))

    out: List[Sentence] = []
    for a, b in spans:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a >= b:
            continue
        for x, y in _cut_long(text, a, b, max_chars):
            piece = text[x:y].strip()
            if piece and re.search(rf"[A-Za-z0-9٠-٩{_AR}]", piece):
                out.append(Sentence(" ".join(piece.split()), owner[x], owner[y - 1]))
    return out


# ----------------------------------------------------------------------------
# Public
# ----------------------------------------------------------------------------
def split_sentences(source: Union[str, Sequence[str]], max_chars: int = MAX_CHARS,
                    breaks: Iterable[int] = ()) -> List[Sentence]:
    """Sentences of a text, or of a list of lines.

    ``breaks`` are line indices after which a paragraph must end; the OCR worker passes them
    when the layout (a larger gap, a different indent) shows a break that the text itself does not."""
    lines = source.splitlines() if isinstance(source, str) else list(source)
    out: List[Sentence] = []
    for para in _paragraphs(lines, breaks):
        out.extend(_split_paragraph(para, max_chars))
    return out

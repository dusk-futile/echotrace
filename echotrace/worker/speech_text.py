"""Turn text into speakable runs for the voice.

The voice (Supertonic) reads raw characters and has no pronunciation dictionary,
so tokens such as ``Homework.docx`` or ``PDF`` are guessed from their spelling and
sometimes come out wrong (a doubled "s" in "docx"). This module

  1. cleans the text (bidi marks, the two Arabic characters the model cannot read),
  2. splits it into Arabic and English runs, so each run is sent with the right
     language tag but the SAME voice,
  3. rewrites risky tokens into plain words first: file names and extensions,
     acronyms, URLs and e-mail addresses, key combinations, brand names.

The rules are data, not code: ``Lexicon`` can be extended from a JSON file so a
teacher or helper can fix a mispronounced word without touching the program.

Pure Python with no heavy imports, so it can also run inside NVDA.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# Arabic letters and the marks that live inside words (diacritics, superscript alef).
_AR_LETTERS = "ء-يٮ-ۓۺ-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿"
_AR_MARKS = "ً-ٰٟۖ-ۭ"
_IS_ARABIC = re.compile(f"[{_AR_LETTERS}{_AR_MARKS}]")
_IS_LATIN = re.compile(r"[A-Za-z]")
_BULLETS = re.compile("[•●▪■◦►▶◆·]")
_INVISIBLE = re.compile("[​-‏‪-‮⁦-⁩﻿­]")


@dataclass(frozen=True)
class Run:
    lang: str      # "ar" or "en"
    text: str


# ----------------------------------------------------------------------------
# Lexicon
# ----------------------------------------------------------------------------
DEFAULT_EXTENSIONS: Dict[str, str] = {
    # words people say
    "docx": "doc x", "doc": "doc", "odt": "O D T", "rtf": "R T F", "txt": "T X T",
    "xlsx": "X L S X", "xls": "X L S", "ods": "O D S", "csv": "C S V",
    "pptx": "P P T X", "ppt": "P P T", "pdf": "P D F",
    "jpg": "J P G", "jpeg": "J peg", "png": "P N G", "gif": "gif", "bmp": "B M P",
    "mp3": "M P 3", "mp4": "M P 4", "wav": "wave",
    "zip": "zip", "rar": "rar", "iso": "I S O", "exe": "E X E", "apk": "A P K", "bat": "bat",
    "html": "H T M L", "htm": "H T M", "xml": "X M L", "json": "jay son", "js": "J S", "py": "P Y",
}

DEFAULT_WORDS: Dict[str, str] = {
    "whatsapp": "Whats App", "youtube": "You Tube", "powerpoint": "Power Point",
    "javascript": "Java Script", "github": "Git Hub", "wifi": "Wi Fi", "wi-fi": "Wi Fi",
    "iphone": "i Phone", "ipad": "i Pad", "ok": "okay",
    "nacl": "N a C l", "hcl": "H C l", "naoh": "N a O H",
    "ctrl": "control", "esc": "escape", "del": "delete", "win": "windows",
}

# All-caps tokens that are read as words, not spelled out.
DEFAULT_WORD_ACRONYMS: Set[str] = {
    "NASA", "UNESCO", "UNICEF", "NATO", "OPEC", "FIFA", "RAM", "ROM", "LAN", "WAN", "BIOS",
    "PIN", "DOS", "OK", "AIDS", "SIM", "COVID",
}

# Acronyms that are always spelled out, even inside an ALL-CAPS heading.
DEFAULT_SPELLED: Set[str] = {
    "PDF", "USB", "CPU", "GPU", "HTML", "CSS", "SQL", "API", "DNA", "RNA", "TV", "PC", "AI", "UK",
    "USA", "UN", "EU", "FAQ", "ID", "IT", "URL", "HDMI", "LCD", "LED", "DVD", "CD", "VR", "SMS",
    "GPS", "PHP", "XML", "OCR", "TTS", "NVDA", "JAWS",
}

_TLDS = "com|org|net|edu|gov|io|info|eg|sa|ae|kw|qa|jo|lb|ma|dz|tn"


@dataclass
class Lexicon:
    extensions: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EXTENSIONS))
    words: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_WORDS))
    word_acronyms: Set[str] = field(default_factory=lambda: set(DEFAULT_WORD_ACRONYMS))
    spelled: Set[str] = field(default_factory=lambda: set(DEFAULT_SPELLED))

    def merged(self, data: dict) -> "Lexicon":
        """Return a copy with a user JSON merged in:
        {"extensions": {}, "words": {}, "word_acronyms": [], "spelled": []}."""
        return Lexicon(
            extensions={**self.extensions, **{k.lower(): v for k, v in data.get("extensions", {}).items()}},
            words={**self.words, **{k.lower(): v for k, v in data.get("words", {}).items()}},
            word_acronyms=self.word_acronyms | {w.upper() for w in data.get("word_acronyms", [])},
            spelled=self.spelled | {w.upper() for w in data.get("spelled", [])},
        )


def load_lexicon(user_json_path: Optional[str] = None) -> Lexicon:
    lex = Lexicon()
    if user_json_path and os.path.exists(user_json_path):
        try:
            with open(user_json_path, encoding="utf-8") as f:
                lex = lex.merged(json.load(f))
        except (OSError, ValueError):
            pass                      # a broken user file must never stop speech
    return lex


# ----------------------------------------------------------------------------
# Cleaning and run splitting
# ----------------------------------------------------------------------------
def clean(text: str) -> str:
    """Normalise text for the voice. Keeps diacritics; fixes characters the model cannot read."""
    t = unicodedata.normalize("NFKC", text)             # presentation forms from OCR/PDF -> base letters
    t = _INVISIBLE.sub("", t)
    t = t.replace("ـ", "")                          # tatweel
    t = t.replace("؛", "،").replace("ٱ", "ا")   # Arabic semicolon, alef wasla
    t = _BULLETS.sub(" ", t)                            # list marks are layout, not speech
    return " ".join(t.split())


def split_runs(text: str, default_lang: str = "ar") -> List[Run]:
    """Split into Arabic and English runs. Digits and punctuation stay with the run before them.

    Text with no letters at all (for example just "1908") is returned as one run in default_lang."""
    runs: List[List[str]] = []
    lead = ""
    for ch in text:
        if _IS_ARABIC.match(ch):
            lang = "ar"
        elif _IS_LATIN.match(ch):
            lang = "en"
        else:
            lang = ""
        if not lang:
            if runs:
                runs[-1][1] += ch
            else:
                lead += ch
        elif runs and runs[-1][0] == lang:
            runs[-1][1] += ch
        else:
            runs.append([lang, lead + ch])
            lead = ""
    if not runs and lead.strip():
        return [Run(default_lang, lead)]
    return _move_openers([Run(lang, seg) for lang, seg in runs])


_OPENERS = "([{«“‘"


def _move_openers(runs: List[Run]) -> List[Run]:
    """An opening bracket or quote belongs to the word after it, not to the run before it:
    "الضوئي (Photosynthesis)" must give "الضوئي" and "(Photosynthesis)", not "الضوئي (" and "Photosynthesis)"."""
    texts = [r.text for r in runs]
    for i in range(len(texts) - 1):
        t = texts[i].rstrip()
        k = len(t)
        while k > 0 and (t[k - 1] in _OPENERS or (t[k - 1] in "\"'" and (k == 1 or t[k - 2].isspace()))):
            k -= 1
        if k < len(t):
            texts[i + 1] = t[k:] + texts[i + 1]
            texts[i] = t[:k]
    return [Run(r.lang, text) for r, text in zip(runs, texts)]


# ----------------------------------------------------------------------------
# Rewrites
# ----------------------------------------------------------------------------
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_DOMAIN = re.compile(rf"\b(?:www\.)?[\w-]+(?:\.[\w-]+)*\.(?:{_TLDS})\b", re.IGNORECASE)
_FILENAME = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9_-]*)\.([A-Za-z0-9]{1,5})\b")
_KEYCOMBO = re.compile(r"\b(?:Ctrl|Alt|Shift|Win|Cmd)(?:\+[A-Za-z0-9]+)+\b")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z][a-z]{2,})")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")


def _dotted(s: str) -> str:
    return s.replace(".", " dot ")


def rewrite_latin(text: str, lex: Lexicon) -> str:
    t = text.replace("،", ",").replace("؟", "?")
    t = _EMAIL.sub(lambda m: m.group().replace("@", " at ").replace(".", " dot "), t)

    def domain(m: re.Match) -> str:
        s = m.group()
        if s.lower().startswith("www."):
            s = "w w w " + s[4:]
        return _dotted(s)
    t = _DOMAIN.sub(domain, t)

    def filename(m: re.Match) -> str:
        ext = m.group(2).lower()
        if ext in lex.extensions:
            return f"{m.group(1)} dot {lex.extensions[ext]}"
        return m.group()
    t = _FILENAME.sub(filename, t)

    def combo(m: re.Match) -> str:
        parts = []
        for p in m.group().split("+"):
            parts.append(lex.words.get(p.lower(), p))
        return " plus ".join(parts)
    t = _KEYCOMBO.sub(combo, t)

    # An ALL-CAPS line ("WELCOME TO SCHOOL") is a heading, not a row of acronyms.
    letters = [c for c in t if c.isascii() and c.isalpha()]
    shouting = (len(_TOKEN.findall(t)) >= 2 and bool(letters)
                and sum(c.isupper() for c in letters) / len(letters) >= 0.8)

    def word(m: re.Match) -> str:
        tok = m.group()
        key = tok.lower()
        if key in lex.words:
            return lex.words[key]
        if (tok.isupper() and tok.isalpha() and 2 <= len(tok) <= 6 and tok not in lex.word_acronyms
                and (not shouting or tok in lex.spelled)):
            return " ".join(tok)                          # spell it: P D F
        if len(tok) > 6 and _CAMEL.search(tok):
            parts = [p for p in _CAMEL.split(tok) if p]
            if all(len(p) >= 3 for p in parts):
                return " ".join(parts)
        m2 = re.fullmatch(r"([A-Z]{2,5})(\d+)", tok)       # MP3 -> M P 3, but COVID19 stays a word
        if m2:
            return tok if m2.group(1) in lex.word_acronyms else " ".join(m2.group(1)) + " " + m2.group(2)
        if re.search(r"\d", tok) and re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", tok):
            return " ".join(re.findall(r"[A-Z][a-z]?|\d+", tok))   # H2O -> H 2 O, A4 -> A 4
        return tok
    t = _TOKEN.sub(word, t)

    t = t.replace("&", " and ").replace("%", " percent")
    return " ".join(t.split())


def rewrite_arabic(text: str) -> str:
    t = text.replace("%", " في المئة").replace("&", " و ")
    return " ".join(t.split())


def prepare(text: str, lex: Optional[Lexicon] = None) -> List[Run]:
    """Text -> runs ready for synthesis (language tag + rewritten text)."""
    lex = lex or Lexicon()
    out: List[Run] = []
    for run in split_runs(clean(text)):
        body = rewrite_arabic(run.text) if run.lang == "ar" else rewrite_latin(run.text, lex)
        if not re.search(r"[A-Za-z0-9\u0660-\u0669\u06F0-\u06F9" + _AR_LETTERS + "]", body):
            # punctuation-only leftovers join the previous run instead of being spoken alone
            if out:
                out[-1] = Run(out[-1].lang, (out[-1].text + " " + body).strip())
            continue
        out.append(Run(run.lang, body))
    return out

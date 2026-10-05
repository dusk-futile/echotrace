#!/usr/bin/env python3
"""Synthetic screen-like text lines for the OCR bake-off.

KITAB-Bench has no English and no dark themes, and says nothing about the
mixed Arabic/English text that textbooks and web pages are full of. This
renders tight single-line crops, the way a screenshot region looks:

    kinds   arabic | mixed (Arabic with 1-2 English terms inside) | english
    themes  light (black on white) | dark (light grey on near-black)
    sizes   11, 13, 16, 20 px (Tahoma / Arial from this Mac)

Arabic text is real Arabic Wikipedia prose from eval/corpus. English is random
common words from the public-domain wordlist, so the OCR dictionaries are not
fed nonsense. It is still synthetic: anti-aliased grayscale, no ClearType, so
treat results as a floor for how hard real screens are, and prefer the real
screenshots when they arrive. Written next to the KITAB samples so the same
runner scores both.
"""
from __future__ import annotations

import glob
import gzip
import html
import json
import os
import random
import re
import sys

import cv2
import numpy as np
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MUBSIR_ROOT") or os.path.dirname(os.path.dirname(HERE))   # a checkout of github.com/dusk-futile/mubsir
OUT = os.path.join(HERE, "data", "kitab")
SUPPLEMENTAL = "/System/Library/Fonts/Supplemental"

SEED = 7
SIZES = [11, 13, 16, 20]
PER_CELL = 24
THEMES = {"light": ("#000000", (1.0, 1.0, 1.0)), "dark": ("#e8e8e8", (0.12, 0.12, 0.12))}
EN_TERMS = ["Microsoft Word", "Google Chrome", "WhatsApp", "PDF", "Windows 10", "YouTube",
            "email", "Wikipedia", "Python", "Facebook", "Excel", "Zoom", "Internet", "Android"]
CSS = ("@font-face{font-family:'Tahoma';src:url('Tahoma.ttf')} "
       "@font-face{font-family:'Arial';src:url('Arial.ttf')} "
       "* {font-family:'%s'; font-size:%dpx; margin:0; padding:0;}")


def arabic_words() -> list[list[str]]:
    paras = []
    for f in sorted(glob.glob(os.path.join(ROOT, "eval", "corpus", "*.txt"))):
        for p in open(f, encoding="utf-8").read().split("\n\n"):
            ws = p.split()
            # plain Arabic only: no Latin, no digits, no brackets, so the gold text is unambiguous
            ws = [w for w in ws if re.fullmatch(r"[ء-يٮ-ۓـ،؛؟.:]+", w)]
            if len(ws) >= 30:
                paras.append(ws)
    return paras


def english_words() -> list[str]:
    path = os.path.join(ROOT, "models", "lexicon", "en_words.txt.gz")
    ws = [w for w in gzip.open(path, "rt", encoding="utf-8").read().split() if 3 <= len(w) <= 9]
    return ws


def make_text(kind: str, size: int, rng: random.Random, arabic, english) -> str:
    budget = int(900 / (size * 0.5))                      # characters that fit in ~900 px
    if kind == "english":
        words, n = [], 0
        while n < budget * 0.55 and len(words) < 12:
            w = rng.choice(english)
            words.append(w)
            n += len(w) + 1
        words[0] = words[0].capitalize()
        return " ".join(words) + "."
    para = rng.choice(arabic)
    start = rng.randrange(0, len(para) - 14)
    words, n = [], 0
    for w in para[start:start + 14]:
        if n + len(w) + 1 > budget * 0.55 and len(words) >= 5:
            break
        words.append(w)
        n += len(w) + 1
    if kind == "mixed":
        for term in rng.sample(EN_TERMS, k=rng.choice([1, 2])):
            words.insert(rng.randrange(1, len(words)), term)
    return " ".join(words)


def render(text: str, size: int, theme: str, family: str) -> np.ndarray:
    fg, bg = THEMES[theme]
    w, h = 1500, int(size * 2.4)
    doc = pymupdf.open()
    page = doc.new_page(width=w, height=h)
    page.draw_rect(page.rect, color=None, fill=bg)
    direction = "ltr" if re.match(r"[A-Za-z]", text) else "rtl"
    body = f"<p dir='{direction}' style='color:{fg}'>{html.escape(text)}</p>"
    page.insert_htmlbox(pymupdf.Rect(6, 4, w - 6, h - 2), body, css=CSS % (family, size),
                        archive=pymupdf.Archive(SUPPLEMENTAL))
    pix = page.get_pixmap(dpi=72)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Crop to the ink the way a screenshot region would be: text plus a small margin.
    bgv = img[0, 0].astype(int)
    ink = (np.abs(img.astype(int) - bgv).sum(axis=2) > 60)
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return img
    m = max(4, size // 2)
    return img[max(0, ys.min() - m):ys.max() + m + 1, max(0, xs.min() - m):xs.max() + m + 1]


def main() -> int:
    rng = random.Random(SEED)
    arabic, english = arabic_words(), english_words()
    if not arabic:
        print("no Arabic corpus found in eval/corpus")
        return 1
    total = 0
    for kind in ("arabic", "mixed", "english"):
        for theme in THEMES:
            sub = os.path.join(OUT, f"scr_{kind}_{theme}")
            os.makedirs(sub, exist_ok=True)
            meta = []
            idx = 0
            for size in SIZES:
                for _ in range(PER_CELL // len(SIZES)):
                    text = make_text(kind, size, rng, arabic, english)
                    family = rng.choice(["Tahoma", "Arial"])
                    img = render(text, size, theme, family)
                    stem = os.path.join(sub, f"{idx:04d}_s{size}")
                    cv2.imwrite(stem + ".png", img)
                    open(stem + ".txt", "w", encoding="utf-8").write(text)
                    meta.append({"idx": idx, "size": size, "family": family,
                                 "w": img.shape[1], "h": img.shape[0]})
                    idx += 1
                    total += 1
            json.dump(meta, open(os.path.join(sub, "manifest.json"), "w"))
            print(f"  scr_{kind}_{theme}: {idx} images")
    print(f"{total} synthetic screen lines written under {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

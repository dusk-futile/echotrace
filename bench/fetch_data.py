#!/usr/bin/env python3
"""Fetch the Phase 0 test material. Standard library only.

Everything lands in bench/data/ (git-ignored). Re-running skips files
that are already there.

    python bench/fetch_data.py models           # PP-OCRv5 ONNX, ~21 MB
    python bench/fetch_data.py kitab --rows 40  # KITAB-Bench sample, ~10 MB
    python bench/fetch_data.py voices           # Piper voices, ~190 MB

Sources (all public, open source):
  * PP-OCRv5 ONNX      github.com/GreatV/oar-ocr  release v0.3.0
  * KITAB-Bench        huggingface.co/datasets/ahmedheakl/arocrbench_*   (MBZUAI, MIT)
  * Piper voices       huggingface.co/rhasspy/piper-voices                (MIT repo)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
UA = {"User-Agent": "echotrace-bench/0.1 (phase-0 evidence gathering)"}

PPOCR_BASE = "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/"
PPOCR_FILES = [
    "arabic_pp-ocrv5_mobile_rec.onnx",   # 7.7 MiB, reads Arabic and English
    "ppocrv5_arabic_dict.txt",
    "pp-ocrv5_mobile_det.onnx",          # 4.6 MiB line detector
]

KITAB_DATASET = "ahmedheakl/arocrbench_"
# Printed-text subsets only: handwriting and historical manuscripts are not the use case.
KITAB_SUBSETS = ["patsocr", "synthesizear", "hindawi", "arabicocr", "isippt"]
HF_API = "https://datasets-server.huggingface.co"

VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
VOICE_FILES = [
    "ar/ar_JO/kareem/low/ar_JO-kareem-low.onnx",
    "ar/ar_JO/kareem/low/ar_JO-kareem-low.onnx.json",
    "ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx",
    "ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx.json",
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
]


def _open(url: str, timeout: int = 120):
    last = None
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}: {last}")


def download(url: str, dest: str) -> int:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  have  {os.path.relpath(dest, HERE)}")
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with _open(url) as r, open(tmp, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    os.replace(tmp, dest)
    print(f"  got   {os.path.relpath(dest, HERE)}  {total / 1e6:.1f} MB")
    return total


def get_json(path: str, **query) -> dict:
    url = f"{HF_API}{path}?{urllib.parse.urlencode(query)}"
    with _open(url) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_models() -> None:
    print("PP-OCRv5 ONNX models")
    for name in PPOCR_FILES:
        download(PPOCR_BASE + name, os.path.join(DATA, "models", "ppocrv5", name))


def fetch_kitab(rows: int, subsets: list[str]) -> None:
    print(f"KITAB-Bench sample: {rows} rows per subset")
    for sub in subsets:
        ds = KITAB_DATASET + sub
        out = os.path.join(DATA, "kitab", sub)
        os.makedirs(out, exist_ok=True)
        try:
            first = get_json("/rows", dataset=ds, config="default", split="train", offset=0, length=1)
        except Exception as e:                       # one bad subset must not stop the rest
            print(f"  skip  {sub}: {e}")
            continue
        total = int(first.get("num_rows_total") or 0)
        if not total:
            print(f"  skip  {sub}: no rows reported")
            continue
        # Spread the sample over the whole subset instead of taking the first rows.
        step = max(1, total // rows)
        wanted = list(range(0, total, step))[:rows]
        manifest, got = [], 0
        for idx in wanted:
            stem = os.path.join(out, f"{idx:04d}")
            if os.path.exists(stem + ".txt") and any(
                    os.path.exists(stem + ext) for ext in (".png", ".jpg")):
                got += 1
                continue
            try:
                page = get_json("/rows", dataset=ds, config="default", split="train",
                                offset=idx, length=1)
                row = page["rows"][0]["row"]
                img = row["image"]
                gold = row.get("answer") or row.get("text") or row.get("ground_truth") or ""
                with _open(img["src"]) as r:
                    data = r.read()
                    ctype = r.headers.get("Content-Type", "")
                ext = ".jpg" if "jpeg" in ctype else ".png"
                with open(stem + ext, "wb") as f:
                    f.write(data)
                with open(stem + ".txt", "w", encoding="utf-8") as f:
                    f.write(gold)
                manifest.append({"idx": idx, "w": img.get("width"), "h": img.get("height"),
                                 "source": row.get("source")})
                got += 1
            except Exception as e:
                print(f"  warn  {sub} row {idx}: {e}")
            time.sleep(0.15)                          # be polite to the public API
        if manifest:
            with open(os.path.join(out, "manifest.json"), "w") as f:
                json.dump(manifest, f)
        size = sum(os.path.getsize(os.path.join(out, n)) for n in os.listdir(out))
        print(f"  {sub:14s} {got}/{len(wanted)} samples of {total}, {size / 1e6:.1f} MB")


def fetch_voices() -> None:
    print("Piper voices")
    for rel in VOICE_FILES:
        download(VOICES_BASE + rel, os.path.join(DATA, "voices", os.path.basename(rel)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("what", choices=["models", "kitab", "voices", "all"])
    ap.add_argument("--rows", type=int, default=40, help="KITAB samples per subset")
    ap.add_argument("--subsets", nargs="*", default=KITAB_SUBSETS)
    a = ap.parse_args()
    if a.what in ("models", "all"):
        fetch_models()
    if a.what in ("kitab", "all"):
        fetch_kitab(a.rows, a.subsets)
    if a.what in ("voices", "all"):
        fetch_voices()
    return 0


if __name__ == "__main__":
    sys.exit(main())

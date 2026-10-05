#!/usr/bin/env python3
"""Break OCR results down by font size, line height, or subset.

    python bench/breakdown.py --tag screen --by size
    python bench/breakdown.py --by height --engines tess ppocr_v5

Reads bench/results/ocr_<engine>[__tag].json written by ocr_bench.py.
Totals are micro-averaged (total edits / total reference characters), scored on
the images every selected engine was run on (so engines are comparable).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data", "kitab")


def height_bucket(h: int) -> str:
    return "h<25" if h < 25 else "25-34" if h < 35 else "35-59" if h < 60 else "h>=60"


def key_for(by: str, row: dict) -> str:
    if by == "subset":
        return row["subset"]
    if by == "size":
        m = re.search(r"_s(\d+)\.", row["file"])
        return f"{int(m.group(1)):02d}px" if m else "n/a"
    if by == "height":
        img = cv2.imread(os.path.join(DATA, row["subset"], row["file"]))
        return height_bucket(img.shape[0]) if img is not None else "n/a"
    if by == "kind":                                  # screen set: arabic / mixed / english
        m = re.match(r"scr_([a-z]+)_", row["subset"])
        return m.group(1) if m else row["subset"]
    if by == "theme":
        m = re.match(r"scr_[a-z]+_([a-z]+)", row["subset"])
        return m.group(1) if m else "n/a"
    raise ValueError(by)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tag", default="")
    ap.add_argument("--by", choices=["size", "height", "subset", "kind", "theme"], required=True)
    ap.add_argument("--engines", nargs="*")
    ap.add_argument("--subsets", nargs="*", help="restrict to these subsets (prefix match)")
    ap.add_argument("--metric", choices=["cer", "wer"], default="cer")
    a = ap.parse_args()

    data = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "ocr_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("tag", "") == a.tag and (not a.engines or d["engine"] in a.engines):
            data.append(d)
    if not data:
        print("no matching results")
        return 1
    keys = None
    for d in data:
        k = {(r["subset"], r["file"]) for r in d["rows"]
             if not a.subsets or any(r["subset"].startswith(p) for p in a.subsets)}
        keys = k if keys is None else keys & k
    unit, den = ("", "ref_len_plain") if a.metric == "cer" else ("w", "ref_words_plain")
    groups = collections.OrderedDict()
    for d in data:
        for r in d["rows"]:
            if (r["subset"], r["file"]) in keys:
                groups.setdefault(key_for(a.by, r), None)
    order = sorted(groups)
    print(f"{a.metric.upper()} % (plain), by {a.by}, on {len(keys)} common images, lower is better\n")
    print("| engine | " + " | ".join(order) + " | all |")
    print("|---|" + "---|" * (len(order) + 1))
    counts = {g: sum(1 for (s, f) in keys if key_for(a.by, {"subset": s, "file": f}) == g) for g in order}
    for d in data:
        acc = {g: [0, 0] for g in order}
        tot = [0, 0]
        for r in d["rows"]:
            if (r["subset"], r["file"]) not in keys:
                continue
            g = key_for(a.by, r)
            acc[g][0] += r[f"{unit}edits_plain"]
            acc[g][1] += r[den]
            tot[0] += r[f"{unit}edits_plain"]
            tot[1] += r[den]
        cells = [f"{100 * acc[g][0] / acc[g][1]:.1f}" if acc[g][1] else "-" for g in order]
        print(f"| {d['engine']} | " + " | ".join(cells) + f" | **{100 * tot[0] / max(1, tot[1]):.1f}** |")
    print("\nimages per group:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())

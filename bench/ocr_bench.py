#!/usr/bin/env python3
"""Phase 0 OCR bake-off runner and summariser.

    python bench/ocr_bench.py run --engine tess_up [--limit 40]
    python bench/ocr_bench.py summary

Each ``run`` is one engine in its own process, so peak memory is that engine's
own. Results go to bench/results/ocr_<engine>.json.

Scoring: the reference and the hypothesis are normalised the same way (NFC,
bidi/zero-width controls removed, digits unified, whitespace collapsed). The
headline number ("plain") also drops diacritics and tatweel, because none of
the offline engines emit tashkeel and our text-to-speech adds its own. The
"raw" number keeps them. Totals are micro-averaged: total edits over total
reference characters, so one short line cannot dominate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import resource
import statistics
import sys
import time
import unicodedata

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MUBSIR_ROOT") or os.path.dirname(os.path.dirname(HERE))   # a checkout of github.com/dusk-futile/mubsir
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

try:
    from eval.metrics import levenshtein                   # noqa: E402
    from mubsir.arabic import strip_tashkeel                # noqa: E402
except ImportError:
    sys.exit("The OCR bench uses a checkout of mubsir (github.com/dusk-futile/mubsir) for its metrics, models and "
             "corpora. Set MUBSIR_ROOT to that folder (it contains mubsir/, eval/ and models/).")

DATA = os.path.join(HERE, "data", "kitab")
RESULTS = os.path.join(HERE, "results")
KIND = {"patsocr": "line", "synthesizear": "line", "isippt": "line",
        "arabicocr": "block", "hindawi": "page"}

_CTRL = re.compile("[​-‏‪-‮⁦-⁩﻿]")
_TATWEEL = re.compile("ـ")
_DIGITS = {ord(a): str(i) for i, a in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGITS.update({ord(a): str(i) for i, a in enumerate("۰۱۲۳۴۵۶۷۸۹")})


def normalise(text: str, plain: bool) -> str:
    t = unicodedata.normalize("NFC", text)
    t = _CTRL.sub("", t).translate(_DIGITS)
    if plain:
        t = _TATWEEL.sub("", strip_tashkeel(t))
    return " ".join(t.split())


def load_samples(subsets, limit):
    items = []
    for sub in subsets:
        d = os.path.join(DATA, sub)
        imgs = sorted(p for p in glob.glob(d + "/*") if p.endswith((".png", ".jpg")))[:limit]
        for p in imgs:
            gold = open(p.rsplit(".", 1)[0] + ".txt", encoding="utf-8").read()
            items.append((sub, os.path.basename(p), p, gold))
    return items


def run(args) -> int:
    import engines
    subsets = args.subsets or list(KIND)
    items = load_samples(subsets, args.limit)
    if not items:
        print("no samples: run fetch_data.py kitab first")
        return 1
    eng = engines.make(args.engine)
    # Warm-up so model load and first-call overhead are not charged to sample 1.
    eng.read(cv2.imread(items[0][2]), KIND.get(items[0][0], "line"))
    rows = []
    for k, (sub, name, path, gold) in enumerate(items, 1):
        img = cv2.imread(path)
        t = time.time()
        try:
            hyp = eng.read(img, KIND.get(sub, "line"))
            err = ""
        except Exception as e:                              # a crash is a result, not a stop
            hyp, err = "", f"{type(e).__name__}: {e}"
        dt = time.time() - t
        row = {"subset": sub, "file": name, "secs": round(dt, 3), "err": err,
               "hyp": hyp[:300], "ref": gold[:300]}
        usage = getattr(eng, "last_usage", None)
        if usage:                                   # cloud models: tokens and dollars per sample
            price = engines.PRICES.get(getattr(eng, "model", ""), (0.0, 0.0))
            row["in_tok"], row["out_tok"] = usage["in"], usage["out"]
            row["usd"] = (usage["in"] * price[0] + usage["out"] * price[1]) / 1e6
        for tag, plain in (("plain", True), ("raw", False)):
            r, h = normalise(gold, plain), normalise(hyp, plain)
            row[f"edits_{tag}"] = levenshtein(r, h)
            row[f"ref_len_{tag}"] = len(r)
            rw, hw = r.split(), h.split()
            row[f"wedits_{tag}"] = levenshtein(rw, hw)
            row[f"ref_words_{tag}"] = len(rw)
        rows.append(row)
        if k % 20 == 0:
            print(f"  {args.engine}: {k}/{len(items)}", flush=True)
    # Tesseract runs as a child process, so its memory only shows under RUSAGE_CHILDREN.
    rss = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3
    os.makedirs(RESULTS, exist_ok=True)
    suffix = f"__{args.tag}" if args.tag else ""
    out = os.path.join(RESULTS, f"ocr_{args.engine}{suffix}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"engine": args.engine, "tag": args.tag or "", "n": len(rows), "load_seconds": round(eng.load_seconds, 2),
                   "peak_rss_mb": round(rss_mb), "rows": rows}, f, ensure_ascii=False)
    print(f"wrote {out}  ({len(rows)} samples, load {eng.load_seconds:.1f}s, peak {rss_mb:.0f} MB)")
    return 0


def micro(rows, tag, unit):
    num = sum(r[f"{unit}edits_{tag}"] for r in rows)
    den = sum(r[f"ref_{'words' if unit else 'len'}_{tag}"] for r in rows)
    return 100.0 * num / den if den else float("nan")


def summary(args) -> int:
    files = sorted(glob.glob(os.path.join(RESULTS, "ocr_*.json")))
    if not files:
        print("no results yet")
        return 1
    data = [json.load(open(f, encoding="utf-8")) for f in files]
    data = [d for d in data if d.get("tag", "") == (args.tag or "")]
    if args.engines:
        data = [d for d in data if d["engine"] in args.engines]
    if args.common:
        # Score every engine on exactly the images that all of them were run on.
        keys = None
        for d in data:
            k = {(r["subset"], r["file"]) for r in d["rows"]}
            keys = k if keys is None else keys & k
        for d in data:
            d["rows"] = [r for r in d["rows"] if (r["subset"], r["file"]) in keys]
    present = {r["subset"] for d in data for r in d["rows"]}
    subsets = [s for s in KIND if s in present] + sorted(s for s in present if s not in KIND)

    def table(title, tag, unit):
        print(f"\n### {title}\n")
        print("| engine | " + " | ".join(subsets) + " | average | s/image (median) | peak MB |")
        print("|---|" + "---|" * (len(subsets) + 3))
        for d in data:
            cells, vals = [], []
            for s in subsets:
                rs = [r for r in d["rows"] if r["subset"] == s]
                v = micro(rs, tag, unit) if rs else float("nan")
                cells.append("-" if v != v else f"{v:.1f}")
                if v == v:
                    vals.append(v)
            avg = sum(vals) / len(vals) if vals else float("nan")
            med = statistics.median(r["secs"] for r in d["rows"])
            print(f"| {d['engine']} | " + " | ".join(cells) +
                  f" | **{avg:.1f}** | {med:.2f} | {d['peak_rss_mb']} |")

    n = {s: sum(1 for r in max(data, key=lambda d: len(d["rows"]))["rows"] if r["subset"] == s) for s in subsets}
    print("Samples per subset:", n)
    table("Character error rate % (plain: no diacritics), lower is better", "plain", "")
    table("Word error rate % (plain), lower is better", "plain", "w")
    table("Character error rate % (raw, diacritics kept)", "raw", "")
    crashes = {d["engine"]: sum(1 for r in d["rows"] if r["err"]) for d in data}
    print("\nCrashed samples per engine:", crashes)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", required=True)
    r.add_argument("--limit", type=int, default=40, help="samples per subset")
    r.add_argument("--subsets", nargs="*")
    r.add_argument("--tag", default="", help="result-file suffix, e.g. 'screen'")
    r.set_defaults(fn=run)
    s = sub.add_parser("summary")
    s.add_argument("--engines", nargs="*", help="only these engines")
    s.add_argument("--common", action="store_true", help="only images every listed engine ran on")
    s.add_argument("--tag", default="", help="show results run with this tag (default: untagged)")
    s.set_defaults(fn=summary)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

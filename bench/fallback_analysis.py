#!/usr/bin/env python3
"""Offline analysis: what if a second engine covers the first engine's failures?

    python bench/fallback_analysis.py \
        --tag screen3 --primary tess_ara_eng_up_pol_p13 --secondary v2_v5_word

Uses only saved results (no OCR is re-run). Rules compared, all micro-averaged
character error (plain, no diacritics) over the images both engines ran on:

  primary / secondary      each engine alone
  empty->secondary         use the primary, but the secondary when the primary
                           returned nothing (Tesseract does this on 5-8% of lines)
  short->secondary         ... or when the primary text is under half the length
                           of the secondary text (it dropped most of the line)
  oracle                   the better of the two per image: an upper bound, not
                           something we can know at run time
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("MUBSIR_ROOT") or os.path.dirname(os.path.dirname(HERE)))   # see ocr_bench.py

import ocr_bench as ob                          # noqa: E402
from eval.metrics import levenshtein            # noqa: E402

RESULTS = os.path.join(HERE, "results")


def load(engine: str, tag: str):
    suffix = f"__{tag}" if tag else ""
    path = os.path.join(RESULTS, f"ocr_{engine}{suffix}.json")
    return {(r["subset"], r["file"]): r for r in json.load(open(path, encoding="utf-8"))["rows"]}


def edits(ref: str, hyp: str) -> int:
    return levenshtein(ob.normalise(ref, True), ob.normalise(hyp, True))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tag", default="")
    ap.add_argument("--primary", required=True)
    ap.add_argument("--secondary", required=True)
    a = ap.parse_args()
    P, S = load(a.primary, a.tag), load(a.secondary, a.tag)
    keys = sorted(set(P) & set(S))
    n_ref = sum(len(ob.normalise(P[k]["ref"], True)) for k in keys)

    def total(pick):
        return 100.0 * sum(edits(P[k]["ref"], pick(k)) for k in keys) / max(1, n_ref)

    def empty_rule(k):
        return S[k]["hyp"] if not P[k]["hyp"].strip() else P[k]["hyp"]

    def short_rule(k):
        p, s = P[k]["hyp"].strip(), S[k]["hyp"].strip()
        return S[k]["hyp"] if (not p or len(p) < 0.5 * len(s)) else P[k]["hyp"]

    def oracle(k):
        return P[k]["hyp"] if edits(P[k]["ref"], P[k]["hyp"]) <= edits(S[k]["ref"], S[k]["hyp"]) else S[k]["hyp"]

    empties = sum(1 for k in keys if not P[k]["hyp"].strip())
    print(f"{len(keys)} images ({a.tag or 'untagged'}); primary returned empty on {empties}\n")
    print("| rule | CER % |")
    print("|---|---|")
    print(f"| {a.primary} alone | {total(lambda k: P[k]['hyp']):.1f} |")
    print(f"| {a.secondary} alone | {total(lambda k: S[k]['hyp']):.1f} |")
    print(f"| primary, secondary if empty | {total(empty_rule):.1f} |")
    print(f"| primary, secondary if empty or under half the length | {total(short_rule):.1f} |")
    print(f"| oracle (best of both per image; upper bound) | {total(oracle):.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

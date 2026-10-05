"""OCR engine adapters for the Phase 0 bake-off.

One interface: ``engine.read(img_bgr, kind) -> str`` where ``kind`` is
"line" (one text line), "block" (a few lines) or "page" (a whole page).
Multi-line output is joined in right-to-left reading order with spaces; the
scorer collapses whitespace anyway.

Engines are deliberately thin. The point is to compare recognisers on the same
pixels, not to re-implement them: Tesseract is driven through its CLI exactly
as mubsir does, PP-OCR through mubsir's ``ArabicRecognizer`` (including its
word-segmentation fix for long Arabic lines), and the hybrid engine is mubsir's
own, unchanged.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MUBSIR_ROOT") or os.path.dirname(os.path.dirname(HERE))   # a checkout of github.com/dusk-futile/mubsir
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(HERE, "data")
V5_DIR = os.path.join(DATA, "models", "ppocrv5")
V4_DIR = os.path.join(ROOT, "models", "arabic_v4")


def auto_upscale(img: np.ndarray, kind: str) -> np.ndarray:
    """Make small text big enough for a recogniser trained near 300 dpi.

    Screen text is typically 9-12 px tall. Tesseract's LSTM wants roughly 30-40
    px x-height, so a small line is enlarged until the whole line is ~70 px.
    """
    h, w = img.shape[:2]
    if kind == "line":
        s = int(np.clip(round(70 / max(h, 1)), 1, 4))
    elif kind == "block":
        s = 2 if h < 250 else 1
    else:                                   # page
        s = 3 if w < 900 else (2 if w < 1400 else 1)
    if s == 1:
        return img
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def pad(img: np.ndarray, px: int = 12) -> np.ndarray:
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=(255, 255, 255))


class Engine:
    name = "base"
    load_seconds = 0.0

    def read(self, img_bgr: np.ndarray, kind: str) -> str:
        raise NotImplementedError


class Tesseract(Engine):
    """Tesseract 5 `ara` (tessdata_best) through the CLI; psm chosen from the image kind."""

    PSM = {"line": 7, "block": 6, "page": 3}

    def __init__(self, upscale: bool, langs: str = "ara", name: Optional[str] = None,
                 polarity: bool = False, line_psm: int = 7):
        from mubsir.ocr.tesseract import find_binary, find_tessdata
        t = time.time()
        self.binary = find_binary()
        self.tessdata = find_tessdata(langs)
        if not self.binary or not self.tessdata:
            raise RuntimeError("Tesseract or tessdata not found")
        self.langs = langs
        self.upscale = upscale
        self.polarity = polarity
        self.PSM = dict(self.PSM, line=line_psm)
        self.name = name or ("tess_up" if upscale else "tess")
        self.load_seconds = time.time() - t

    def read(self, img_bgr: np.ndarray, kind: str) -> str:
        img = auto_upscale(img_bgr, kind) if self.upscale else img_bgr
        if self.polarity:
            img = normalize_polarity(img)
        img = pad(img)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "in.png")
            cv2.imwrite(path, img)
            env = dict(os.environ, TESSDATA_PREFIX=os.path.abspath(self.tessdata))
            p = subprocess.run([self.binary, path, "stdout", "-l", self.langs,
                                "--psm", str(self.PSM[kind])],
                               capture_output=True, env=env, timeout=300)
        return p.stdout.decode("utf-8", "replace")


_KEEP_RUN = re.compile(r"[a-zA-Z0-9 :*./%+-]")


def pred_reverse(text: str) -> str:
    """PaddleOCR's own fix for Arabic recognisers (CTCLabelDecode.pred_reverse).

    The PP-OCRv5 Arabic model emits characters in visual (left-to-right) order.
    Reversing gives logical order, except that runs of digits and Latin letters
    must keep their own direction, so they are treated as single tokens.
    """
    tokens: List[str] = []
    run = ""
    for c in text:
        if _KEEP_RUN.search(c):
            run += c
        else:
            if run:
                tokens.append(run)
                run = ""
            tokens.append(c)
    if run:
        tokens.append(run)
    return "".join(tokens[::-1])


class PPOCR(Engine):
    """DBNet line detection + word-split PP-OCR recognition (mubsir's recogniser)."""

    def __init__(self, name: str, rec_model: str, dict_path: str, det_model: Optional[str] = None,
                 reverse: bool = False, split: bool = True):
        from mubsir.ocr.ppocr_arabic import ArabicRecognizer
        from rapidocr_onnxruntime import RapidOCR
        t = time.time()
        self.name = name
        self.reverse = reverse
        self.split = split
        self.det = RapidOCR(det_model_path=det_model) if det_model else RapidOCR()
        self.rec = ArabicRecognizer(rec_model, dict_path)
        self._check_dictionary(rec_model)
        self.load_seconds = time.time() - t

    def _check_dictionary(self, rec_model: str) -> None:
        # A dictionary that does not match the model's output width silently
        # produces wrong characters, so fail loudly instead.
        n_out = self.rec.sess.get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != len(self.rec.chars):
            raise RuntimeError(
                f"{rec_model}: model has {n_out} classes but the dictionary gives "
                f"{len(self.rec.chars)} (blank + {len(self.rec.chars) - 2} chars + space)")

    def _detect(self, img: np.ndarray):
        res, _ = self.det(img, use_det=True, use_cls=False, use_rec=False)
        out = []
        H, W = img.shape[:2]
        for b in res or []:
            a = np.array(b, dtype=np.float32)
            x0, y0 = int(max(0, a[:, 0].min())), int(max(0, a[:, 1].min()))
            x1, y1 = int(min(W, a[:, 0].max())), int(min(H, a[:, 1].max()))
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                out.append((x0, y0, x1, y1))
        return out

    def _read_line(self, crop: np.ndarray) -> str:
        from mubsir.ocr.ppocr_arabic import segment_words
        words = (segment_words(crop) if self.split else None) or [(0, crop.shape[1])]
        pieces = self.rec.recognize([crop[:, a:b] for a, b in words])
        if self.reverse:
            pieces = [(pred_reverse(t), c) for t, c in pieces]
        ordered = [t for t, _ in reversed(pieces) if t.strip()]     # right-to-left
        return " ".join(ordered)

    def read(self, img_bgr: np.ndarray, kind: str) -> str:
        if kind == "line":
            return self._read_line(pad(img_bgr, 6))
        boxes = self._detect(img_bgr)
        boxes.sort(key=lambda b: (b[1] // max(1, (b[3] - b[1]) // 2), -b[2]))
        out = []
        for x0, y0, x1, y1 in boxes:
            p = max(1, int((y1 - y0) * 0.06))
            crop = img_bgr[max(0, y0 - p):y1 + p, x0:x1]
            text = self._read_line(crop)
            if text:
                out.append(text)
        return " ".join(out)


# ----------------------------------------------------------------------------
# v2 recogniser: fixes found by the screen-text bake-off
#   1. polarity: PP-OCR and mubsir's word splitter both assume dark text on a
#      light background, so light-on-dark lines came back (almost) empty.
#   2. bidi: recognised pieces arrive in visual left-to-right order; they must
#      be put back in logical order for the line's base direction, with runs of
#      English words (e.g. "Microsoft Word") kept in their own order.
#   3. chunking: cutting at EVERY gap (mubsir's per-word split) occasionally
#      cuts inside a word at small sizes. Cutting only at the widest gap near
#      evenly spaced targets keeps chunks under a maximum aspect ratio with far
#      fewer cuts.
# ----------------------------------------------------------------------------
_AR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN = re.compile(r"[A-Za-z]")


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def normalize_polarity(img: np.ndarray) -> np.ndarray:
    """Return dark-on-light. The ink is the minority class: if it is the bright one, invert."""
    g = to_gray(img)
    _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - img if (mask > 0).mean() < 0.5 else img


def visual_to_logical(texts: List[str], base_rtl: bool) -> List[str]:
    """Pieces in visual left-to-right order -> logical order (the NVDA approach for Windows OCR).

    RTL line: reverse everything, then flip each run of non-Arabic pieces back so
    English phrases keep their own left-to-right order. LTR line: reverse only
    the runs of Arabic pieces.
    """
    flags = [bool(_AR.search(t)) for t in texts]
    seq = list(zip(texts, flags))
    if base_rtl:
        seq = seq[::-1]
    out: List[str] = []
    i = 0
    flip_when = (lambda f: not f) if base_rtl else (lambda f: f)   # runs that must be re-reversed
    while i < len(seq):
        if flip_when(seq[i][1]):
            j = i
            while j < len(seq) and flip_when(seq[j][1]):
                j += 1
            out.extend(t for t, _ in reversed(seq[i:j]))
            i = j
        else:
            out.append(seq[i][0])
            i += 1
    return out


def _gaps(gray: np.ndarray):
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    col = (bw > 0).sum(axis=0)
    runs, start = [], None
    for i, v in enumerate(col):
        if v == 0:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(col)))
    return [(a, b) for a, b in runs if a > 0 and b < len(col)], bw


def chunk_segments(crop: np.ndarray, max_aspect: float):
    """Split a dark-on-light line into the fewest chunks of width <= max_aspect * height,
    cutting at the widest gap near each evenly spaced target."""
    h, w = crop.shape[:2]
    if w <= max_aspect * h:
        return [(0, w)]
    gaps, bw = _gaps(to_gray(crop))
    floor = max(2.0, h * 0.06)
    gaps = [g for g in gaps if g[1] - g[0] >= floor]
    if not gaps:
        return [(0, w)]
    n = int(np.ceil(w / (max_aspect * h)))
    span = w / n
    cuts = []
    for k in range(1, n):
        target = k * span
        near = [g for g in gaps if abs((g[0] + g[1]) / 2 - target) <= 0.3 * span]
        pick = (max(near, key=lambda g: g[1] - g[0]) if near
                else min(gaps, key=lambda g: abs((g[0] + g[1]) / 2 - target)))
        cuts.append((pick[0] + pick[1]) // 2)
    edges = [0] + sorted(set(cuts)) + [w]
    segs = [(a, b) for a, b in zip(edges, edges[1:]) if b - a >= 6 and (bw[:, a:b] > 0).sum() >= 20]
    return segs or [(0, w)]


class PPOCR2(PPOCR):
    """PPOCR with polarity normalisation, bidi ordering and chunked (or per-word) splitting."""

    def __init__(self, name: str, rec_model: str, dict_path: str, reverse: bool, mode: str = "chunk",
                 max_aspect: float = 8.0, polarity: bool = True):
        super().__init__(name, rec_model, dict_path, reverse=reverse)
        self.mode, self.max_aspect, self.polarity = mode, max_aspect, polarity

    def read(self, img_bgr: np.ndarray, kind: str) -> str:
        if kind == "line":
            return self._read_line(img_bgr, pad_px=6)      # normalise polarity BEFORE padding
        return super().read(img_bgr, kind)

    def _read_line(self, crop: np.ndarray, pad_px: int = 0) -> str:
        from mubsir.ocr.ppocr_arabic import segment_words
        if self.polarity:
            crop = normalize_polarity(crop)
        if pad_px:
            crop = pad(crop, pad_px)
        segs = segment_words(crop) if self.mode == "word" else chunk_segments(crop, self.max_aspect)
        segs = segs or [(0, crop.shape[1])]
        texts = [t for t, _ in self.rec.recognize([crop[:, a:b] for a, b in segs])]
        if self.reverse:    # visual -> logical inside each piece, only where there is Arabic to reverse
            texts = [pred_reverse(t) if _AR.search(t) else t for t in texts]
        texts = [t for t in texts if t.strip()]
        ar = sum(len(_AR.findall(t)) for t in texts)
        la = sum(len(_LATIN.findall(t)) for t in texts)
        return " ".join(visual_to_logical(texts, base_rtl=ar >= la))


class MubsirHybrid(Engine):
    """mubsir's own hybrid (DBNet lines + stacked Tesseract), unchanged."""

    name = "mubsir_hybrid"

    def __init__(self):
        from mubsir.ocr.hybrid import HybridEngine
        t = time.time()
        self.engine = HybridEngine()
        self.load_seconds = time.time() - t

    def read(self, img_bgr: np.ndarray, kind: str) -> str:
        from mubsir.lines import merge_fragments
        from mubsir.reading_order import order_lines
        lines = merge_fragments(self.engine.page_lines(img_bgr, 1, 1.0))
        lines = order_lines(lines, float(img_bgr.shape[1]), rtl=True)
        return " ".join(l.text for l in lines)


# ----------------------------------------------------------------------------
# Cloud vision models. Keys come from the environment (or bench/.env,
# which is git-ignored), never from code. Each call records token usage so the
# bake-off can report dollars per sample.
# ----------------------------------------------------------------------------
# USD per 1M tokens (input, output), official pages, Oct 2026.
PRICES = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "qwen3-vl-flash": (0.05, 0.40),
    "qwen3.8-flash": (0.15, 0.47),
}

TRANSCRIBE_PROMPT = ("Transcribe all text in the image exactly as written, in reading order. "
                     "Arabic is right-to-left. Output the text only: no commentary, no markdown.")


def _env(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val:
        return val
    path = os.path.join(HERE, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            k, _, v = line.strip().partition("=")
            if k == name and v:
                return v.strip().strip('"').strip("'")
    return None


def _png_b64(img: np.ndarray, max_side: int = 1280) -> str:
    import base64
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 90) -> dict:
    import json as _json
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=_json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return _json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 503):
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")
    raise RuntimeError(f"giving up: {last}")


class GeminiVision(Engine):
    """Gemini via the public REST API. Thinking is minimised to keep output tokens (and cost) low."""

    def __init__(self, model: str, thinking: Optional[dict] = None, media_resolution: Optional[str] = None):
        self.key = _env("GEMINI_API_KEY")
        if not self.key:
            raise RuntimeError("set GEMINI_API_KEY in the environment or bench/.env")
        self.model, self.thinking, self.media_resolution = model, thinking, media_resolution
        self.name = model
        self.last_usage: dict = {}

    def read(self, img_bgr: np.ndarray, kind: str) -> str:
        cfg: dict = {"temperature": 0, "maxOutputTokens": 2048}
        if self.thinking is not None:
            cfg["thinkingConfig"] = self.thinking
        if self.media_resolution:
            cfg["mediaResolution"] = self.media_resolution
        payload = {
            "systemInstruction": {"parts": [{"text": TRANSCRIBE_PROMPT}]},
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "image/png", "data": _png_b64(img_bgr)}}]}],
            "generationConfig": cfg,
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        out = _post_json(url, payload, {"x-goog-api-key": self.key})
        u = out.get("usageMetadata", {})
        self.last_usage = {"in": u.get("promptTokenCount", 0),
                           "out": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)}
        parts = (out.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)


class OpenAICompatVision(Engine):
    """Any OpenAI-compatible chat endpoint with image input (Qwen on DashScope, OpenRouter, ...)."""

    def __init__(self, model: str, base_url: str, key_env: str):
        self.key = _env(key_env)
        if not self.key:
            raise RuntimeError(f"set {key_env} in the environment or bench/.env")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.name = model
        self.last_usage: dict = {}

    def read(self, img_bgr: np.ndarray, kind: str) -> str:
        payload = {
            "model": self.model, "temperature": 0, "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": TRANSCRIBE_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64," + _png_b64(img_bgr)}}]}],
        }
        out = _post_json(self.base_url + "/chat/completions", payload,
                         {"Authorization": f"Bearer {self.key}"})
        u = out.get("usage", {})
        self.last_usage = {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
        return out["choices"][0]["message"]["content"] or ""


class MubsirHybridUp(MubsirHybrid):
    """The same hybrid, but fed upscaled low-resolution input (a fairer test on 75 dpi pages)."""

    name = "mubsir_hybrid_up"

    def read(self, img_bgr: np.ndarray, kind: str) -> str:
        return super().read(auto_upscale(img_bgr, kind), kind)


def make(name: str) -> Engine:
    if name == "mubsir_hybrid_up":
        return MubsirHybridUp()
    if name == "gemini_25_flash_lite":
        return GeminiVision("gemini-2.5-flash-lite")
    if name == "gemini_35_flash_lite":
        return GeminiVision("gemini-3.5-flash-lite", thinking={"thinkingLevel": "minimal"},
                            media_resolution="MEDIA_RESOLUTION_MEDIUM")
    if name == "gemini_38_flash":
        return GeminiVision("gemini-3.8-flash", thinking={"thinkingLevel": "low"},
                            media_resolution="MEDIA_RESOLUTION_MEDIUM")
    if name == "qwen_vl_flash":
        return OpenAICompatVision("qwen3-vl-flash",
                                  "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                                  "DASHSCOPE_API_KEY")
    if name == "tess":
        return Tesseract(upscale=False)
    if name == "tess_up":
        return Tesseract(upscale=True)
    if name == "tess_ara_eng_up":
        return Tesseract(upscale=True, langs="ara+eng", name="tess_ara_eng_up")
    if name == "tess_eng_up":
        return Tesseract(upscale=True, langs="eng", name="tess_eng_up")
    if name == "ppocr_v4":
        return PPOCR("ppocr_v4", os.path.join(V4_DIR, "model.onnx"),
                     os.path.join(V4_DIR, "arabic_dict.txt"))
    if name == "ppocr_v5":
        return PPOCR("ppocr_v5", os.path.join(V5_DIR, "arabic_pp-ocrv5_mobile_rec.onnx"),
                     os.path.join(V5_DIR, "ppocrv5_arabic_dict.txt"), reverse=True)
    if name == "ppocr_v5_det5":
        return PPOCR("ppocr_v5_det5", os.path.join(V5_DIR, "arabic_pp-ocrv5_mobile_rec.onnx"),
                     os.path.join(V5_DIR, "ppocrv5_arabic_dict.txt"),
                     det_model=os.path.join(V5_DIR, "pp-ocrv5_mobile_det.onnx"), reverse=True)
    if name == "tess_ara_eng_up_pol_p13":
        return Tesseract(upscale=True, langs="ara+eng", name=name, polarity=True, line_psm=13)
    if name == "tess_ara_up_pol_p13":
        return Tesseract(upscale=True, langs="ara", name=name, polarity=True, line_psm=13)
    if name == "tess_ara_eng_up_pol":
        return Tesseract(upscale=True, langs="ara+eng", name=name, polarity=True)
    if name.startswith("v2_"):               # v2_<v4|v5>_<word|chunkN>
        _, ver, mode = name.split("_")
        rec = (os.path.join(V5_DIR, "arabic_pp-ocrv5_mobile_rec.onnx"), os.path.join(V5_DIR, "ppocrv5_arabic_dict.txt"), True) \
            if ver == "v5" else (os.path.join(V4_DIR, "model.onnx"), os.path.join(V4_DIR, "arabic_dict.txt"), False)
        if mode == "word":
            return PPOCR2(name, rec[0], rec[1], reverse=rec[2], mode="word")
        return PPOCR2(name, rec[0], rec[1], reverse=rec[2], mode="chunk", max_aspect=float(mode[5:]))
    if name == "ppocr_v4_nosplit":
        return PPOCR("ppocr_v4_nosplit", os.path.join(V4_DIR, "model.onnx"),
                     os.path.join(V4_DIR, "arabic_dict.txt"), split=False)
    if name == "ppocr_v5_nosplit":
        return PPOCR("ppocr_v5_nosplit", os.path.join(V5_DIR, "arabic_pp-ocrv5_mobile_rec.onnx"),
                     os.path.join(V5_DIR, "ppocrv5_arabic_dict.txt"), reverse=True, split=False)
    if name == "mubsir_hybrid":
        return MubsirHybrid()
    raise ValueError(f"unknown engine {name!r}")


ENGINES = ["tess", "tess_up", "ppocr_v4", "ppocr_v5", "ppocr_v5_det5", "mubsir_hybrid"]

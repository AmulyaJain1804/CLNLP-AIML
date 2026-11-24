"""Text normalization, script detection and filtering helpers (M2)."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_WS = re.compile(r"\s+")


def normalize(text: str, cfg: dict) -> str:
    text = unicodedata.normalize(cfg.get("unicode_normalization", "NFC"), text)
    for ch in cfg.get("remove_chars", []):
        text = text.replace(ch, "")
    if cfg.get("collapse_whitespace", True):
        text = _WS.sub(" ", text).strip()
    return text


def char_script(ch: str) -> str | None:
    if not ch.isalpha() and unicodedata.category(ch) not in ("Mn", "Mc"):
        return None
    try:
        return unicodedata.name(ch).split(" ")[0]
    except ValueError:
        return None


def script_profile(lines) -> Counter:
    c = Counter()
    for line in lines:
        for ch in line:
            s = char_script(ch)
            if s:
                c[s] += 1
    return c


def script_fraction(text: str, script: str) -> float:
    letters = [s for s in (char_script(ch) for ch in text) if s]
    if not letters:
        return 0.0
    return sum(s == script for s in letters) / len(letters)


def broken_marks(text: str) -> int:
    """Combining marks (Mn/Mc) that are not attached to a base character:
    at string start or right after whitespace/punctuation. Normalization must not increase this."""
    n = 0
    prev = " "
    for ch in text:
        if unicodedata.category(ch) in ("Mn", "Mc"):
            if prev.isspace() or unicodedata.category(prev)[0] in ("P", "Z", "S"):
                n += 1
        prev = ch
    return n


def filter_reason(src: str, tgt: str, f: dict, tgt_script: str | None) -> str | None:
    if not src or not tgt:
        return "empty"
    ws, wt = len(src.split()), len(tgt.split())
    if min(ws, wt) < f["min_words"]:
        return "too_short"
    if max(ws, wt) > f["max_words"]:
        return "too_long"
    ls, lt = len(src), len(tgt)
    if min(ls, lt) >= f["ratio_min_chars"] and max(ls, lt) / min(ls, lt) > f["max_char_ratio"]:
        return "length_ratio"
    if f.get("drop_identical") and src.lower() == tgt.lower():
        return "identical"
    if tgt_script and tgt_script != "LATIN" and script_fraction(tgt, tgt_script) < f["min_script_frac"]:
        return "wrong_script"
    return None

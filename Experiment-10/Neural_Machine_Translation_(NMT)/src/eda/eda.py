"""Stage 2 EDA (M2). Figures -> reports/figures/eda/ (PNG + PDF), summary -> reports/eda_summary.md

Usage:  python -m src.eda.eda
"""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import sentencepiece as spm

from src.utils import FIGURES, PROCESSED, REPORTS, pairs_config, read_lines, selected_pairs
from src.viz.style import FIGSIZE_WIDE, INK2, SIDE_COLORS, SLOTS, plt, save

OUT = FIGURES / "eda"
_WORD = re.compile(r"\w+", re.UNICODE)


def words(line: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(line) if not w.isdigit()]


def load(pair: str, split: str):
    x = pair.split("-", 1)[1]
    return read_lines(PROCESSED / pair / f"{split}.en"), read_lines(PROCESSED / pair / f"{split}.{x}")


def fig_split_sizes(cfg) -> None:
    pairs = list(cfg["pairs"])
    stats = {p: json.loads((PROCESSED / p / "stats.json").read_text(encoding="utf-8")) for p in pairs}
    pairs.sort(key=lambda p: -stats[p]["splits"]["train"])
    fig, ax = plt.subplots(figsize=(14, 5))
    xs = np.arange(len(pairs))
    w = 0.27
    for i, (split, color) in enumerate([("train", SLOTS[0]), ("dev", SLOTS[1]), ("test", SLOTS[2])]):
        ax.bar(xs + (i - 1) * w, [stats[p]["splits"][split] for p in pairs], w * 0.92,
               color=color, label=split)
    ax.set_yscale("log")
    names = {"en-mni": "Manipuri (Beng.)", "en-mni-Mtei": "Manipuri (Mtei)"}
    ax.set_xticks(xs, [f"{p}{' *' if cfg['pairs'][p].get('selected') else ''}\n"
                       f"{names.get(p, cfg['pairs'][p]['name'])}\nCat. {cfg['pairs'][p]['category']}"
                       for p in pairs], fontsize=8)
    ax.set_ylabel("Sentence pairs (log scale)")
    ax.set_title("Train / dev / test size per language pair after cleaning (* = modelled)")
    ax.legend(ncols=3, loc="upper right")
    save(fig, OUT / "split_sizes")


def _hist_grid(pairs, fn, xlabel, title, name, bins=60, clip=None):
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.6), sharey=False)
    for ax, pair in zip(np.atleast_1d(axes), pairs):
        x = pair.split("-", 1)[1]
        en, xx = load(pair, "train")
        a, b = fn(pair, en, "en"), fn(pair, xx, x)
        hi = clip or np.percentile(np.concatenate([a, b]), 99)
        edges = np.linspace(0, hi, bins)
        ax.hist(np.clip(a, 0, hi), edges, color=SIDE_COLORS["en"], alpha=0.55, label="English")
        ax.hist(np.clip(b, 0, hi), edges, color=SIDE_COLORS["x"], alpha=0.55, label=x)
        ax.set_title(pair, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Sentences")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save(fig, OUT / name)


_SP = {}


def sp_for(pair):
    if pair not in _SP:
        _SP[pair] = spm.SentencePieceProcessor(model_file=str(PROCESSED / pair / "spm.model"))
    return _SP[pair]


def fig_lengths(pairs) -> dict:
    _hist_grid(pairs, lambda p, L, lang: np.array([len(s) for s in L]), "Characters per sentence",
               "Sentence length distribution (characters, train)", "length_chars")
    _hist_grid(pairs, lambda p, L, lang: np.array([len(t) for t in sp_for(p).encode(L)]),
               "SentencePiece tokens per sentence", "Sentence length distribution (subword tokens, train)",
               "length_tokens")
    # ratio
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.6))
    ratios = {}
    for ax, pair in zip(np.atleast_1d(axes), pairs):
        en, xx = load(pair, "train")
        r = np.array([len(b) / max(1, len(a)) for a, b in zip(en, xx)])
        ratios[pair] = {"median": round(float(np.median(r)), 3), "p5": round(float(np.percentile(r, 5)), 3),
                        "p95": round(float(np.percentile(r, 95)), 3)}
        ax.hist(np.clip(r, 0, 3), np.linspace(0, 3, 61), color=SLOTS[0])
        ax.axvline(np.median(r), color=INK2, ls="--", lw=1)
        ax.text(np.median(r), ax.get_ylim()[1] * 0.92, f" median {np.median(r):.2f}", fontsize=8, color=INK2)
        ax.set_title(pair, fontsize=11)
        ax.set_xlabel(f"len({pair.split('-', 1)[1]}) / len(en)  [chars]")
    np.atleast_1d(axes)[0].set_ylabel("Sentence pairs")
    fig.suptitle("Target/source character-length ratio (train, clipped at 3)", fontweight="bold")
    fig.tight_layout()
    save(fig, OUT / "length_ratio")
    return ratios


def fig_oov(pairs) -> dict:
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    labels, en_oov, x_oov = [], [], []
    for i, pair in enumerate(pairs):
        x = pair.split("-", 1)[1]
        res[pair] = {}
        for side, (tr, dv) in (("en", (load(pair, "train")[0], load(pair, "dev")[0])),
                               (x, (load(pair, "train")[1], load(pair, "dev")[1]))):
            cnt = Counter(w for s in tr for w in words(s))
            dev_tokens = [w for s in dv for w in words(s)]
            oov = sum(w not in cnt for w in dev_tokens) / max(1, len(dev_tokens))
            res[pair][side] = {"train_types": len(cnt), "dev_oov_token_rate": round(oov, 4)}
            if side != "en":
                ranked = [w for w, _ in cnt.most_common()]
                ks = np.unique(np.logspace(1, np.log10(max(10, len(ranked))), 40).astype(int))
                cov = [sum(w in set(ranked[:k]) for w in dev_tokens) / len(dev_tokens) for k in ks]
                axes[1].plot(ks, cov, color=SLOTS[i], label=f"{x} ({pair})")
        labels.append(pair)
        en_oov.append(res[pair]["en"]["dev_oov_token_rate"] * 100)
        x_oov.append(res[pair][x]["dev_oov_token_rate"] * 100)
    xs = np.arange(len(labels))
    axes[0].bar(xs - 0.2, en_oov, 0.38, color=SIDE_COLORS["en"], label="English side")
    axes[0].bar(xs + 0.2, x_oov, 0.38, color=SIDE_COLORS["x"], label="Indic side")
    for xi, (a, b) in enumerate(zip(en_oov, x_oov)):
        axes[0].text(xi - 0.2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
        axes[0].text(xi + 0.2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(xs, labels)
    axes[0].set_ylabel("Dev word tokens unseen in train (%)")
    axes[0].set_title("Word-level OOV rate on dev")
    axes[0].legend()
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Top-k most frequent train word types kept")
    axes[1].set_ylabel("Dev token coverage (Indic side)")
    axes[1].set_title("Vocabulary coverage curve (Indic side)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, OUT / "oov_coverage")
    return res


def fig_zipf(pairs) -> None:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)
    for i, pair in enumerate(pairs):
        en, xx = load(pair, "train")
        for ax, lines, lab in ((axes[0], en, f"en ({pair})"), (axes[1], xx, pair.split("-", 1)[1])):
            f = np.array(sorted(Counter(w for s in lines for w in words(s)).values(), reverse=True))
            ax.loglog(np.arange(1, len(f) + 1), f, color=SLOTS[i], label=lab, lw=1.6)
    for ax, t in zip(axes, ("English sides", "Indic sides")):
        ax.set_xlabel("Rank (log)")
        ax.set_title(f"Zipf plot — {t} (train words)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Frequency (log)")
    fig.tight_layout()
    save(fig, OUT / "zipf")


STOP_EN = set("the of and to in a is that for on with as by was are it be this from at or an his he "
              "they which not have will their i you all were has but its had them him shall unto".split())


def ngram_tables(pairs) -> str:
    out = ["## Top n-grams per language (train)", "",
           "Produced by lower-casing, extracting `\\w+` word tokens (Unicode-aware, digits removed) and",
           "counting unigrams (English stop-words removed) and bigrams. Indic-script word clouds are not",
           "drawn because matplotlib/wordcloud cannot shape Bengali/Devanagari conjuncts correctly; tables",
           "are used instead. Word clouds are drawn for English and Latin-script Mizo.", ""]
    for pair in pairs:
        x = pair.split("-", 1)[1]
        en, xx = load(pair, "train")
        out.append(f"### {pair}")
        out.append("")
        out.append(f"| # | en unigram | en bigram | {x} unigram | {x} bigram |")
        out.append("|---|---|---|---|---|")
        cols = []
        for lines, stop in ((en, STOP_EN), (xx, set())):
            toks = [words(s) for s in lines]
            uni = Counter(w for t in toks for w in t if w not in stop and len(w) > 1)
            bi = Counter(f"{a} {b}" for t in toks for a, b in zip(t, t[1:]))
            cols += [[f"{w} ({c})" for w, c in uni.most_common(10)],
                     [f"{w} ({c})" for w, c in bi.most_common(10)]]
        for i in range(10):
            out.append(f"| {i + 1} | " + " | ".join(c[i] if i < len(c) else "" for c in cols) + " |")
        out.append("")
    return "\n".join(out)


def word_clouds(pairs) -> None:
    from wordcloud import WordCloud
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#1c5cab", "#2a78d6", "#256abf", "#eb6834", "#0d366b"])
    targets = [(f"en ({p})", load(p, "train")[0], STOP_EN) for p in pairs]
    targets += [(f"{p.split('-', 1)[1]} ({p})", load(p, "train")[1], set())
                for p in pairs if pairs_config()["pairs"][p]["script"] == "Latin"]
    fig, axes = plt.subplots(1, len(targets), figsize=(3.6 * len(targets), 3.2))
    for ax, (title, lines, stop) in zip(np.atleast_1d(axes), targets):
        freq = Counter(w for s in lines for w in words(s) if w not in stop and len(w) > 2)
        wc = WordCloud(width=600, height=450, background_color="white", colormap=cmap,
                       random_state=42, max_words=80).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Word clouds (train, stop-words removed; Latin-script sides only)", fontweight="bold")
    fig.tight_layout()
    save(fig, OUT / "wordclouds")


def main() -> None:
    cfg = pairs_config()
    pairs = selected_pairs()
    fig_split_sizes(cfg)
    ratios = fig_lengths(pairs)
    oov = fig_oov(pairs)
    fig_zipf(pairs)
    word_clouds(pairs)

    st = {p: json.loads((PROCESSED / p / "stats.json").read_text(encoding="utf-8")) for p in cfg["pairs"]}
    md = ["# EDA summary (auto-generated by `python -m src.eda.eda`)", "",
          "## Data after cleaning (all 11 pairs)", "",
          "| Pair | Language | Cat. | Train | Dev | Test | Removed from train | Train∩test overlap removed | X script (top) | SPM vocab | avg tok/sent en / X |",
          "|---|---|---|---:|---:|---:|---|---:|---|---:|---|"]
    for p, s in st.items():
        x = p.split("-", 1)[1]
        rem = ", ".join(f"{k}: {v}" for k, v in s["filters"]["train_removed"].items())
        scr = next(iter(s["scripts"][x]), "-")
        tok = s["spm"]["avg_tokens_per_sentence_train"]
        md.append(f"| {p}{' *' if cfg['pairs'][p].get('selected') else ''} | {cfg['pairs'][p]['name']} | "
                  f"{cfg['pairs'][p]['category']} | {s['splits']['train']} | {s['splits']['dev']} | "
                  f"{s['splits']['test']} | {rem} | {s['filters']['train_test_overlap_removed']} | {scr} | "
                  f"{s['spm']['vocab_size']} | {tok['en']} / {tok[x]} |")
    md += ["", "\\* = pairs modelled in this project.", "",
           "## Script integrity", "",
           "Broken combining marks (a combining vowel sign/virama not attached to a base letter) on the",
           "Indic side, before vs after NFC + whitespace normalisation — normalisation must not increase it:", "",
           "| Pair | before | after |", "|---|---:|---:|"]
    for p, s in st.items():
        b = s["broken_combining_marks"]
        md.append(f"| {p} | {b['before_norm']} | {b['after_norm']} |")
    md += ["", "## Length ratio (chars, X / en, train)", "", "| Pair | median | p5 | p95 |", "|---|---:|---:|---:|"]
    md += [f"| {p} | {r['median']} | {r['p5']} | {r['p95']} |" for p, r in ratios.items()]
    md += ["", "## Word-level OOV on dev", "", "| Pair | side | train word types | dev OOV token rate |",
           "|---|---|---:|---:|"]
    for p, d in oov.items():
        for side, v in d.items():
            md.append(f"| {p} | {side} | {v['train_types']} | {v['dev_oov_token_rate']:.2%} |")
    md += ["", "Subword (SentencePiece) UNK rate on dev is ~0 for every pair because `character_coverage=1.0`.", "",
           "## Domain breakdown", "",
           "The release provides **no domain labels**, so no per-domain breakdown is possible. Manual",
           "inspection suggests a mix of Bible text (Mizo, Khasi), government/news text (Assamese,",
           "Manipuri) and general/educational text (Bodo).", "", ngram_tables(pairs)]
    (REPORTS / "eda_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("EDA figures ->", OUT, "| summary -> reports/eda_summary.md")


if __name__ == "__main__":
    main()

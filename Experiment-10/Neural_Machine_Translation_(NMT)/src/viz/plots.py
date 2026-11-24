"""M5 result figures (PNG + PDF in reports/figures/results/). Run after src.eval.evaluate.

  loss_curves_{pair}      train (solid) / dev (dashed) loss, both directions, all models overlaid
  dev_chrf_curves         dev chrF++ over training steps, one panel per direction
  test_chrf_bars          test chrF++ per direction and model, 95% bootstrap CI error bars
  test_bleu_bars          same for BLEU
  heatmap_chrf            models x directions test chrF++
  learning_curve          baseline test chrF++ vs training-set size (en-as, as-en)
  length_vs_score         sentence chrF++ vs source length (binned mean +- 95% CI, all directions)
  nllb_zeroshot_gain      NLLB dev chrF++ before vs after fine-tuning
  attention_{direction}   baseline cross-attention (last decoder layer, heads averaged) for one test sentence

Usage:  python -m src.viz.plots [--no-attention]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from src.eval.evaluate import MAIN_MODELS, RESULTS
from src.utils import FIGURES, OUTPUTS, PROCESSED, directions, selected_pairs, split_direction
from src.viz.style import FIGSIZE, FIGSIZE_WIDE, INK2, MODEL_COLORS, MODEL_LABELS, SEQ, SLOTS, plt, save

OUT = FIGURES / "results"


def read_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def train_log(model, direction):
    rows = read_csv(OUTPUTS / model / direction / "train_log.csv")
    return {k: [fnum(r[k]) for r in rows] for k in (rows[0].keys() if rows else [])}


def legend_below(fig, ax, ncol=3, rows=2):
    """Lay out the panels first, then put the shared legend in a reserved strip under them."""
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06 * rows, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0))


# ---------------------------------------------------------------- training curves
def loss_curves():
    for pair in selected_pairs():
        x = pair.split("-", 1)[1]
        dirs = [f"en-{x}", f"{x}-en"]
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)
        for ax, d in zip(axes, dirs):
            for m in MAIN_MODELS:
                lg = train_log(m, d)
                if not lg:
                    continue
                st = np.array(lg["step"])
                for key, ls, lab in (("train_loss", "-", "train"), ("dev_loss", "--", "dev")):
                    y = np.array([np.nan if v is None else v for v in lg[key]])
                    ok = ~np.isnan(y)
                    if ok.any():
                        ax.plot(st[ok], y[ok], ls, color=MODEL_COLORS[m],
                                label=f"{MODEL_LABELS[m]} ({lab})", marker="o" if ok.sum() < 6 else None, ms=4)
            ax.set_title(d)
            ax.set_xlabel("optimizer step")
        axes[0].set_ylabel("cross-entropy (label-smoothed)")
        fig.suptitle(f"Training and dev loss, en↔{x}")
        legend_below(fig, axes[0], ncol=3, rows=2)
        save(fig, OUT / f"loss_curves_{pair}")


def dev_chrf_curves():
    dirs = directions()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=False)
    for ax, d in zip(axes.flat, dirs):
        for m in MAIN_MODELS:
            lg = train_log(m, d)
            if not lg or "dev_chrf" not in lg:
                continue
            pts = [(s, c) for s, c in zip(lg["step"], lg["dev_chrf"]) if c is not None]
            if pts:
                s, c = zip(*pts)
                ax.plot(s, c, color=MODEL_COLORS[m], label=MODEL_LABELS[m], marker="o", ms=3)
        ax.set_title(d)
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("dev chrF++ (greedy)")
    fig.suptitle("Dev chrF++ during training")
    legend_below(fig, axes[0, 0], rows=0.7)
    save(fig, OUT / "dev_chrf_curves")


# ---------------------------------------------------------------- test results
def score_bars(rows, key, label, name):
    dirs = [d for d in directions() if any(r["direction"] == d for r in rows)]
    by = {(r["model"], r["direction"]): r for r in rows}
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    w = 0.26
    for j, m in enumerate(MAIN_MODELS):
        xs, ys, lo, hi = [], [], [], []
        for i, d in enumerate(dirs):
            r = by.get((m, d))
            if r and fnum(r.get(key)) is not None:
                v = fnum(r[key])
                xs.append(i + (j - 1) * w)
                ys.append(v)
                lo.append(v - fnum(r[f"{key}_lo"]))
                hi.append(fnum(r[f"{key}_hi"]) - v)
        ax.bar(xs, ys, w * 0.92, color=MODEL_COLORS[m], label=MODEL_LABELS[m],
               yerr=[lo, hi], error_kw={"elinewidth": 1, "capsize": 2, "ecolor": INK2})
    for i, d in enumerate(dirs):
        if (MAIN_MODELS[2], d) not in by:
            ax.text(i + w, 1, "n/a", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(dirs)), dirs)
    ax.set_ylabel(f"test {label}")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0, max(fnum(r[f"{key}_hi"]) for r in rows) * 1.15)
    ax.set_title(f"Test {label} per direction (95% bootstrap CI)")
    save(fig, OUT / name)


def heatmap(rows):
    dirs = [d for d in directions() if any(r["direction"] == d for r in rows)]
    by = {(r["model"], r["direction"]): fnum(r["chrf++"]) for r in rows}
    M = np.array([[by.get((m, d), np.nan) for d in dirs] for m in MAIN_MODELS])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=SEQ, aspect="auto", vmin=np.nanmin(M) - 5, vmax=np.nanmax(M))
    ax.set_xticks(range(len(dirs)), dirs)
    ax.set_yticks(range(len(MAIN_MODELS)), [MODEL_LABELS[m] for m in MAIN_MODELS])
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt, col = ("n/a", INK2) if np.isnan(v) else (f"{v:.1f}", "white" if v > np.nanmean(M) + 3 else "#0b0b0b")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)
    fig.colorbar(im, ax=ax, label="test chrF++", fraction=0.03)
    ax.set_title("Test chrF++: models × directions")
    save(fig, OUT / "heatmap_chrf")


def learning_curve():
    rows = read_csv(RESULTS / "learning_curve.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for k, d in enumerate(sorted({r["direction"] for r in rows}, key=directions().index)):
        rr = sorted((r for r in rows if r["direction"] == d), key=lambda r: fnum(r["n_train"]))
        n = [fnum(r["n_train"]) for r in rr]
        y = [fnum(r["chrf++"]) for r in rr]
        ax.fill_between(n, [fnum(r["chrf++_lo"]) for r in rr], [fnum(r["chrf++_hi"]) for r in rr],
                        color=SLOTS[k], alpha=0.18, lw=0)
        ax.plot(n, y, color=SLOTS[k], marker="o", label=d)
        ax.annotate(f"{y[-1]:.1f}", (n[-1], y[-1]), textcoords="offset points", xytext=(6, -3), fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([5000, 10000, 25000, 50000], ["5k", "10k", "25k", "50k"])
    ax.set_xlabel("training sentence pairs (log scale)")
    ax.set_ylabel("test chrF++")
    ax.legend(title="direction")
    ax.set_title("Learning curve: Transformer baseline vs data size")
    save(fig, OUT / "learning_curve")


def length_vs_score():
    rows = read_csv(RESULTS / "sentence_scores.csv")
    if not rows:
        return
    edges = [0, 5, 10, 15, 20, 30, 40, 60, 1000]
    labels = ["1-5", "6-10", "11-15", "16-20", "21-30", "31-40", "41-60", "60+"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rng = np.random.default_rng(0)
    for m in MAIN_MODELS:
        r = [x for x in rows if x["model"] == m]
        if not r:
            continue
        L = np.array([int(x["src_words"]) for x in r])
        S = np.array([float(x["chrf++"]) for x in r])
        b = np.digitize(L, edges[1:-1], right=True)
        # faint per-sentence scatter (subsample, jittered) behind the binned means
        sub = rng.choice(len(L), size=min(1500, len(L)), replace=False)
        ax.scatter(b[sub] + rng.uniform(-0.3, 0.3, len(sub)) + (MAIN_MODELS.index(m) - 1) * 0.08, S[sub],
                   s=3, color=MODEL_COLORS[m], alpha=0.12, lw=0)
        mean, ci = [], []
        for k in range(len(labels)):
            v = S[b == k]
            mean.append(v.mean() if len(v) else np.nan)
            ci.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.errorbar(range(len(labels)), mean, yerr=ci, color=MODEL_COLORS[m], marker="o", ms=5,
                    capsize=2, label=MODEL_LABELS[m])
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("source sentence length (words)")
    ax.set_ylabel("sentence chrF++")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.set_title("Sentence length vs quality (all directions, test)")
    save(fig, OUT / "length_vs_score")


def nllb_zeroshot():
    pts = []
    for d in directions():
        f = OUTPUTS / "nllb_finetune" / d / "run_config.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("finished"):
                pts.append((d, r["zeroshot_dev_chrf"], r["dev_chrf_beam"]))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (d, z, f) in enumerate(pts):
        ax.plot([i, i], [z, f], color=INK2, lw=1.5, zorder=1)
        ax.scatter([i], [z], s=60, color="#ffffff", edgecolor=MODEL_COLORS["nllb_finetune"], lw=2, zorder=2,
                   label="zero-shot (dev subset, greedy)" if i == 0 else None)
        ax.scatter([i], [f], s=60, color=MODEL_COLORS["nllb_finetune"], zorder=2,
                   label="fine-tuned (dev, beam)" if i == 0 else None)
        ax.annotate(f"+{f - z:.1f}", (i, f), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    ax.set_xticks(range(len(pts)), [p[0] for p in pts])
    ax.set_ylabel("dev chrF++")
    ax.set_ylim(0, max(p[2] for p in pts) + 10)
    ax.legend(loc="lower right")
    ax.set_title("NLLB-200: zero-shot vs fine-tuned (Bodo uses the hin_Deva proxy tag)")
    save(fig, OUT / "nllb_zeroshot_gain")


# ---------------------------------------------------------------- attention
def attention(direction="lus-en", idx=None, max_tokens=22):
    import torch

    from src.models.baseline.data import SPTokenizer, pad_batch
    from src.models.baseline.model import BOS, greedy_decode
    from src.models.baseline.train import load_model
    from src.utils import read_lines

    ck = OUTPUTS / "baseline" / direction / "checkpoint_best.pt"
    if not ck.exists():
        print("no checkpoint for attention plot:", ck)
        return
    src_l, tgt_l, pair = split_direction(direction)
    tok = SPTokenizer(PROCESSED / pair / "spm.model")
    model, _ = load_model(ck, torch.device("cpu"))
    srcs = read_lines(PROCESSED / pair / f"test.{src_l}")
    if idx is None:  # a medium-length sentence the model translates well
        sc = [r for r in read_csv(RESULTS / "sentence_scores.csv") if r["model"] == "baseline" and r["direction"] == direction]
        good = [int(r["idx"]) for r in sc if 7 <= int(r["src_words"]) <= 11]
        good.sort(key=lambda i: -float(sc[i]["chrf++"]))
        idx = good[0] if good else 0
    ids = tok.encode([srcs[idx]])[0]
    src = pad_batch([ids])
    with torch.no_grad():
        hyp = greedy_decode(model, src)[0]
        hyp = [t for t in hyp if t > 3][:max_tokens]
        mem, mask = model.encode(src)
        tgt_in = torch.tensor([[BOS] + hyp])
        _, attn = model.decode(tgt_in, mem, mask, return_attn=True)
    # decoder input = [BOS] + hyp, so row t is the attention used to PREDICT hyp[t]
    A = attn[0].mean(0)[:len(hyp), :].float().numpy()
    sp = tok.sp
    xl = [sp.id_to_piece(i) for i in ids] + ["</s>"]
    yl = [sp.id_to_piece(i) for i in hyp]
    fig, ax = plt.subplots(figsize=(max(6, 0.42 * len(xl) + 2), max(4, 0.36 * len(yl) + 1.5)))
    im = ax.imshow(A, cmap=SEQ, aspect="auto", vmin=0, vmax=A.max())
    ax.set_xticks(range(len(xl)), xl, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(yl)), yl, fontsize=9)
    ax.grid(False)
    ax.set_xlabel(f"source subwords ({src_l})")
    ax.set_ylabel(f"generated subwords ({tgt_l})")
    fig.colorbar(im, ax=ax, label="attention weight", fraction=0.03)
    ax.set_title(f"Baseline cross-attention, {direction} test #{idx} (last decoder layer, mean of heads)")
    save(fig, OUT / f"attention_{direction}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-attention", action="store_true")
    a = ap.parse_args()
    rows = read_csv(RESULTS / "scores.csv")
    loss_curves()
    dev_chrf_curves()
    if rows:
        score_bars(rows, "chrf++", "chrF++", "test_chrf_bars")
        score_bars(rows, "bleu", "BLEU", "test_bleu_bars")
        if any(r.get("comet") for r in rows):
            score_bars([r for r in rows if r.get("comet")], "comet", "COMET", "test_comet_bars")
        heatmap(rows)
    learning_curve()
    length_vs_score()
    nllb_zeroshot()
    if not a.no_attention:
        attention("lus-en")
        attention("en-lus")
    print("figures in", OUT)


if __name__ == "__main__":
    main()

"""M5 evaluation: BLEU + chrF++ (sacreBLEU, with signatures), COMET (if comet.json exists),
bootstrap 95% CIs and paired-bootstrap significance against the baseline.

Reads   outputs/{model}/{direction}/test.hyp  + data/processed/{pair}/test.{tgt}
Writes  reports/results/scores.csv            one row per (model, direction): score + CI
        reports/results/significance.csv      paired bootstrap of every model vs. baseline
        reports/results/sentence_scores.csv   sentence-level chrF++ (for length analysis / errors)
        reports/results/learning_curve.csv    baseline at 10/25/50/100% of the data
        reports/results/results_table.md      markdown tables + metric signatures

Usage:  python -m src.eval.evaluate [--n-boot 1000] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

from src.utils import OUTPUTS, PROCESSED, REPORTS, directions, read_lines, split_direction

MAIN_MODELS = ["baseline", "backtranslation", "nllb_finetune"]
FRAC_MODELS = {"baseline_frac0.1": 0.1, "baseline_frac0.25": 0.25, "baseline_frac0.5": 0.5}
RESULTS = REPORTS / "results"


def metrics():
    # chrF++ = chrF with word bigrams (word_order=2), the WMT Indic primary metric
    return {"chrf++": CHRF(word_order=2), "bleu": BLEU(force=True)}  # refs are partly pre-tokenised (" ."); scored as released


class Scored:
    """Per-sentence sufficient statistics so that corpus scores of any resample are exact."""

    def __init__(self, hyps, refs, comet=None):
        self.m = metrics()
        self.stats = {k: np.asarray(m._extract_corpus_statistics(hyps, [refs]), dtype=np.float64)
                      for k, m in self.m.items()}
        self.comet = None if comet is None else np.asarray(comet, dtype=np.float64)
        self.n = len(hyps)

    def score(self, key, idx=None):
        if key == "comet":
            c = self.comet if idx is None else self.comet[idx]
            return float(c.mean() * 100)  # report COMET x100 like the WMT tables
        st = self.stats[key] if idx is None else self.stats[key][idx]
        return float(self.m[key]._compute_score_from_stats(list(st.sum(0))).score)

    def keys(self):
        return list(self.stats) + (["comet"] if self.comet is not None else [])

    def sentence_chrf(self):
        m = self.m["chrf++"]
        return [m._compute_score_from_stats(list(r)).score for r in self.stats["chrf++"]]


def load_system(model, direction):
    d = OUTPUTS / model / direction
    hyp_f = d / "test.hyp"
    if not hyp_f.exists():
        return None
    src_l, tgt_l, pair = split_direction(direction)
    refs = read_lines(PROCESSED / pair / f"test.{tgt_l}")
    hyps = read_lines(hyp_f)
    assert len(hyps) == len(refs), f"{model}/{direction}: {len(hyps)} hyps vs {len(refs)} refs"
    comet = None
    cf = d / "comet.json"
    if cf.exists():
        seg = json.loads(cf.read_text(encoding="utf-8"))["segment_scores"]
        comet = seg if len(seg) == len(hyps) else None
    return Scored(hyps, refs, comet)


def run(n_boot: int, seed: int):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, sig_rows, sent_rows, lc_rows = [], [], [], []
    signatures = {}
    for k, m in metrics().items():
        m.corpus_score(["x"], [["x"]])  # the signature needs one call to fix nrefs
        signatures[k] = m.get_signature().format()

    for direction in directions():
        src_l, tgt_l, pair = split_direction(direction)
        systems = {m: load_system(m, direction) for m in MAIN_MODELS}
        systems = {m: s for m, s in systems.items() if s is not None}
        if not systems:
            continue
        n = next(iter(systems.values())).n
        # the SAME resampled index sets for every system -> paired bootstrap
        boot_idx = rng.integers(0, n, size=(n_boot, n))
        boot = {}
        for m, s in systems.items():
            boot[m] = {k: np.array([s.score(k, ix) for ix in boot_idx]) for k in s.keys()}
            row = {"model": m, "direction": direction, "n": n}
            for k in s.keys():
                lo, hi = np.percentile(boot[m][k], [2.5, 97.5])
                row.update({k: round(s.score(k), 2), f"{k}_lo": round(lo, 2), f"{k}_hi": round(hi, 2)})
            rows.append(row)
            print(f"{m:16s} {direction:8s} " + "  ".join(f"{k} {row[k]:6.2f}" for k in s.keys()), flush=True)

        # paired bootstrap (Koehn 2004): fraction of resamples where the system does NOT beat baseline
        if "baseline" in systems:
            for m in systems:
                if m == "baseline":
                    continue
                for k in systems[m].keys():
                    if k not in boot["baseline"]:
                        continue
                    delta = boot[m][k] - boot["baseline"][k]
                    obs = systems[m].score(k) - systems["baseline"].score(k)
                    p = float(np.mean(delta <= 0)) if obs > 0 else float(np.mean(delta >= 0))
                    sig_rows.append({"model": m, "vs": "baseline", "direction": direction, "metric": k,
                                     "delta": round(obs, 2), "delta_lo": round(np.percentile(delta, 2.5), 2),
                                     "delta_hi": round(np.percentile(delta, 97.5), 2), "p_value": round(p, 4),
                                     "significant_0.05": p < 0.05})

        # sentence level chrF++ (length analysis, error analysis)
        srcs = read_lines(PROCESSED / pair / f"test.{src_l}")
        refs = read_lines(PROCESSED / pair / f"test.{tgt_l}")
        for m, s in systems.items():
            for i, c in enumerate(s.sentence_chrf()):
                sent_rows.append({"model": m, "direction": direction, "idx": i,
                                  "src_words": len(srcs[i].split()), "ref_words": len(refs[i].split()),
                                  "chrf++": round(c, 2),
                                  "comet": "" if s.comet is None else round(float(s.comet[i]), 4)})

        # learning curve (only directions that have the fractional runs)
        for m, frac in list(FRAC_MODELS.items()) + [("baseline", 1.0)]:
            if any(f.startswith("baseline_frac") and (OUTPUTS / f / direction / "test.hyp").exists()
                   for f in FRAC_MODELS):
                s = systems.get(m) or load_system(m, direction)
                if s is None:
                    continue
                cfg = json.loads((OUTPUTS / m / direction / "run_config.json").read_text(encoding="utf-8"))
                ix = rng.integers(0, s.n, size=(n_boot, s.n))
                b = np.array([s.score("chrf++", i) for i in ix])
                lc_rows.append({"direction": direction, "fraction": frac,
                                "n_train": cfg["config"].get("n_train_real"),
                                "chrf++": round(s.score("chrf++"), 2), "chrf++_lo": round(np.percentile(b, 2.5), 2),
                                "chrf++_hi": round(np.percentile(b, 97.5), 2), "bleu": round(s.score("bleu"), 2)})

    write_csv(RESULTS / "scores.csv", rows)
    write_csv(RESULTS / "significance.csv", sig_rows)
    write_csv(RESULTS / "sentence_scores.csv", sent_rows)
    write_csv(RESULTS / "learning_curve.csv", lc_rows)
    (RESULTS / "signatures.json").write_text(json.dumps(
        {**signatures, "comet": "Unbabel/wmt22-comet-da (x100)", "bootstrap": f"n={n_boot}, seed={seed}, 95% percentile CI"},
        indent=2), encoding="utf-8")
    write_markdown(rows, sig_rows, lc_rows, signatures, n_boot)
    print("wrote", RESULTS)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def zeroshot_rows():
    out = []
    for d in directions():
        f = OUTPUTS / "nllb_finetune" / d / "run_config.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("zeroshot_dev_chrf") is not None:
                out.append((d, r["zeroshot_dev_chrf"], r.get("dev_chrf_beam")))
    return out


def write_markdown(rows, sig_rows, lc_rows, signatures, n_boot):
    from src.viz.style import MODEL_LABELS
    has_comet = any("comet" in r for r in rows)
    keys = ["chrf++", "bleu"] + (["comet"] if has_comet else [])
    by = {(r["model"], r["direction"]): r for r in rows}
    dirs = [d for d in directions() if any((m, d) in by for m in MAIN_MODELS)]
    sig = {(s["model"], s["direction"], s["metric"]): s for s in sig_rows}
    L = ["# Test-set results", "",
         f"Held-out test split (see `docs/task_notes.md`). 95% bootstrap CIs over {n_boot} resamples; "
         "`*` = significantly better than the baseline, `†` = significantly worse "
         "(paired bootstrap, one-sided p < 0.05).", ""]
    for k in keys:
        L += [f"## {k.upper() if k != 'chrf++' else 'chrF++'} (test)", "",
              "| Direction | " + " | ".join(MODEL_LABELS.get(m, m) for m in MAIN_MODELS) + " |",
              "|---|" + "---:|" * len(MAIN_MODELS)]
        means = {m: [] for m in MAIN_MODELS}
        for d in dirs:
            cells = []
            vals = {m: by[(m, d)].get(k) for m in MAIN_MODELS if (m, d) in by}
            best = max((v for v in vals.values() if v is not None), default=None)
            for m in MAIN_MODELS:
                r = by.get((m, d))
                if r is None or r.get(k) is None:
                    cells.append("n/a")
                    continue
                means[m].append(r[k])
                mark = ""
                s = sig.get((m, d, k))
                if s and s["significant_0.05"]:
                    mark = "*" if s["delta"] > 0 else "†"
                txt = f"{r[k]:.2f}{mark} <sub>[{r[k + '_lo']:.1f}, {r[k + '_hi']:.1f}]</sub>"
                cells.append(f"**{txt}**" if r[k] == best else txt)
            L.append(f"| {d} | " + " | ".join(cells) + " |")
        L.append("| **Average** | " + " | ".join(
            f"{np.mean(v):.2f}" + (f" ({len(v)}/{len(dirs)})" if len(v) < len(dirs) else "") if v else "n/a"
            for v in means.values()) + " |")
        L.append("")
    zs = zeroshot_rows()
    if zs:
        L += ["## NLLB-200 zero-shot vs fine-tuned (dev chrF++, greedy subset / beam full)", "",
              "| Direction | Zero-shot (dev subset, greedy) | Fine-tuned (dev, beam) |", "|---|---:|---:|"]
        L += [f"| {d} | {a:.2f} | {b:.2f} |" for d, a, b in zs]
        L.append("")
    if lc_rows:
        L += ["## Learning curve (baseline, test chrF++)", "", "| Direction | Fraction | Train pairs | chrF++ | 95% CI |",
              "|---|---:|---:|---:|---|"]
        L += [f"| {r['direction']} | {r['fraction']:.0%} | {r['n_train']} | {r['chrf++']:.2f} | "
              f"[{r['chrf++_lo']:.1f}, {r['chrf++_hi']:.1f}] |" for r in lc_rows]
        L.append("")
    L += ["## Metric signatures", ""] + [f"- **{k}**: `{v}`" for k, v in signatures.items()]
    L += ["- **COMET**: `Unbabel/wmt22-comet-da`, system score = mean segment score x100" if has_comet
          else "- **COMET**: not available for these runs (see `docs/task_notes.md`)"]
    (RESULTS / "results_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(a.n_boot, a.seed)


if __name__ == "__main__":
    main()

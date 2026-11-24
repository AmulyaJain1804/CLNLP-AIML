"""Collect results into presentation/data.js and copy the figures the slides use.

Rerun after `bash run.sh eval && bash run.sh figures` (e.g. when the NLLB en-mni run arrives):
    python scripts/build_presentation.py
The site is static: open presentation/index.html in a browser (no server needed).
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import OUTPUTS, PROCESSED, REPORTS, directions, read_lines, selected_pairs, split_direction  # noqa: E402

SITE = ROOT / "presentation"
RES = REPORTS / "results"
MODELS = ["baseline", "backtranslation", "nllb_finetune"]
FIGS = ["eda/split_sizes", "eda/length_ratio", "eda/zipf", "results/learning_curve", "results/nllb_zeroshot_gain",
        "results/length_vs_score", "results/attention_lus-en", "results/dev_chrf_curves", "results/loss_curves_en-as"]


def rows(p):
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return None


def main():
    data = {"directions": directions(), "models": MODELS}

    data["scores"] = [{"model": r["model"], "direction": r["direction"],
                       **{k: num(r.get(k)) for k in ("chrf++", "chrf++_lo", "chrf++_hi", "bleu", "bleu_lo", "bleu_hi",
                                                     "comet", "comet_lo", "comet_hi") if r.get(k)}}
                      for r in rows(RES / "scores.csv")]
    data["significance"] = [{"model": r["model"], "direction": r["direction"], "metric": r["metric"],
                             "delta": num(r["delta"]), "p": num(r["p_value"]),
                             "sig": r["significant_0.05"] == "True"} for r in rows(RES / "significance.csv")]
    data["learning_curve"] = [{"direction": r["direction"], "fraction": float(r["fraction"]), "n": int(r["n_train"]),
                               "chrf": num(r["chrf++"]), "lo": num(r["chrf++_lo"]), "hi": num(r["chrf++_hi"])}
                              for r in rows(RES / "learning_curve.csv")]

    pairs = []
    for p in selected_pairs():
        st = json.loads((PROCESSED / p / "stats.json").read_text(encoding="utf-8"))
        q = OUTPUTS / "synthetic" / p / "quality.json"
        q = json.loads(q.read_text(encoding="utf-8")) if q.exists() else {}
        pairs.append({"pair": p, "splits": st["splits"], "removed": st["filters"]["train_removed"],
                      "train_input": st["filters"]["train_input"], "spm_vocab": st["spm"]["vocab_size"],
                      "synthetic": q.get("n_synthetic"), "roundtrip_chrf": q.get("roundtrip_chrf++"),
                      "mono_pool": q.get("mono_pool_size")})
    data["pairs"] = pairs

    zs = []
    for d in directions():
        f = OUTPUTS / "nllb_finetune" / d / "run_config.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("finished"):
                zs.append({"direction": d, "zeroshot": num(r["zeroshot_dev_chrf"]), "finetuned": num(r["dev_chrf_beam"])})
    data["zeroshot"] = zs

    cats = Counter()
    per_model = {m: Counter() for m in MODELS}
    for r in rows(RES / "error_categories.csv"):
        for c in r["categories"].split(";"):
            per_model[r["model"]][c] += 1
            cats[c] += 1
    data["errors"] = {m: dict(c) for m, c in per_model.items()}

    hours = 0.0
    for f in OUTPUTS.glob("*/*/run_config.json"):
        hours += json.loads(f.read_text(encoding="utf-8")).get("wall_clock_s", 0) / 3600
    data["gpu_hours"] = round(hours, 1)

    # translation showcase: per direction, a 8-20 word sentence where all systems exist and NLLB is clearly better
    sent = {}
    for r in rows(RES / "sentence_scores.csv"):
        sent.setdefault((r["model"], r["direction"]), {})[int(r["idx"])] = float(r["chrf++"])
    examples = []
    for d in directions():
        if not all((m, d) in sent for m in MODELS):
            continue
        src_l, tgt_l, pair = split_direction(d)
        src = read_lines(PROCESSED / pair / f"test.{src_l}")
        ref = read_lines(PROCESSED / pair / f"test.{tgt_l}")
        hyp = {m: read_lines(OUTPUTS / m / d / "test.hyp") for m in MODELS}
        best = None
        for i in range(len(src)):
            n = len(src[i].split())
            if not 8 <= n <= 18 or not 0.6 <= len(ref[i].split()) / n <= 1.6:
                continue
            s = {m: sent[(m, d)][i] for m in MODELS}
            score = s["nllb_finetune"] - s["baseline"] if 25 <= s["baseline"] <= 55 and 55 <= s["nllb_finetune"] <= 82 else -1
            if score > 0 and (best is None or score > best[0]):
                best = (score, i, s)
        if best:
            _, i, s = best
            examples.append({"direction": d, "idx": i, "src": src[i], "ref": ref[i],
                             "hyp": {m: hyp[m][i] for m in MODELS}, "chrf": {m: round(v, 1) for m, v in s.items()}})
    data["examples"] = examples

    SITE.mkdir(exist_ok=True)
    (SITE / "data.js").write_text("// generated by scripts/build_presentation.py - do not edit\nwindow.NMT_DATA = "
                                  + json.dumps(data, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
    (SITE / "figures").mkdir(exist_ok=True)
    for f in FIGS:
        src = REPORTS / "figures" / f"{f}.png"
        if src.exists():
            shutil.copy(src, SITE / "figures" / f"{Path(f).name}.png")
    print("wrote", SITE / "data.js", f"({len(data['scores'])} scores, {len(examples)} examples)")


if __name__ == "__main__":
    main()

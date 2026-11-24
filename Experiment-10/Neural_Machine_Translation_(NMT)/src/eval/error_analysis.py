"""M5 error analysis: 64 low-scoring test sentences per model (8 per direction), auto-categorised.

Categories (a sentence can get several; the first matching one is its *primary* category):
  reference_noise  the test pair itself looks misaligned (reference <0.4x or >2.5x the source length)
  script_error     <80% of the hypothesis letters are in the target script (wrong-script output)
  hallucination    hypothesis >1.6x the reference length, or a repeated 3-gram loop
  omission         hypothesis <0.6x the reference length
  untranslated     source words copied verbatim into a hypothesis in another language
  named_entity     names/numbers of the source missing from the hypothesis
  mistranslation   none of the above: fluent-length output with wrong lexical choice / meaning
Labels are heuristic (rule-based, reproducible); the examples are printed so a reader can verify them.

Reads reports/results/sentence_scores.csv (run src.eval.evaluate first).
Writes reports/error_analysis.md and reports/results/error_categories.csv.
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict

from src.eval.evaluate import MAIN_MODELS, RESULTS
from src.preprocess.clean import script_fraction
from src.utils import OUTPUTS, PROCESSED, REPORTS, directions, load_yaml, read_lines, split_direction

PER_DIRECTION = 8
CATS = ["reference_noise", "script_error", "hallucination", "omission", "untranslated", "named_entity", "mistranslation"]
CAT_NAMES = {"reference_noise": "Reference noise (misaligned pair)", "script_error": "Script error", "hallucination": "Hallucination / repetition",
             "omission": "Omission (too short)", "untranslated": "Untranslated words",
             "named_entity": "Named entity / number", "mistranslation": "Mistranslation (lexical / meaning)"}
NUM = re.compile(r"\d+")


def repeated_ngram(words, n=3, k=3):
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return any(c >= k for c in grams.values())


def entities(src, src_lang):
    """Numbers, plus capitalised non-initial words when the source is in Latin script."""
    ents = set(NUM.findall(src))
    if src_lang in ("en", "lus"):
        toks = re.findall(r"[A-Za-z][\w'-]*", src)
        ents |= {t for i, t in enumerate(toks) if i > 0 and t[0].isupper() and len(t) > 2}
    return ents


def categorise(src, ref, hyp, src_lang, tgt_lang, scripts):
    cats = []
    hw, rw, sw = hyp.split(), ref.split(), src.split()
    if not 0.4 <= len(rw) / max(1, len(sw)) <= 2.5:
        cats.append("reference_noise")
    tgt_script = scripts.get(tgt_lang, "LATIN")
    if hyp.strip() and script_fraction(hyp, tgt_script) < 0.8:
        cats.append("script_error")
    ratio = len(hw) / max(1, len(rw))
    if ratio > 1.6 or repeated_ngram(hw):
        cats.append("hallucination")
    if ratio < 0.6:
        cats.append("omission")
    # untranslated: source words (len>3) copied into the output although they are not in the reference
    src_words = {w.lower().strip(".,;:!?\"'()") for w in src.split() if len(w) > 3}
    ref_words = {w.lower().strip(".,;:!?\"'()") for w in rw}
    copied = [w for w in hw if w.lower().strip(".,;:!?\"'()") in src_words - ref_words]
    if src_lang != tgt_lang and len(copied) >= 2:
        cats.append("untranslated")
    ents = entities(src, src_lang)
    if ents:
        missing = [e for e in ents if e not in hyp and (e in ref or NUM.fullmatch(e))]
        if missing:
            cats.append("named_entity")
    return cats or ["mistranslation"]


def short(s, n=160):
    s = s.replace("|", "\\|")
    return s if len(s) <= n else s[:n] + "…"


def main():
    scripts = load_yaml("configs/preprocess.yaml")["expected_script"]
    scores = defaultdict(list)
    with open(RESULTS / "sentence_scores.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            scores[(r["model"], r["direction"])].append((float(r["chrf++"]), int(r["idx"])))

    picked = defaultdict(list)  # model -> rows
    for model in MAIN_MODELS:
        for d in directions():
            if (model, d) not in scores:
                continue
            src_l, tgt_l, pair = split_direction(d)
            src = read_lines(PROCESSED / pair / f"test.{src_l}")
            ref = read_lines(PROCESSED / pair / f"test.{tgt_l}")
            hyp = read_lines(OUTPUTS / model / d / "test.hyp")
            # lowest chrF++ among sentences with >=5 reference words (very short refs are noise)
            cand = sorted((s, i) for s, i in scores[(model, d)] if len(ref[i].split()) >= 5)
            for s, i in cand[:PER_DIRECTION]:
                cats = categorise(src[i], ref[i], hyp[i], src_l, tgt_l, scripts)
                picked[model].append({"model": model, "direction": d, "idx": i, "chrf++": s,
                                      "primary": cats[0], "categories": ";".join(cats),
                                      "src": src[i], "ref": ref[i], "hyp": hyp[i]})

    rows = [r for m in picked for r in picked[m]]
    with open(RESULTS / "error_categories.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    from src.viz.style import MODEL_LABELS
    L = ["# Error analysis", "",
         f"For every model we took the **{PER_DIRECTION} lowest-scoring test sentences per direction** "
         "(sentence chrF++, references with at least 5 words), "
         f"which gives up to {PER_DIRECTION * 8} errors per model. Each one is labelled by the reproducible rules in "
         "`src/eval/error_analysis.py`. A sentence can carry several labels; the table counts every label, "
         "and the *primary* label is the first one that matched, in the order below.", "",
         "| Category | Rule |", "|---|---|",
         "| Reference noise | the test pair itself is misaligned: reference < 0.4x or > 2.5x the source length "
         "(the model is not at fault) |",
         "| Script error | under 80% of output letters in the target script |",
         "| Hallucination / repetition | output > 1.6x reference length, or a 3-gram repeated 3+ times |",
         "| Omission | output < 0.6x reference length |",
         "| Untranslated words | 2+ source words copied into the output that are not in the reference |",
         "| Named entity / number | a source number or capitalised name missing from the output |",
         "| Mistranslation | none of the above (adequate length, wrong words or meaning) |", "",
         "## Counts (all labels)", "",
         "| Category | " + " | ".join(MODEL_LABELS.get(m, m) for m in picked) + " |",
         "|---|" + "---:|" * len(picked)]
    counts = {m: Counter(c for r in picked[m] for c in r["categories"].split(";")) for m in picked}
    for c in CATS:
        L.append(f"| {CAT_NAMES[c]} | " + " | ".join(str(counts[m][c]) for m in picked) + " |")
    L.append("| *Sentences analysed* | " + " | ".join(str(len(picked[m])) for m in picked) + " |")
    L += ["", "## Primary category by direction", ""]
    for m in picked:
        L += [f"### {MODEL_LABELS.get(m, m)}", "", "| Direction | " + " | ".join(CAT_NAMES[c].split(" (")[0] for c in CATS) + " |",
              "|---|" + "---:|" * len(CATS)]
        for d in directions():
            pc = Counter(r["primary"] for r in picked[m] if r["direction"] == d)
            if pc:
                L.append(f"| {d} | " + " | ".join(str(pc[c]) for c in CATS) + " |")
        L.append("")
    L += ["## Examples", "", "Two examples per category and model (src / ref / hyp, sentence chrF++).", ""]
    for m in picked:
        L += [f"### {MODEL_LABELS.get(m, m)}", ""]
        for c in CATS:
            ex = [r for r in picked[m] if r["primary"] == c][:2]
            if not ex:
                continue
            L.append(f"**{CAT_NAMES[c]}**")
            L.append("")
            for r in ex:
                L += [f"- `{r['direction']}` #{r['idx']} (chrF++ {r['chrf++']:.1f})",
                      f"  - SRC: {short(r['src'])}", f"  - REF: {short(r['ref'])}", f"  - HYP: {short(r['hyp'])}"]
            L.append("")
    notes = REPORTS / "error_observations.md"  # hand-written reading of the examples (kept across reruns)
    L += ["## Observations", "", notes.read_text(encoding="utf-8").strip() if notes.exists()
          else "<!-- write reports/error_observations.md and rerun -->", ""]
    (REPORTS / "error_analysis.md").write_text("\n".join(L), encoding="utf-8")
    for m in picked:
        print(m, dict(counts[m]))
    print("wrote", REPORTS / "error_analysis.md")


if __name__ == "__main__":
    main()

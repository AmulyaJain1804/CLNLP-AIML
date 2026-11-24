# Task notes: WMT26 Shared Task on Low-Resource Indic Language Translation

Source: https://www2.statmt.org/wmt26/indic-mt-task.html

## Language pairs
The 2026 release covers English ↔ 11 north-east Indian languages in two categories:

| Category | Pairs (ISO code used here) |
|---|---|
| I (more data) | Assamese (`as`), Manipuri in Bengali script (`mni`), Manipuri in Meitei Mayek (`mni-Mtei`), Mizo (`lus`), Khasi (`kha`), Nyishi (`njz`) |
| II (very low resource) | Bodo (`bodo`), Karbi (`mjw`), Kokborok (`trp`), Nagamese (`nag`), Tagin (`tgj`) |

All 11 pairs are downloaded, cleaned, tokenised and covered by the EDA. **We model 4 pairs in both
directions (8 directions): en↔as, en↔mni, en↔lus, en↔bodo**
(`configs/pairs.yaml`, `selected: true`).

Why these four:
- Three Category I pairs with three different scripts (Bengali/Assamese, Bengali/Manipuri, Latin) and one
  Category II pair (Bodo, Devanagari), so both resource levels and three scripts are covered.
- Three of them (as, mni, lus) are NLLB-200 languages, which makes the transfer-learning comparison meaningful.
  Bodo is not in NLLB, so it tests transfer through a proxy language tag.

Why the others were skipped:
- The compute budget was 3 models × 8 directions + 6 learning-curve runs on free Kaggle/Colab GPUs.
- Karbi, Kokborok and Nagamese have under 2.1k training pairs and only 50–113 test sentences, so their
  confidence intervals would be too wide to compare models.
- Nyishi and Tagin sentences average about 7 tokens and are dominated by duplicates (8.9k removed for Nyishi).
- Khasi and Meitei Mayek were dropped for time only. The pipeline supports them; set `selected: true` and rerun.

## Data release used
The official **2026 training data** (Google Drive folder linked from the task page), downloaded on
2026-11-15 with `python -m src.data.download`. SHA-256 checksums are in `data/raw/_original/checksums.json`;
provenance and counts are in `docs/SOURCES.md`.

## Official metrics
The task page says systems are evaluated with automatic metrics "using BLEU, TER, COMET etc." and
by human evaluation from native speakers. It names no primary metric and no COMET variant.
**Assumption:** following the project brief and common practice for morphologically rich Indic
languages, **chrF++ is our primary metric**, with BLEU secondary. We report chrF++ (sacreBLEU,
`word_order=2`), BLEU (sacreBLEU, 13a), and COMET (`Unbabel/wmt22-comet-da`) when it could be
computed, each with its signature (`reports/results/signatures.json`). TER is not reported.

## Shared-task rules relevant to us
- Test data was released on 12 June 2026, and submissions closed in June 2026. Results have been declared.
- A *primary* system must be constrained (official data only) or use only public monolingual data. A
  *contrastive* system may use external data.
- Under these rules, our baseline and back-translation systems would count as constrained/primary (the BT
  monolingual text is English from the official release). The NLLB system would be contrastive, because it
  uses a pretrained external model.

## Test-set protocol and our assumption (important)
- The official 2026 test set and the 2025 gold test set are distributed as **password-protected zips**,
  and the password is only given to registered teams. Our team is not registered, so **we cannot use the
  official test split**. This conflicts with the brief ("use the official splits"), and this is how we
  resolved it.
- **Assumption.** We hold out our own dev and test sets from the official training release. The split
  is random with seed 42: dev = min(5%, 1000) pairs, test = min(5%, 2000) pairs
  (`src/data/prepare_raw.py`). The split is done once on the raw data, before any cleaning, and saved
  to `data/raw/{pair}/{dev,test}.*`.
- **No test information is used for training or design.**
  - Train pairs whose source or target also occurs in dev or test are removed. The counts are in
    `reports/eda_summary.md`.
  - SentencePiece models are trained on the train split only.
  - Model selection and early stopping use dev only.
  - The test set is decoded once per final model.
- **Consequences.** Our numbers are not comparable to the official leaderboard. The held-out data is
  in-domain with the training data, which is mostly Bible and JW-style religious text, government news
  and conversational text. About 1/3 of the worst-scoring test sentences are misaligned pairs inherited
  from the release; see `reports/error_analysis.md`.

## License and usage restrictions
- **Data.** The task page gives no explicit licence. We treat the data as research-only, for the
  shared task, and **do not redistribute it**: `data/` is excluded from the repository and is
  regenerated with `bash run.sh data`. The organisers credit the corpus creators in their
  "Language Resource Contributors" sheet.
- **NLLB-200-distilled-600M**: CC-BY-NC-4.0 (non-commercial; research use permitted).
- **COMET `wmt22-comet-da`**: Apache-2.0.
- **Tools.** sacreBLEU (Apache-2.0), SentencePiece (Apache-2.0), PyTorch (BSD), 🤗 Transformers (Apache-2.0).

## Other assumptions and deviations
| Topic | Decision |
|---|---|
| Bodo in NLLB | Bodo is not one of the 200 NLLB languages. We use the **`hin_Deva` tag as a proxy**, since Bodo is written in Devanagari. Zero-shot quality is therefore very low (dev chrF++ 12.3 en→bodo); fine-tuning teaches the model Bodo under the Hindi tag. |
| Back-translation monolingual data | No in-domain monolingual Assamese/Mizo/… text is released. The English monolingual pool is therefore the **English side of the *other* 10 training pairs** (deduplicated, minus anything in our dev and test sets). X→en uses classic back-translation; en→X uses forward translation (self-training), both with a `<BT>` source tag at a 1:1 ratio with the real data. |
| Compute reductions | Baseline: d_model 256, 4+4 layers, at most 20k steps. NLLB: at most 3 epochs / 4.5k optimizer steps, with dev chrF++ computed on a fixed 300-sentence dev subset during training. Both are documented in `configs/`. |
| NLLB checkpoints | Not saved (1.2 GB each, over the Kaggle disk quota). The `test.hyp` files plus the configs reproduce the results. |
| NLLB en→mni | The first run (Kaggle T4) ran out of GPU memory at step ~330. It was **re-run separately** on a Colab L4 (24 GB) with the default settings (`notebooks/03_finish_mni_and_comet.ipynb`); all 8 directions are complete. |
| COMET | Computed on GPU for all 24 system outputs (`notebooks/03_finish_mni_and_comet.ipynb`, `outputs/*/*/comet.json`), after fixing a protobuf/TensorFlow conflict on Colab. COMET's XLM-R encoder has not seen Manipuri, Mizo or Bodo, so it is a weak signal for those targets. **chrF++ is our primary metric.** |
| Git hash | Training ran from a zipped bundle on Kaggle/Colab without `.git`, so `run_config.json` records `git_commit: "unavailable"`. The exact code is the `src/` in this repository at the tag of the submission. |
| Hardware | Baseline, learning-curve and BT runs: 1× A100-40GB (Colab). NLLB: 2× T4 (Kaggle), en→mni and COMET: 1× L4 (Colab). See each `run_config.json`. |

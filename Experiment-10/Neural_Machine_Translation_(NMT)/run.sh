#!/usr/bin/env bash
# One command per stage.  Usage:  bash run.sh <stage> [args]
#   setup      create .venv with uv and install pinned requirements
#   data       M1: download the 2026 release + build data/raw/{pair}/{split}.{src,tgt} + docs/SOURCES.md
#   preprocess M2: clean, dedup, filter, train SentencePiece -> data/processed/
#   eda        M2: EDA figures + reports/eda_summary.md
#   bundle     zip code + processed data for the GPU notebooks (dist/nmt_bundle.zip)
#   notebooks  regenerate notebooks/*.ipynb
#   baseline   M3: train baseline for one direction        e.g. bash run.sh baseline en-as
#   bt         M4: synthetic data + BT model for one direction   e.g. bash run.sh bt as-en
#   nllb       M4: fine-tune NLLB-200 for one direction    e.g. bash run.sh nllb en-as
#   comet      M5: COMET for all test.hyp (GPU recommended)
#   eval       M5: BLEU/chrF++ + bootstrap CIs + significance + error analysis
#   figures    M5: all result figures
#   presentation  refresh presentation/data.js + figures (open presentation/index.html)
#   all-local  data -> preprocess -> eda -> eval -> figures   (needs outputs/ from the GPU runs)
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe     # Windows venv
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python; fi
export PYTHONIOENCODING=utf-8

stage="${1:-help}"; shift || true
case "$stage" in
  setup)      uv venv --python 3.11 .venv && uv pip install --python .venv -r requirements.txt ;;
  data)       $PY -m src.data.download && $PY -m src.data.prepare_raw ;;
  preprocess) $PY -m src.preprocess.run ;;
  eda)        $PY -m src.eda.eda ;;
  bundle)     $PY scripts/make_bundle.py "$@" ;;
  notebooks)  $PY scripts/build_notebooks.py ;;
  baseline)   $PY -m src.models.baseline.train --direction "$1" "${@:2}" ;;
  bt)         d="$1"; s="${d%%-*}"; t="${d#*-}"; x=$([ "$s" = en ] && echo "$t" || echo "$s")
              [ -f "outputs/synthetic/en-$x/quality.json" ] || $PY -m src.models.advanced.backtranslation generate --pair "en-$x"
              $PY -m src.models.advanced.backtranslation train --direction "$d" ;;
  nllb)       $PY -m src.models.advanced.nllb_finetune --direction "$1" "${@:2}" ;;
  comet)      $PY -m src.eval.comet_score "$@" ;;
  eval)       $PY -m src.eval.evaluate "$@" && $PY -m src.eval.error_analysis ;;
  figures)    $PY -m src.viz.plots "$@" ;;
  presentation) $PY scripts/build_presentation.py ;;
  all-local)  bash run.sh data && bash run.sh preprocess && bash run.sh eda && bash run.sh eval && bash run.sh figures && bash run.sh presentation ;;
  *)          sed -n '2,17p' "$0" ;;
esac

"""COMET (Unbabel/wmt22-comet-da) for every outputs/{model}/{direction}/test.hyp. Run on GPU.

Writes outputs/{model}/{direction}/comet.json  {model, version, system_score, segment_scores}.
Caveat: COMET's encoder (XLM-R) has seen Assamese but not Manipuri, Mizo or Bodo, so COMET
is least reliable for those targets; chrF++ is our primary metric (see docs/task_notes.md).

Usage:  python -m src.eval.comet_score [--models baseline nllb_finetune] [--batch-size 64]
"""
from __future__ import annotations

import argparse
import json
import os

# Colab ships TensorFlow; unbabel-comet downgrades protobuf, and transformers would then import a
# broken TF. COMET is PyTorch-only, so keep TF (and Flax) out of the import chain.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

from src.utils import OUTPUTS, PROCESSED, read_lines, split_direction

COMET_MODEL = "Unbabel/wmt22-comet-da"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import comet
    import torch
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(COMET_MODEL))
    gpus = 1 if torch.cuda.is_available() else 0
    for mdir in sorted(OUTPUTS.iterdir()):
        if not mdir.is_dir() or mdir.name in ("logs", "synthetic") or (args.models and mdir.name not in args.models):
            continue
        for ddir in sorted(mdir.iterdir()):
            hyp_f = ddir / "test.hyp"
            if not hyp_f.exists() or ((ddir / "comet.json").exists() and not args.overwrite):
                continue
            src_l, tgt_l, pair = split_direction(ddir.name)
            src = read_lines(PROCESSED / pair / f"test.{src_l}")
            ref = read_lines(PROCESSED / pair / f"test.{tgt_l}")
            hyp = read_lines(hyp_f)
            data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyp, ref)]
            out = model.predict(data, batch_size=args.batch_size, gpus=gpus, progress_bar=False)
            res = {"model": COMET_MODEL, "comet_version": getattr(comet, "__version__", "unknown"),
                   "system_score": float(out.system_score), "segment_scores": [float(x) for x in out.scores]}
            (ddir / "comet.json").write_text(json.dumps(res), encoding="utf-8")
            print(f"COMET {mdir.name}/{ddir.name}: {res['system_score']:.4f}", flush=True)


if __name__ == "__main__":
    main()

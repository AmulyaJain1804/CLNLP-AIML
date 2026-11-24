"""Zip everything the GPU notebooks need into dist/nmt_bundle.zip.

Upload that zip as a Kaggle Dataset (or to Colab / Google Drive). Contents: src/, configs/,
data/processed/ (cleaned data + SentencePiece models), requirements-gpu.txt, and
outputs/ (so a later notebook can reuse e.g. baseline checkpoints if you re-bundle).

Usage:  python scripts/make_bundle.py [--with-outputs]
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-outputs", action="store_true")
    args = ap.parse_args()
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / "nmt_bundle.zip"
    roots = ["src", "configs", "data/processed", "requirements-gpu.txt"]
    if args.with_outputs:
        roots.append("outputs")
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for r in roots:
            p = ROOT / r
            files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
            for f in files:
                if "__pycache__" in f.parts or f.suffix in (".pyc", ".pt", ".zip") or "checkpoint" in f.name:
                    continue
                z.write(f, f.relative_to(ROOT).as_posix())
                n += 1
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB, {n} files)")


if __name__ == "__main__":
    main()

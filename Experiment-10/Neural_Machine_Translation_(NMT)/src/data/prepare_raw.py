"""Stage 1b (M1): convert the original spreadsheets into aligned line files and split.

Output (handoff to M2):  data/raw/{pair}/{train,dev,test}.{src,tgt}   (src = English, tgt = X)
                         data/raw/{pair}/stats.json
                         docs/SOURCES.md

Only *format* operations happen here (no cleaning):
  * take the English / X columns (per configs/pairs.yaml), skip header row if present
  * drop rows where BOTH sides are empty (spreadsheet padding, not sentence pairs)
  * replace in-cell line breaks with a space (needed for one-sentence-per-line files)
  * deterministic shuffle (seed) and split into train / dev / test

Because the official 2026 test set is password protected (registered teams only), dev
and test are held out from the official training release. See docs/task_notes.md.

Usage:  python -m src.data.prepare_raw
"""
from __future__ import annotations

import json
import random

import pandas as pd

from src.utils import RAW, ROOT, pairs_config, write_lines

ORIG = RAW / "_original"
TASK_URL = "https://www2.statmt.org/wmt26/indic-mt-task.html"
DRIVE_URL = "https://drive.google.com/drive/folders/1TsaINvXwb5GbiEbEkj2OgcK-1wfrWESG"


def cell(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return " ".join(str(v).replace("\r", " ").replace("\n", " ").split(" ")).strip()


def load_pair(spec: dict) -> tuple[list[str], list[str], int]:
    path = ORIG / "training_2026" / spec["file"]
    if path.suffix == ".csv":
        df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(path, header=None, dtype=str)
    if spec["header"]:
        df = df.iloc[1:]
    en = [cell(v) for v in df.iloc[:, spec["en_col"]]]
    xx = [cell(v) for v in df.iloc[:, spec["x_col"]]]
    rows = len(en)
    kept = [(a, b) for a, b in zip(en, xx) if a or b]
    return [a for a, _ in kept], [b for _, b in kept], rows


def main() -> None:
    cfg = pairs_config()
    sp = cfg["split"]
    checks = json.loads((ORIG / "checksums.json").read_text(encoding="utf-8")) \
        if (ORIG / "checksums.json").exists() else {}
    all_stats = {}
    for pair, spec in cfg["pairs"].items():
        en, xx, rows = load_pair(spec)
        n = len(en)
        idx = list(range(n))
        random.Random(cfg["seed"]).shuffle(idx)
        n_test = min(int(n * sp["test_frac"]), sp["test_max"])
        n_dev = min(int(n * sp["dev_frac"]), sp["dev_max"])
        splits = {"test": idx[:n_test], "dev": idx[n_test:n_test + n_dev],
                  "train": sorted(idx[n_test + n_dev:])}
        out = RAW / pair
        stats = {"spreadsheet_rows": rows, "pairs_after_padding_removal": n}
        for name, ids in splits.items():
            write_lines(out / f"{name}.src", (en[i] for i in ids))
            write_lines(out / f"{name}.tgt", (xx[i] for i in ids))
            # alignment check: both sides must have identical line counts
            n_src = sum(1 for _ in open(out / f"{name}.src", encoding="utf-8"))
            n_tgt = sum(1 for _ in open(out / f"{name}.tgt", encoding="utf-8"))
            assert n_src == n_tgt == len(ids), f"misaligned {pair}/{name}"
            stats[name] = len(ids)
        (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        all_stats[pair] = stats
        print(f"{pair:12s} rows={rows:6d} kept={n:6d} train={stats['train']:6d} "
              f"dev={stats['dev']:5d} test={stats['test']:5d}")
    write_sources(cfg, all_stats, checks)


def write_sources(cfg: dict, stats: dict, checks: dict) -> None:
    lines = [
        "# SOURCES — data provenance and license log (owner: M1)",
        "",
        f"* **Task:** WMT26 Shared Task on Low-Resource Indic Language Translation — {TASK_URL}",
        f"* **Release:** official 2026 training data, Google Drive folder {DRIVE_URL}",
        "  (downloaded with `python -m src.data.download`; SHA-256 checksums in",
        "  `data/raw/_original/checksums.json`).",
        "* **Official 2026 test set / 2025 gold test set:** downloaded (`test2026.zip`, `test2025.zip`)",
        "  but **password protected** (released only to registered participants) — not used.",
        "* **License / usage:** the task page states no explicit license. We treat the data as",
        "  *research use only, for the purpose of the shared task*, do not redistribute it, and",
        "  cite the task. Original corpus creators are listed by the organisers in the",
        "  'Language Resource Contributors' sheet linked from the task page.",
        "* **Our splits:** dev/test are held out from the training release with a fixed seed",
        f"  ({cfg['seed']}); dev = min({cfg['split']['dev_frac']:.0%}, {cfg['split']['dev_max']}),"
        f" test = min({cfg['split']['test_frac']:.0%}, {cfg['split']['test_max']}).",
        "",
        "| Pair | Language | Cat. | Original file | Rows | Pairs kept | Train | Dev | Test | SHA-256 (first 12) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for pair, s in stats.items():
        spec = cfg["pairs"][pair]
        sha = checks.get(f"training_2026/{spec['file']}", "")[:12]
        lines.append(f"| {pair} | {spec['name']} | {spec['category']} | `{spec['file']}` | "
                     f"{s['spreadsheet_rows']} | {s['pairs_after_padding_removal']} | "
                     f"{s['train']} | {s['dev']} | {s['test']} | `{sha}` |")
    lines += ["", "Rows = data rows in the spreadsheet (header excluded); 'pairs kept' drops rows",
              "where both cells are empty (spreadsheet padding — notably ~5.2k rows in Nagamese)."]
    (ROOT / "docs" / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

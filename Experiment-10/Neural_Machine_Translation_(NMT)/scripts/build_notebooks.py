"""Generate the GPU notebooks in notebooks/ (keeps their shared setup cells identical).

Usage:  python scripts/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


SETUP = r'''
# ---- 1. Locate the project bundle and unpack it (works on Kaggle and Colab) ----
import os, sys, glob, zipfile, shutil, subprocess
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
# NB: recent Colab images also contain a /kaggle folder, so check Colab first
IN_KAGGLE = (not IN_COLAB) and ("KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/input"))
WORK = "/kaggle/working/project" if IN_KAGGLE else "/content/project"

def find_bundle():
    pats = ["/kaggle/input/**/nmt_bundle.zip", "/content/nmt_bundle.zip",
            "/content/drive/MyDrive/nmt_bundle.zip", "/content/drive/MyDrive/**/nmt_bundle.zip"]
    for p in pats:
        hits = glob.glob(p, recursive=True)
        if hits:
            return ("zip", hits[0])
    # Kaggle auto-extracts uploaded zips -> look for an extracted project folder
    for d in glob.glob("/kaggle/input/**/src/models", recursive=True):
        return ("dir", os.path.dirname(os.path.dirname(d)))
    return (None, None)

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

kind, path = find_bundle()
if kind is None and IN_COLAB:
    from google.colab import files
    print("Upload dist/nmt_bundle.zip"); up = files.upload(); kind, path = "zip", "/content/" + list(up)[0]
assert kind, "nmt_bundle.zip not found - attach it as a Kaggle dataset / put it in MyDrive"
if not os.path.exists(WORK + "/src"):
    os.makedirs(WORK, exist_ok=True)
    if kind == "zip":
        zipfile.ZipFile(path).extractall(WORK)
    else:
        shutil.copytree(path, WORK, dirs_exist_ok=True)
os.chdir(WORK); sys.path.insert(0, WORK)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # less fragmentation (inherited by jobs)

# On Colab keep outputs on Google Drive so a disconnect never loses finished runs
if IN_COLAB:
    drive_out = "/content/drive/MyDrive/nmt_outputs"
    os.makedirs(drive_out, exist_ok=True)
    if not os.path.islink("outputs"):
        if os.path.isdir("outputs"):
            shutil.copytree("outputs", drive_out, dirs_exist_ok=True); shutil.rmtree("outputs")
        os.symlink(drive_out, "outputs")
print("bundle:", path, "| workdir:", os.getcwd())
'''

GPU_INFO = r'''
# ---- 2. Dependencies + GPU check ----
!pip install -q {pkgs}
import torch
n = torch.cuda.device_count()
print("torch", torch.__version__, "| GPUs:", n, [torch.cuda.get_device_name(i) for i in range(n)])
print("bf16 supported:", torch.cuda.is_bf16_supported() if n else None)
assert n > 0, "No GPU! Kaggle: Settings -> Accelerator -> GPU T4 x2.  Colab: Runtime -> Change runtime type -> A100"
'''

HELPERS = r'''
# ---- helpers: skip finished runs (safe to re-run the notebook after a disconnect) ----
import json
from pathlib import Path
from src.run_queue import run_jobs

def finished(model, direction):
    f = Path("outputs") / model / direction / "run_config.json"
    return f.exists() and json.loads(f.read_text()).get("finished", False)

def todo(jobs):
    return [(n, a) for (n, a, m, d) in jobs if not finished(m, d)]

DIRECTIONS = ["en-as", "as-en", "en-lus", "lus-en", "en-mni", "mni-en", "en-bodo", "bodo-en"]  # big pairs first
'''

COMET = r'''
# ---- COMET (optional; installed last so it cannot break the training environment) ----
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unbabel-comet"], capture_output=True, text=True)
print("pip:", "OK" if r.returncode == 0 else r.stderr[-3000:])
env = dict(os.environ, USE_TF="0", TRANSFORMERS_NO_TF="1")
r = subprocess.run([sys.executable, "-m", "src.eval.comet_score", "--models"] + {models}, capture_output=True, text=True, env=env)
print(r.stdout[-3000:])
if r.returncode != 0:
    print("COMET failed (non-fatal; chrF++/BLEU are computed locally). Error tail:")
    print(r.stderr[-4000:])
'''

PACK = r'''
# ---- Package results: small results zip + separate checkpoints zip ----
import zipfile, os
from pathlib import Path
out_dir = "/kaggle/working" if IN_KAGGLE else "/content/drive/MyDrive"
res = f"{{out_dir}}/{name}_results.zip"; ck = f"{{out_dir}}/{name}_checkpoints.zip"
with zipfile.ZipFile(res, "w", zipfile.ZIP_DEFLATED) as zr, zipfile.ZipFile(ck, "w") as zc:
    for root in {roots}:
        for f in Path("outputs", root).rglob("*") if Path("outputs", root).exists() else []:
            if f.is_file():
                (zc if f.suffix == ".pt" or "checkpoint" in f.parts else zr).write(f, f.as_posix())
    for f in Path("outputs/logs").glob("*.log"):
        zr.write(f, f.as_posix())
for z in (res, ck):
    print(z, round(os.path.getsize(z) / 1e6, 1), "MB")
print("Download the zip(s) and extract them in the project root (they contain outputs/...).")
'''


def nb(cells):
    return {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


HOWTO = """
**How to run**

1. On your laptop: `python scripts/make_bundle.py` creates `dist/nmt_bundle.zip`.
2. **Kaggle:** New Notebook, then File, then Import Notebook (this file). Next, Add Input, then Upload Dataset (`nmt_bundle.zip`).
   Settings: **Accelerator = GPU T4 x2**, **Internet = On**. Click **Save Version, then Save & Run All (Commit)** so it
   keeps running after you close the tab. When it finishes, download `{name}_results.zip` (and `_checkpoints.zip`) from Output.
3. **Colab:** Upload this notebook, put `nmt_bundle.zip` in *My Drive* (root), Runtime, then **A100 GPU**, then Run all.
   Outputs are written straight to `MyDrive/nmt_outputs/`, so if Colab disconnects, just run all again and finished runs are skipped.
4. On your laptop, extract the results zip in the project root, then `bash run.sh eval`.
"""


def nb1():
    jobs = r'''
# ---- 3. Phase A: baseline Transformer (8 directions) + learning curve (en-as, as-en at 10/25/50%) ----
jobs = [(f"baseline_{d}", ["-m", "src.models.baseline.train", "--direction", d], "baseline", d) for d in DIRECTIONS]
for frac in (0.1, 0.25, 0.5):
    for d in ("en-as", "as-en"):
        m = f"baseline_frac{frac}"
        jobs.append((f"{m}_{d}", ["-m", "src.models.baseline.train", "--direction", d, "--train-frac", str(frac),
                                  "--model-name", m], m, d))
print(run_jobs(todo(jobs)))
'''
    gen = r'''
# ---- 4. Phase B: synthetic data with the baseline models (one corpus per pair) ----
gen_jobs = []
for pair in ("en-as", "en-lus", "en-mni", "en-bodo"):
    if not Path(f"outputs/synthetic/{pair}/quality.json").exists():
        gen_jobs.append((f"synthetic_{pair}", ["-m", "src.models.advanced.backtranslation", "generate", "--pair", pair]))
print(run_jobs(gen_jobs))
for pair in ("en-as", "en-lus", "en-mni", "en-bodo"):
    q = json.loads(Path(f"outputs/synthetic/{pair}/quality.json").read_text())
    print(pair, {k: q[k] for k in ("n_synthetic", "synthetic_ratio", "roundtrip_chrf++")})
'''
    bt = r'''
# ---- 5. Phase C: retrain with real + synthetic data (BT for X->en, forward-translation for en->X) ----
jobs = [(f"backtranslation_{d}", ["-m", "src.models.advanced.backtranslation", "train", "--direction", d],
         "backtranslation", d) for d in DIRECTIONS]
print(run_jobs(todo(jobs)))
'''
    summary = r'''
# ---- 6. Quick dev summary ----
for m in ["baseline", "backtranslation"]:
    for d in DIRECTIONS:
        f = Path("outputs") / m / d / "run_config.json"
        if f.exists():
            r = json.loads(f.read_text())
            print(f"{m:16s} {d:8s} dev chrF++ (beam) {r.get('dev_chrf_beam', float('nan')):6.2f}  "
                  f"steps {r.get('steps')}  {r['wall_clock_s']/60:.0f} min")
'''
    cells = [
        md("# Notebook 1: Baseline Transformer, learning curve and back-translation\n"
           "WMT26 Indic MT, low-resource NMT project. Runs on **Kaggle 2×T4** (two jobs in parallel, fp16) "
           "or **Colab A100** (sequential, bf16).\n\nEstimated time: about 5 h on 2×T4, about 2.5 h on A100.\n"
           + HOWTO.format(name="nb1")),
        code(SETUP), code(GPU_INFO.format(pkgs="sentencepiece==0.2.3 sacrebleu==2.6.0 pyyaml")), code(HELPERS),
        code(jobs), code(gen), code(bt), code(summary),
        code(COMET.format(models='["baseline", "backtranslation"]')),
        code(PACK.format(name="nb1", roots='["baseline", "baseline_frac0.1", "baseline_frac0.25", '
                                          '"baseline_frac0.5", "backtranslation", "synthetic"]')),
    ]
    return nb(cells)


def nb2():
    smoke = r'''
# ---- 3. 2-minute smoke test (catches library / memory problems before the long run) ----
r = subprocess.run([sys.executable, "-m", "src.models.advanced.nllb_finetune", "--direction", "en-bodo",
                    "--model-name", "_nllb_smoke", "--max-steps", "10", "--eval-every", "5", "--limit-train", "400"],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:])
assert r.returncode == 0, "smoke test failed - see the error above"
shutil.rmtree("outputs/_nllb_smoke", ignore_errors=True)
'''
    jobs = r'''
# ---- 4. Fine-tune NLLB-200-distilled-600M, 8 directions ----
jobs = [(f"nllb_{d}", ["-m", "src.models.advanced.nllb_finetune", "--direction", d], "nllb_finetune", d)
        for d in DIRECTIONS]
print(run_jobs(todo(jobs)))
for d in DIRECTIONS:
    f = Path("outputs/nllb_finetune") / d / "run_config.json"
    if f.exists():
        r = json.loads(f.read_text())
        print(f"{d:8s} zero-shot dev chrF++ {r.get('zeroshot_dev_chrf', float('nan')):6.2f} -> fine-tuned "
              f"{r.get('dev_chrf_beam', float('nan')):6.2f}   {r['wall_clock_s']/60:.0f} min")
'''
    cells = [
        md("# Notebook 2: Fine-tuning NLLB-200-distilled-600M (transfer learning)\n"
           "WMT26 Indic MT, low-resource NMT project. Runs on **Kaggle 2×T4** (two directions in parallel, "
           "fp16, about 12 GB/GPU) or **Colab A100** (bf16).\n\nEstimated time: about 4 h on 2×T4, about 1.5 h on A100. "
           "Independent of Notebook 1, so both can run at the same time on different accounts.\n"
           + HOWTO.format(name="nb2")),
        code(SETUP), code(GPU_INFO.format(pkgs="-r requirements-gpu.txt")), code(HELPERS),
        code(smoke), code(jobs),
        code(COMET.format(models='["nllb_finetune"]')),
        code(PACK.format(name="nb2", roots='["nllb_finetune"]')),
    ]
    return nb(cells)


def nb3():
    rerun = r"""
# ---- 3. NLLB en->mni (the run that ran out of memory on the T4). Skipped if already finished ----
d = "en-mni"
if not finished("nllb_finetune", d):
    extra = [] if torch.cuda.get_device_properties(0).total_memory > 30e9 else ["--batch-size", "4", "--grad-accum", "8"]
    print(run_jobs([(f"nllb_{d}", ["-m", "src.models.advanced.nllb_finetune", "--direction", d] + extra)]))
r = json.loads(Path(f"outputs/nllb_finetune/{d}/run_config.json").read_text())
print(f"{d}: zero-shot dev chrF++ {r['zeroshot_dev_chrf']:.2f} -> fine-tuned {r['dev_chrf_beam']:.2f}  ({r['wall_clock_s']/60:.0f} min)")
"""
    pack = r"""
# ---- 5. Package: en-mni NLLB outputs + every comet.json -> MyDrive/nb3_results.zip (or Kaggle Output) ----
import zipfile
out_dir = "/kaggle/working" if IN_KAGGLE else "/content/drive/MyDrive"
res = f"{out_dir}/nb3_results.zip"
with zipfile.ZipFile(res, "w", zipfile.ZIP_DEFLATED) as z:
    for f in list(Path("outputs/nllb_finetune/en-mni").glob("*")) + list(Path("outputs").glob("*/*/comet.json")) \
             + list(Path("outputs/logs").glob("nllb_en-mni.log")):
        if f.is_file():
            z.write(f, f.as_posix()); print("  +", f)
print(res, round(os.path.getsize(res) / 1e6, 2), "MB  -> extract in the project root")
"""
    cells = [
        md("# Notebook 3: finish the GPU work (NLLB en-mni + COMET for all systems)\n"
           "Needs the bundle built **with outputs**: `python scripts/make_bundle.py --with-outputs` "
           "(adds every `test.hyp`, no checkpoints).\n\n"
           "**Colab A100** (recommended): put `nmt_bundle.zip` in My Drive, Runtime, then A100, then Run all. About 25 min.\n"
           "**Kaggle T4:** works too; the en-mni run then uses batch 4 × accumulation 8 automatically. About 1.5 h.\n\n"
           "Result: `nb3_results.zip` (My Drive root / Kaggle Output). Extract it in the project root, then run\n"
           "`bash run.sh eval && bash run.sh figures && bash run.sh presentation`."),
        code(SETUP), code(GPU_INFO.format(pkgs="-r requirements-gpu.txt")), code(HELPERS),
        code(rerun),
        code(COMET.format(models='["baseline", "backtranslation", "nllb_finetune"]')),
        code(pack),
    ]
    return nb(cells)


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, fn in (("01_baseline_and_backtranslation.ipynb", nb1), ("02_nllb_finetune.ipynb", nb2),
                     ("03_finish_mni_and_comet.ipynb", nb3)):
        (out / name).write_text(json.dumps(fn(), indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote notebooks/" + name)


if __name__ == "__main__":
    main()

"""Shared helpers: paths, config loading, seeding, run metadata."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def load_yaml(path: str | Path) -> dict:
    with open(ROOT / path if not Path(path).is_absolute() else path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def pairs_config() -> dict:
    return load_yaml("configs/pairs.yaml")


def selected_pairs() -> list[str]:
    cfg = pairs_config()
    return [p for p, v in cfg["pairs"].items() if v.get("selected")]


def directions(pairs: list[str] | None = None) -> list[str]:
    """'en-as' pair -> directions ['en-as', 'as-en']."""
    out = []
    for p in pairs or selected_pairs():
        x = p.split("-", 1)[1]
        out += [f"en-{x}", f"{x}-en"]
    return out


def split_direction(direction: str) -> tuple[str, str, str]:
    """'as-en' -> (src='as', tgt='en', pair='en-as'); handles 'mni-Mtei-en'."""
    if direction.startswith("en-"):
        src, tgt = "en", direction[3:]
        return src, tgt, direction
    src, tgt = direction[:-3], "en"
    return src, tgt, f"en-{src}"


def read_lines(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: str | Path, lines) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT", "unavailable (no git repo at run time)")


def hardware_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            info["gpu"] = torch.cuda.get_device_name(idx)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / 1e9, 1)
        else:
            info["gpu"] = None
    except ImportError:
        pass
    return info


class RunRecorder:
    """Writes outputs/{model}/{direction}/run_config.json and train_log.csv."""

    def __init__(self, out_dir: Path, config: dict, seed: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.start = time.time()
        self.meta = {"config": config, "seed": seed, "hardware": hardware_info(),
                     "git_commit": git_commit(),
                     "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.log_path = self.out_dir / "train_log.csv"
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("step,epoch,train_loss,dev_loss,dev_chrf,dev_bleu,lr,elapsed_s\n")
        self.save()

    def log(self, step, epoch, train_loss, dev_loss="", dev_chrf="", dev_bleu="", lr=""):
        def fmt(v):
            return f"{v:.5g}" if isinstance(v, float) else str(v)
        row = [step, epoch, train_loss, dev_loss, dev_chrf, dev_bleu, lr,
               round(time.time() - self.start, 1)]
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(",".join(fmt(v) for v in row) + "\n")

    def save(self, **extra):
        self.meta.update(extra)
        self.meta["wall_clock_s"] = round(time.time() - self.start, 1)
        with open(self.out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

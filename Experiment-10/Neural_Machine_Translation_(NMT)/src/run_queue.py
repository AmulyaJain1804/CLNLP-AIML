"""Run a list of training jobs, one job per GPU at a time (Kaggle 2xT4 -> 2 jobs in parallel).

Each job is a python module + args; its stdout goes to outputs/logs/{name}.log.
Usage (from a notebook):
    from src.run_queue import run_jobs
    run_jobs([("baseline_en-as", ["-m", "src.models.baseline.train", "--direction", "en-as"]), ...])
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from src.utils import OUTPUTS, ROOT


def n_gpus() -> int:
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except ImportError:
        return 1


def run_jobs(jobs, gpus: int | None = None, poll: float = 15.0) -> dict:
    """jobs: list of (name, argv-after-python). Returns {name: returncode}."""
    gpus = gpus or n_gpus()
    logs = OUTPUTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    running: dict[int, tuple] = {}
    results = {}
    t0 = time.time()
    while pending or running:
        for g in range(gpus):
            if g not in running and pending:
                name, argv = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1",
                           PYTHONIOENCODING="utf-8",
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                fh = open(logs / f"{name}.log", "w", encoding="utf-8")
                p = subprocess.Popen([sys.executable] + argv, cwd=ROOT, env=env, stdout=fh,
                                     stderr=subprocess.STDOUT)
                running[g] = (name, p, fh, time.time())
                print(f"[{(time.time()-t0)/60:6.1f} min] GPU{g} START {name}", flush=True)
        time.sleep(poll)
        for g, (name, p, fh, st) in list(running.items()):
            if p.poll() is not None:
                fh.close()
                results[name] = p.returncode
                status = "OK" if p.returncode == 0 else f"FAILED (rc={p.returncode}) -> see outputs/logs/{name}.log"
                print(f"[{(time.time()-t0)/60:6.1f} min] GPU{g} END   {name} {status} "
                      f"({(time.time()-st)/60:.1f} min)", flush=True)
                if p.returncode != 0:
                    tail = (logs / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-2000:]
                    print(tail, flush=True)
                del running[g]
    return results

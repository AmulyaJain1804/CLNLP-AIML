"""M4 advanced model B: back-translation (X->en) and forward-translation / self-training (en->X).

One synthetic corpus per pair serves both directions:
    English monolingual sentence  --(baseline en->X)-->  synthetic X
    * X->en  : (synthetic X, real English)  = classic back-translation (Sennrich et al., 2016)
    * en->X  : (real English, synthetic X)  = forward-translation / self-training (Zhang & Zong, 2016)
Both are tagged with <BT> on the source side and mixed 1:1 with the real data; the same
small Transformer is then retrained from scratch (so the only change vs the baseline is the data).

Steps:
    python -m src.models.advanced.backtranslation generate --pair en-as     # needs outputs/baseline/en-as
    python -m src.models.advanced.backtranslation train --direction as-en   # BT
    python -m src.models.advanced.backtranslation train --direction en-as   # FT / self-training
Output: outputs/synthetic/{pair}/{mono.en, syn.X, quality.json}
"""
from __future__ import annotations

import argparse
import json
import random
import time

import sacrebleu

from src.models.baseline.data import SPTokenizer
from src.models.baseline.train import get_device, load_model, main as train_main, translate_lines
from src.utils import OUTPUTS, PROCESSED, load_yaml, pairs_config, read_lines, set_seed, split_direction, write_lines

CFG = "configs/backtranslation.yaml"


def mono_pool(pair: str, cfg: dict) -> list[str]:
    x = pair.split("-", 1)[1]
    exclude = set()
    for split in ("train", "dev", "test"):
        exclude.update(read_lines(PROCESSED / pair / f"{split}.en"))
    pool, seen = [], set()
    for other in pairs_config()["pairs"]:
        if other == pair:
            continue
        for s in read_lines(PROCESSED / other / "train.en"):
            n = len(s.split())
            if s in exclude or s in seen or not (cfg["min_words"] <= n <= cfg["max_words"]):
                continue
            seen.add(s)
            pool.append(s)
    random.Random(cfg["seed"]).shuffle(pool)
    return pool


def generate(pair: str, limit: int | None = None) -> None:
    cfg = load_yaml(CFG)
    set_seed(cfg["seed"])
    x = pair.split("-", 1)[1]
    out = OUTPUTS / "synthetic" / pair
    device = get_device()
    n_real = len(read_lines(PROCESSED / pair / "train.en"))
    pool = mono_pool(pair, cfg)
    n = min(len(pool), int(n_real * cfg["synthetic_ratio"]), limit or 10**9)
    mono = pool[:n]
    tok = SPTokenizer(PROCESSED / pair / "spm.model")
    fwd, _ = load_model(OUTPUTS / "baseline" / f"en-{x}" / "checkpoint_best.pt", device)
    t0 = time.time()
    syn = translate_lines(fwd, tok, mono, device, beam=cfg["generator_beam"],
                          batch_size=cfg["generator_batch_size"])
    gen_time = time.time() - t0
    keep = [(m, s) for m, s in zip(mono, syn) if s.strip()]
    write_lines(out / "mono.en", (m for m, _ in keep))
    write_lines(out / f"syn.{x}", (s for _, s in keep))

    # quality estimate: round-trip en -> X(syn) -> en with the baseline X->en model
    bwd, _ = load_model(OUTPUTS / "baseline" / f"{x}-en" / "checkpoint_best.pt", device)
    k = min(cfg["roundtrip_sample"], len(keep))
    rt = translate_lines(bwd, tok, [s for _, s in keep[:k]], device, beam=cfg["generator_beam"],
                         batch_size=cfg["generator_batch_size"])
    ref = [m for m, _ in keep[:k]]
    q = {"pair": pair, "n_real_train": n_real, "mono_pool_size": len(pool), "n_synthetic": len(keep),
         "synthetic_ratio": round(len(keep) / n_real, 3), "generator": f"baseline en-{x}, beam {cfg['generator_beam']}",
         "generation_time_s": round(gen_time, 1),
         "roundtrip_chrf++": round(sacrebleu.corpus_chrf(rt, [ref], word_order=2).score, 2),
         "roundtrip_bleu": round(sacrebleu.corpus_bleu(rt, [ref]).score, 2),
         "roundtrip_sample": k,
         "avg_len_ratio_syn_over_src": round(sum(len(s) for _, s in keep) / max(1, sum(len(m) for m, _ in keep)), 3),
         "examples": [{"en": m, "syn": s, "roundtrip": r} for (m, s), r in list(zip(keep[:k], rt))[:5]]}
    (out / "quality.json").write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[synthetic/{pair}] {len(keep)} synthetic pairs (pool {len(pool)}), round-trip chrF++ "
          f"{q['roundtrip_chrf++']}", flush=True)


def train(direction: str, extra_args: list[str]) -> None:
    cfg = load_yaml(CFG)
    src_l, tgt_l, pair = split_direction(direction)
    x = pair.split("-", 1)[1]
    syn_dir = OUTPUTS / "synthetic" / pair
    if src_l == "en":   # forward translation / self-training: real en -> synthetic X
        es, et = syn_dir / "mono.en", syn_dir / f"syn.{x}"
    else:               # back-translation: synthetic X -> real en
        es, et = syn_dir / f"syn.{x}", syn_dir / "mono.en"
    argv = ["--direction", direction, "--model-name", "backtranslation",
            "--extra-src", str(es), "--extra-tgt", str(et)] + (["--tag-extra"] if cfg["tag"] else [])
    train_main(argv + extra_args)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--pair", required=True)
    g.add_argument("--limit", type=int, help="debug only: number of mono sentences")
    t = sub.add_parser("train")
    t.add_argument("--direction", required=True)
    args, rest = ap.parse_known_args()
    if args.cmd == "generate":
        generate(args.pair, args.limit)
    else:
        train(args.direction, rest)


if __name__ == "__main__":
    main()

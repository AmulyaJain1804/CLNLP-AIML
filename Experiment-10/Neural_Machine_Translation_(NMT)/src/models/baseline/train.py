"""Train the from-scratch Transformer for one translation direction (M3; reused by M4 back-translation).

Writes outputs/{model_name}/{direction}/:
    checkpoint_best.pt   best dev-chrF++ weights      train_log.csv   step-level log
    dev.hyp / test.hyp   beam-search translations      run_config.json seed, hparams, hardware, time, git

Examples:
    python -m src.models.baseline.train --direction en-as
    python -m src.models.baseline.train --direction en-as --train-frac 0.25 --model-name baseline_frac0.25
    python -m src.models.baseline.train --direction as-en --model-name backtranslation \
        --extra-src outputs/synthetic/en-as/syn.as --extra-tgt outputs/synthetic/en-as/mono.en --tag-extra
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import time

import sacrebleu
import torch
import torch.nn.functional as F

from src.models.baseline.data import SPTokenizer, make_batches, sentence_batches, tensors_for
from src.models.baseline.model import PAD, ModelArgs, TransformerNMT, beam_decode, greedy_decode
from src.utils import OUTPUTS, PROCESSED, RunRecorder, load_yaml, read_lines, set_seed, split_direction, write_lines

BT_TAG = "<BT>"


def get_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(device):
    if device.type != "cuda":
        return None
    # bf16 only on Ampere+ (A100 etc.); T4 (sm_75) reports bf16 "supported" but only emulates it -> very slow
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def translate_lines(model, tok, lines, device, beam=5, batch_size=64, max_len=200, **kw):
    ids = tok.encode(lines, max_len=max_len)
    out = [None] * len(lines)
    dt = amp_dtype(device)
    for idx, src in sentence_batches(ids, batch_size):
        with torch.autocast(device.type, dtype=dt, enabled=dt is not None):
            hyp = beam_decode(model, src.to(device), beam=beam, **kw) if beam > 1 \
                else greedy_decode(model, src.to(device), kw.get("max_len_a", 1.5), kw.get("max_len_b", 10))
        for i, h in zip(idx, tok.decode(hyp)):
            out[i] = h
    return out


@torch.no_grad()
def dev_loss(model, batches, src_ids, tgt_ids, device):
    model.eval()
    tot, n = 0.0, 0
    dt = amp_dtype(device)
    for b in batches:
        src, tin, tout = (t.to(device) for t in tensors_for(b, src_ids, tgt_ids))
        with torch.autocast(device.type, dtype=dt, enabled=dt is not None):
            logits = model(src, tin)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), tout.reshape(-1),
                               ignore_index=PAD, reduction="sum")
        tot += loss.item()
        n += (tout != PAD).sum().item()
    return tot / max(1, n)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TransformerNMT(ModelArgs(**ck["model_args"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, help="e.g. en-as or as-en")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--model-name", default="baseline")
    ap.add_argument("--train-frac", type=float, default=1.0, help="learning-curve subsampling")
    ap.add_argument("--extra-src", help="synthetic source file (back-translation)")
    ap.add_argument("--extra-tgt", help="synthetic target file (back-translation)")
    ap.add_argument("--tag-extra", action="store_true", help=f"prepend {BT_TAG} to synthetic sources")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--limit-eval", type=int, help="debug only: truncate dev/test to N sentences")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    tc, dc = cfg["train"], cfg["decode"]
    if args.max_steps:
        tc["max_steps"] = args.max_steps
    if args.eval_every:
        tc["eval_every"] = args.eval_every
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = get_device(args.device)
    src_l, tgt_l, pair = split_direction(args.direction)
    pdir = PROCESSED / pair
    out_dir = OUTPUTS / args.model_name / args.direction
    tok = SPTokenizer(pdir / "spm.model")

    tr_src, tr_tgt = read_lines(pdir / f"train.{src_l}"), read_lines(pdir / f"train.{tgt_l}")
    if args.train_frac < 1.0:
        idx = sorted(random.Random(seed).sample(range(len(tr_src)), int(len(tr_src) * args.train_frac)))
        tr_src, tr_tgt = [tr_src[i] for i in idx], [tr_tgt[i] for i in idx]
    n_real = len(tr_src)
    n_syn = 0
    if args.extra_src:
        ex_src, ex_tgt = read_lines(args.extra_src), read_lines(args.extra_tgt)
        assert len(ex_src) == len(ex_tgt)
        if args.tag_extra:
            ex_src = [f"{BT_TAG} {s}" for s in ex_src]
        tr_src, tr_tgt = tr_src + ex_src, tr_tgt + ex_tgt
        n_syn = len(ex_src)
    dv_src, dv_tgt = read_lines(pdir / f"dev.{src_l}"), read_lines(pdir / f"dev.{tgt_l}")
    te_src = read_lines(pdir / f"test.{src_l}")
    if args.limit_eval:
        dv_src, dv_tgt, te_src = dv_src[:args.limit_eval], dv_tgt[:args.limit_eval], te_src[:args.limit_eval]

    ml = tc["max_len"]
    s_ids, t_ids = tok.encode(tr_src, ml), tok.encode(tr_tgt, ml)
    ds_ids, dt_ids = tok.encode(dv_src, ml), tok.encode(dv_tgt, ml)
    dev_batches = make_batches(ds_ids, dt_ids, tc["max_tokens"], shuffle=False)

    margs = ModelArgs(vocab_size=tok.vocab_size, **cfg["model"])
    model = TransformerNMT(margs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=tuple(tc["betas"]),
                            weight_decay=tc["weight_decay"])
    warm = tc["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, math.sqrt(warm / (s + 1))))
    dt = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=dt == torch.float16)

    run_cfg = {"model_name": args.model_name, "direction": args.direction, "pair": pair,
               "model_args": margs.to_dict(), "train": tc, "decode": dc,
               "train_frac": args.train_frac, "n_train_real": n_real, "n_train_synthetic": n_syn,
               "synthetic_files": [args.extra_src, args.extra_tgt] if args.extra_src else None,
               "tagged_bt": args.tag_extra, "n_params": n_params, "spm": str(pdir / "spm.model"),
               "amp_dtype": str(dt), "device": str(device)}
    rec = RunRecorder(out_dir, run_cfg, seed)
    print(f"[{args.model_name}/{args.direction}] params={n_params/1e6:.1f}M train={len(s_ids)} "
          f"(real {n_real}, synthetic {n_syn}) dev={len(ds_ids)} device={device}", flush=True)

    best, best_state, bad, step, epoch = -1.0, None, 0, 0, 0
    stop = False
    running, run_n = 0.0, 0
    while not stop and epoch < tc["max_epochs"]:
        epoch += 1
        for b in make_batches(s_ids, t_ids, tc["max_tokens"], shuffle=True, seed=seed + epoch):
            model.train()
            src, tin, tout = (t.to(device) for t in tensors_for(b, s_ids, t_ids))
            with torch.autocast(device.type, dtype=dt, enabled=dt is not None):
                logits = model(src, tin)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), tout.reshape(-1),
                                   ignore_index=PAD, label_smoothing=tc["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_norm"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running += loss.item()
            run_n += 1
            if step % tc["eval_every"] == 0 or step == tc["max_steps"]:
                dl = dev_loss(model, dev_batches, ds_ids, dt_ids, device)
                hyps = translate_lines(model, tok, dv_src, device, beam=1, batch_size=128, max_len=ml)
                chrf = sacrebleu.corpus_chrf(hyps, [dv_tgt], word_order=2).score
                bleu = sacrebleu.corpus_bleu(hyps, [dv_tgt]).score
                tl = running / run_n
                running, run_n = 0.0, 0
                rec.log(step, epoch, tl, dl, chrf, bleu, sched.get_last_lr()[0])
                print(f"  step {step:6d} ep {epoch:3d} train_loss {tl:.3f} dev_loss {dl:.3f} "
                      f"dev_chrF++ {chrf:.2f} dev_BLEU {bleu:.2f}", flush=True)
                if chrf > best:
                    best, bad = chrf, 0
                    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                    torch.save({"model_args": margs.to_dict(), "state_dict": best_state, "step": step,
                                "direction": args.direction, "dev_chrf": chrf}, out_dir / "checkpoint_best.pt")
                else:
                    bad += 1
                if bad >= tc["patience"]:
                    print("  early stopping", flush=True)
                    stop = True
            if step >= tc["max_steps"]:
                stop = True
            if stop:
                break

    model.load_state_dict(best_state)
    t0 = time.time()
    kw = dict(beam=dc["beam"], batch_size=dc["batch_size"], max_len=ml, len_penalty=dc["len_penalty"],
              max_len_a=dc["max_len_a"], max_len_b=dc["max_len_b"])
    dev_h = translate_lines(model, tok, dv_src, device, **kw)
    test_h = translate_lines(model, tok, te_src, device, **kw)
    write_lines(out_dir / "dev.hyp", dev_h)
    write_lines(out_dir / "test.hyp", test_h)
    dev_final = sacrebleu.corpus_chrf(dev_h, [dv_tgt], word_order=2).score
    rec.save(best_dev_chrf_greedy=best, dev_chrf_beam=dev_final, steps=step, epochs=epoch,
             decode_time_s=round(time.time() - t0, 1), finished=True)
    print(f"[{args.model_name}/{args.direction}] done: best dev chrF++ (greedy) {best:.2f}, "
          f"dev chrF++ (beam {dc['beam']}) {dev_final:.2f}", flush=True)


if __name__ == "__main__":
    main()

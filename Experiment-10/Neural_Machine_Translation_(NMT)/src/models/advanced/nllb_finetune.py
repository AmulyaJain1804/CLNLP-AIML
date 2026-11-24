"""M4 advanced model A: fine-tune NLLB-200-distilled-600M on one direction.

Transfer learning from a massively multilingual pretrained model. Assamese, Manipuri
(Bengali script) and Mizo are NLLB-200 languages; Bodo is not, so it is trained with the
hin_Deva tag as a proxy (same Devanagari script) -- a pure transfer experiment.

Writes outputs/nllb_finetune/{direction}/: dev.hyp, test.hyp, zeroshot_dev.hyp,
train_log.csv, run_config.json (and checkpoint/ if save_checkpoint: true).
Step 0 of train_log.csv is the zero-shot (no fine-tuning) dev score.

Usage:  python -m src.models.advanced.nllb_finetune --direction en-as
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import time

import sacrebleu
import torch

from src.utils import (OUTPUTS, PROCESSED, RunRecorder, load_yaml, pairs_config, read_lines, set_seed,
                       split_direction, write_lines)


def amp_dtype(device):
    if device.type != "cuda":
        return None
    # bf16 only on Ampere+ (A100 etc.); T4 (sm_75) reports bf16 "supported" but only emulates it -> very slow
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


@torch.no_grad()
def generate(model, tok, lines, tgt_code, device, beam, batch_size, max_new_tokens, max_len):
    model.eval()
    out = [None] * len(lines)
    order = sorted(range(len(lines)), key=lambda i: len(lines[i]))
    forced = tok.convert_tokens_to_ids(tgt_code)
    dt = amp_dtype(device)
    for k in range(0, len(order), batch_size):
        idx = order[k:k + batch_size]
        enc = tok([lines[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        with torch.autocast(device.type, dtype=dt, enabled=dt is not None):
            gen = model.generate(**enc, forced_bos_token_id=forced, num_beams=beam,
                                 max_new_tokens=max_new_tokens)
        for i, h in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = h.strip()
    return out


def main(argv=None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--config", default="configs/nllb.yaml")
    ap.add_argument("--model-name", default="nllb_finetune")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--limit-train", type=int, help="debug: use only N training pairs")
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, help="override train.batch_size (micro-batch)")
    ap.add_argument("--grad-accum", type=int, help="override train.grad_accum")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    tc, dc = cfg["train"], cfg["decode"]
    if args.max_steps:
        tc["max_steps"] = args.max_steps
    if args.eval_every:
        tc["eval_every"] = args.eval_every
    if args.batch_size:
        tc["batch_size"] = args.batch_size
    if args.grad_accum:
        tc["grad_accum"] = args.grad_accum
    seed = cfg["seed"]
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    src_l, tgt_l, pair = split_direction(args.direction)
    codes = {"en": cfg["en_code"], pair.split("-", 1)[1]: pairs_config()["pairs"][pair]["nllb_code"]}
    src_code, tgt_code = codes[src_l], codes[tgt_l]
    pdir = PROCESSED / pair
    out_dir = OUTPUTS / args.model_name / args.direction

    tok = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], src_lang=src_code, tgt_lang=tgt_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name_or_path"]).to(device)
    if tc.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    tr_src, tr_tgt = read_lines(pdir / f"train.{src_l}"), read_lines(pdir / f"train.{tgt_l}")
    if args.limit_train:
        tr_src, tr_tgt = tr_src[:args.limit_train], tr_tgt[:args.limit_train]
    dv_src, dv_tgt = read_lines(pdir / f"dev.{src_l}"), read_lines(pdir / f"dev.{tgt_l}")
    te_src = read_lines(pdir / f"test.{src_l}")
    gen_n = tc["dev_gen_sentences"]
    ml = tc["max_len"]

    def encode_pairs(src, tgt):
        enc = tok(src, text_target=tgt, truncation=True, max_length=ml)
        return list(zip(enc["input_ids"], enc["labels"]))

    train_ex = encode_pairs(tr_src, tr_tgt)
    dev_ex = encode_pairs(dv_src, dv_tgt)
    pad_id = tok.pad_token_id

    def collate(exs):
        S = max(len(s) for s, _ in exs)
        T = max(len(t) for _, t in exs)
        ids = torch.full((len(exs), S), pad_id)
        att = torch.zeros((len(exs), S), dtype=torch.long)
        lab = torch.full((len(exs), T), -100)
        for i, (s, t) in enumerate(exs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
            lab[i, :len(t)] = torch.tensor(t)
        return {"input_ids": ids.to(device), "attention_mask": att.to(device), "labels": lab.to(device)}

    def batches(exs, bs, shuffle, ep):
        # length-bucketed batches: sort in chunks of 100 batches, then shuffle batch order
        idx = list(range(len(exs)))
        rng = random.Random(seed + ep)
        if shuffle:
            rng.shuffle(idx)
        chunk = bs * 100
        out = []
        for k in range(0, len(idx), chunk):
            part = sorted(idx[k:k + chunk], key=lambda i: len(exs[i][0]))
            out += [part[j:j + bs] for j in range(0, len(part), bs)]
        if shuffle:
            rng.shuffle(out)
        return out

    dt = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=dt == torch.float16)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps_per_epoch = math.ceil(len(train_ex) / tc["batch_size"] / tc["grad_accum"])
    total = min(tc["max_steps"], steps_per_epoch * tc["max_epochs"])
    warm = tc["warmup_steps"]
    if not args.eval_every:  # small pairs: keep >= ~6 dev evaluations per run
        tc["eval_every"] = max(50, min(tc["eval_every"], total // 6))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    start_id = model.config.decoder_start_token_id

    def token_loss(batch, smoothing, reduction="mean"):
        """Loss over real target tokens only. NLLB's vocabulary has 256k entries, so we never build the
        full (batch x length x 256k) logits tensor: the LM head is applied to non-padding positions only."""
        lab = batch["labels"]
        dec_in = torch.full_like(lab, pad_id)
        dec_in[:, 0] = start_id
        dec_in[:, 1:] = lab[:, :-1].masked_fill(lab[:, :-1] == -100, pad_id)
        sel = lab != -100
        with torch.autocast(device.type, dtype=dt, enabled=dt is not None):
            hid = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              decoder_input_ids=dec_in).last_hidden_state
            logits = model.lm_head(hid[sel])
        return torch.nn.functional.cross_entropy(logits.float(), lab[sel], label_smoothing=smoothing,
                                                 reduction=reduction), int(sel.sum())

    @torch.no_grad()
    def eval_dev():
        model.eval()
        tot, n = 0.0, 0
        for b in batches(dev_ex, tc["batch_size"], False, 0):
            loss, k = token_loss(collate([dev_ex[i] for i in b]), 0.0, "sum")
            tot += loss.item()
            n += k
        hyps = generate(model, tok, dv_src[:gen_n], tgt_code, device, 1, dc["batch_size"], 200, ml)
        return (tot / n, sacrebleu.corpus_chrf(hyps, [dv_tgt[:gen_n]], word_order=2).score,
                sacrebleu.corpus_bleu(hyps, [dv_tgt[:gen_n]]).score)

    run_cfg = {"model_name": args.model_name, "direction": args.direction, "pair": pair,
               "pretrained": cfg["model_name_or_path"], "src_code": src_code, "tgt_code": tgt_code,
               "proxy_language_tag": pair == "en-bodo", "train": tc, "decode": dc,
               "n_train": len(train_ex), "total_steps": total, "amp_dtype": str(dt), "device": str(device),
               "n_params": sum(p.numel() for p in model.parameters())}
    rec = RunRecorder(out_dir, run_cfg, seed)
    print(f"[{args.model_name}/{args.direction}] {src_code}->{tgt_code} train={len(train_ex)} "
          f"steps={total} device={device}", flush=True)

    # zero-shot reference point (step 0): full dev, beam, before any fine-tuning
    zs = generate(model, tok, dv_src, tgt_code, device, dc["beam"], dc["batch_size"], dc["max_new_tokens"], ml)
    write_lines(out_dir / "zeroshot_dev.hyp", zs)
    zs_chrf = sacrebleu.corpus_chrf(zs, [dv_tgt], word_order=2).score
    zs_bleu = sacrebleu.corpus_bleu(zs, [dv_tgt]).score
    dl, chrf, bleu = eval_dev()
    rec.log(0, 0, "", dl, chrf, bleu, 0.0)
    print(f"  zero-shot dev chrF++ (beam, full dev) {zs_chrf:.2f}  BLEU {zs_bleu:.2f}", flush=True)

    best, best_state, bad, step, ep = -1.0, None, 0, 0, 0
    running, run_n, micro = 0.0, 0, 0
    stop = False
    while not stop and ep < tc["max_epochs"]:
        ep += 1
        for b in batches(train_ex, tc["batch_size"], True, ep):
            model.train()
            loss, _ = token_loss(collate([train_ex[i] for i in b]), tc["label_smoothing"])
            scaler.scale(loss / tc["grad_accum"]).backward()
            running += loss.item()
            run_n += 1
            micro += 1
            if micro % tc["grad_accum"]:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_norm"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % tc["eval_every"] == 0 or step == total:
                dl, chrf, bleu = eval_dev()
                tl = running / run_n
                running, run_n = 0.0, 0
                rec.log(step, ep, tl, dl, chrf, bleu, sched.get_last_lr()[0])
                print(f"  step {step:5d} ep {ep} train_loss {tl:.3f} dev_loss {dl:.3f} "
                      f"dev_chrF++({gen_n}) {chrf:.2f} BLEU {bleu:.2f}", flush=True)
                if chrf > best:
                    best, bad = chrf, 0
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= tc["patience"]:
                        print("  early stopping", flush=True)
                        stop = True
            if step >= total:
                stop = True
            if stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    del opt
    torch.cuda.empty_cache() if device.type == "cuda" else None
    t0 = time.time()
    dev_h = generate(model, tok, dv_src, tgt_code, device, dc["beam"], dc["batch_size"], dc["max_new_tokens"], ml)
    test_h = generate(model, tok, te_src, tgt_code, device, dc["beam"], dc["batch_size"], dc["max_new_tokens"], ml)
    write_lines(out_dir / "dev.hyp", dev_h)
    write_lines(out_dir / "test.hyp", test_h)
    if cfg.get("save_checkpoint"):
        model.half().save_pretrained(out_dir / "checkpoint")
        tok.save_pretrained(out_dir / "checkpoint")
    dev_final = sacrebleu.corpus_chrf(dev_h, [dv_tgt], word_order=2).score
    rec.save(zeroshot_dev_chrf=zs_chrf, zeroshot_dev_bleu=zs_bleu, best_dev_chrf_subset=best,
             dev_chrf_beam=dev_final, steps=step, epochs=ep, decode_time_s=round(time.time() - t0, 1),
             finished=True)
    print(f"[{args.model_name}/{args.direction}] done: zero-shot {zs_chrf:.2f} -> fine-tuned dev chrF++ "
          f"{dev_final:.2f}", flush=True)


if __name__ == "__main__":
    main()

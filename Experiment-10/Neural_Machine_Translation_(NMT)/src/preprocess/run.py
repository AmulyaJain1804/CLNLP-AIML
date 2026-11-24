"""Stage 2 (M2): clean, filter, deduplicate and SentencePiece-tokenize every pair.

Input : data/raw/{pair}/{split}.{src,tgt}
Output: data/processed/{pair}/{split}.{en,X}          cleaned plain text
        data/processed/{pair}/{split}.tok.{en,X}      SentencePiece pieces
        data/processed/{pair}/spm.model / spm.vocab   joint unigram model (train split only)
        data/processed/{pair}/preprocess_config.yaml
        data/processed/{pair}/stats.json              filter counts, overlap, scripts, vocab

Deterministic: same input + config -> byte-identical output.
Usage:  python -m src.preprocess.run [--pairs en-as en-mni]
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
from collections import Counter

import sentencepiece as spm

from src.preprocess.clean import broken_marks, filter_reason, normalize, script_profile
from src.utils import PROCESSED, RAW, ROOT, load_yaml, pairs_config, read_lines, write_lines

CFG_PATH = "configs/preprocess.yaml"


def process_pair(pair: str, cfg: dict) -> dict:
    x = pair.split("-", 1)[1]
    out = PROCESSED / pair
    out.mkdir(parents=True, exist_ok=True)
    f = cfg["filters"]
    stats: dict = {"pair": pair, "filters": {}, "splits": {}}

    data = {}
    marks_before = marks_after = 0
    for split in ("train", "dev", "test"):
        src_raw = read_lines(RAW / pair / f"{split}.src")
        tgt_raw = read_lines(RAW / pair / f"{split}.tgt")
        assert len(src_raw) == len(tgt_raw)
        marks_before += sum(broken_marks(t) for t in tgt_raw)
        src = [normalize(s, cfg) for s in src_raw]
        tgt = [normalize(t, cfg) for t in tgt_raw]
        marks_after += sum(broken_marks(t) for t in tgt)
        data[split] = (src, tgt)
    stats["broken_combining_marks"] = {"before_norm": marks_before, "after_norm": marks_after}

    # dev/test: drop empty pairs only
    for split in ("dev", "test"):
        src, tgt = data[split]
        keep = [(s, t) for s, t in zip(src, tgt) if s and t]
        stats["filters"][f"{split}_empty_removed"] = len(src) - len(keep)
        data[split] = ([s for s, _ in keep], [t for _, t in keep])

    # train: filters, dedup, overlap removal
    held_src = set(data["dev"][0]) | set(data["test"][0])
    held_tgt = set(data["dev"][1]) | set(data["test"][1])
    test_src, test_tgt = set(data["test"][0]), set(data["test"][1])
    reasons = Counter()
    seen = set()
    overlap_test = 0
    keep = []
    for s, t in zip(*data["train"]):
        r = filter_reason(s, t, f, cfg["expected_script"].get(x))
        if r is None and f.get("dedup_train") and (s, t) in seen:
            r = "duplicate"
        if r is None and f.get("remove_dev_test_overlap") and (s in held_src or t in held_tgt):
            r = "dev_test_overlap"
            overlap_test += (s in test_src or t in test_tgt)
        if r:
            reasons[r] += 1
            continue
        seen.add((s, t))
        keep.append((s, t))
    stats["filters"]["train_input"] = len(data["train"][0])
    stats["filters"]["train_removed"] = dict(sorted(reasons.items()))
    stats["filters"]["train_test_overlap_removed"] = overlap_test
    data["train"] = ([s for s, _ in keep], [t for _, t in keep])

    for split, (src, tgt) in data.items():
        write_lines(out / f"{split}.en", src)
        write_lines(out / f"{split}.{x}", tgt)
        stats["splits"][split] = len(src)

    # scripts
    stats["scripts"] = {
        lang: {k: round(v / max(1, sum(prof.values())), 4) for k, v in prof.most_common(3)}
        for lang, prof in (("en", script_profile(data["train"][0])),
                           (x, script_profile(data["train"][1])))}

    # SentencePiece: joint model trained on TRAIN only (no dev/test information)
    sp_cfg = cfg["sentencepiece"]
    vocab = sp_cfg["vocab_size"].get(pair, sp_cfg["vocab_size"]["default"])
    corpus = "\n".join(data["train"][0] + data["train"][1])
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus.split("\n")), model_writer=model,
        model_type=sp_cfg["model_type"], vocab_size=vocab,
        character_coverage=sp_cfg["character_coverage"],
        user_defined_symbols=sp_cfg["user_defined_symbols"],
        # corpus < input_sentence_size, so no sampling happens; no shuffle + 1 thread -> deterministic
        input_sentence_size=sp_cfg["input_sentence_size"], shuffle_input_sentence=False,
        seed_sentencepiece_size=1000000, num_threads=1,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3, minloglevel=2)
    (out / "spm.model").write_bytes(model.getvalue())
    sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
    with open(out / "spm.vocab", "w", encoding="utf-8", newline="\n") as fv:
        for i in range(sp.get_piece_size()):
            fv.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")

    tok_stats = {}
    for split, (src, tgt) in data.items():
        for lang, lines in (("en", src), (x, tgt)):
            pieces = sp.encode(lines, out_type=str)
            write_lines(out / f"{split}.tok.{lang}", (" ".join(p) for p in pieces))
            if split == "train":
                tok_stats[lang] = round(sum(map(len, pieces)) / max(1, len(pieces)), 2)
    stats["spm"] = {"vocab_size": sp.get_piece_size(), "type": sp_cfg["model_type"],
                    "avg_tokens_per_sentence_train": tok_stats,
                    "unk_rate_dev": {lang: round(_unk_rate(sp, lines), 5) for lang, lines in
                                     (("en", data["dev"][0]), (x, data["dev"][1]))}}

    shutil.copy(ROOT / CFG_PATH, out / "preprocess_config.yaml")
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def _unk_rate(sp, lines) -> float:
    ids = [i for line in lines for i in sp.encode(line)]
    return sum(i == sp.unk_id() for i in ids) / max(1, len(ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", help="default: all pairs in configs/pairs.yaml")
    args = ap.parse_args()
    cfg = load_yaml(CFG_PATH)
    pairs = args.pairs or list(pairs_config()["pairs"])
    for pair in pairs:
        s = process_pair(pair, cfg)
        print(f"{pair:12s} train={s['splits']['train']:6d} dev={s['splits']['dev']:5d} "
              f"test={s['splits']['test']:5d} removed={s['filters']['train_removed']} "
              f"vocab={s['spm']['vocab_size']} tok/sent={s['spm']['avg_tokens_per_sentence_train']}")


if __name__ == "__main__":
    main()

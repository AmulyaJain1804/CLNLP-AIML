"""Token-bucketed batching for the from-scratch Transformer."""
from __future__ import annotations

import random

import sentencepiece as spm
import torch

from src.models.baseline.model import BOS, EOS, PAD


class SPTokenizer:
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, lines, max_len=None):
        ids = self.sp.encode(list(lines))
        if max_len:
            ids = [x[:max_len] for x in ids]
        return ids

    def decode(self, ids_list):
        return [self.sp.decode(ids) for ids in ids_list]


def pad_batch(seqs, prefix=(), suffix=(EOS,)):
    seqs = [list(prefix) + s + list(suffix) for s in seqs]
    T = max(map(len, seqs))
    out = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def make_batches(src_ids, tgt_ids, max_tokens, shuffle=True, seed=0):
    """Group sentences of similar length so each batch has <= max_tokens (incl. padding)."""
    order = sorted(range(len(src_ids)), key=lambda i: (len(src_ids[i]), len(tgt_ids[i])))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = max(len(src_ids[i]), len(tgt_ids[i])) + 2
        if cur and max(cur_max, L) * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    if shuffle:
        random.Random(seed).shuffle(batches)
    return batches


def tensors_for(batch, src_ids, tgt_ids):
    src = pad_batch([src_ids[i] for i in batch])
    tgt = pad_batch([tgt_ids[i] for i in batch], prefix=(BOS,))
    return src, tgt[:, :-1], tgt[:, 1:]


def sentence_batches(ids, batch_size):
    """For inference: length-sorted batches, returns (original_indices, tensor) tuples."""
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    for k in range(0, len(order), batch_size):
        idx = order[k:k + batch_size]
        yield idx, pad_batch([ids[i] for i in idx])

"""Small pre-norm Transformer encoder-decoder for low-resource NMT (M3).

Joint SentencePiece vocabulary per pair -> one shared embedding matrix (encoder input,
decoder input and output projection tied; Press & Wolf, 2017).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, BOS, EOS = 0, 1, 2, 3


@dataclass
class ModelArgs:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    ff_dim: int = 1024
    enc_layers: int = 4
    dec_layers: int = 4
    dropout: float = 0.3
    attn_dropout: float = 0.1
    max_positions: int = 512

    def to_dict(self):
        return asdict(self)


class MultiHeadAttention(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.h, self.dk = h, d // h
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.dropout = dropout

    def forward(self, x, mem, mask, return_attn=False):
        """mask: bool, broadcastable to (B, h, Tq, Tk); True = attend."""
        B, Tq, D = x.shape
        Tk = mem.size(1)
        q = self.q(x).view(B, Tq, self.h, self.dk).transpose(1, 2)
        k = self.k(mem).view(B, Tk, self.h, self.dk).transpose(1, 2)
        v = self.v(mem).view(B, Tk, self.h, self.dk).transpose(1, 2)
        if return_attn:
            s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
            s = s.masked_fill(~mask, float("-inf"))
            a = s.softmax(-1)
            out = a @ v
        else:
            a = None
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, Tq, D)
        return self.o(out), a


class FeedForward(nn.Sequential):
    def __init__(self, d, ff, dropout):
        super().__init__(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d))


class EncoderLayer(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(a.d_model), nn.LayerNorm(a.d_model)
        self.att = MultiHeadAttention(a.d_model, a.n_heads, a.attn_dropout)
        self.ff = FeedForward(a.d_model, a.ff_dim, a.dropout)
        self.drop = nn.Dropout(a.dropout)

    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.drop(self.att(h, h, mask)[0])
        return x + self.drop(self.ff(self.n2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(a.d_model) for _ in range(3))
        self.self_att = MultiHeadAttention(a.d_model, a.n_heads, a.attn_dropout)
        self.cross_att = MultiHeadAttention(a.d_model, a.n_heads, a.attn_dropout)
        self.ff = FeedForward(a.d_model, a.ff_dim, a.dropout)
        self.drop = nn.Dropout(a.dropout)

    def forward(self, y, mem, self_mask, mem_mask, return_attn=False):
        h = self.n1(y)
        y = y + self.drop(self.self_att(h, h, self_mask)[0])
        c, attn = self.cross_att(self.n2(y), mem, mem_mask, return_attn)
        y = y + self.drop(c)
        return y + self.drop(self.ff(self.n3(y))), attn


def sinusoidal(n, d):
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerNMT(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        d = args.d_model
        self.embed = nn.Embedding(args.vocab_size, d, padding_idx=PAD)
        nn.init.normal_(self.embed.weight, 0, d ** -0.5)
        self.register_buffer("pe", sinusoidal(args.max_positions, d), persistent=False)
        self.enc = nn.ModuleList(EncoderLayer(args) for _ in range(args.enc_layers))
        self.dec = nn.ModuleList(DecoderLayer(args) for _ in range(args.dec_layers))
        self.enc_norm, self.dec_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(args.dropout)
        self.scale = math.sqrt(d)

    def _emb(self, ids):
        T = ids.size(1)
        return self.drop(self.embed(ids) * self.scale + self.pe[:T].to(self.embed.weight.dtype))

    def encode(self, src):
        mask = (src != PAD)[:, None, None, :]  # B,1,1,S
        x = self._emb(src)
        for layer in self.enc:
            x = layer(x, mask)
        return self.enc_norm(x), mask

    def decode(self, tgt_in, mem, mem_mask, return_attn=False):
        T = tgt_in.size(1)
        causal = torch.ones(T, T, dtype=torch.bool, device=tgt_in.device).tril()
        self_mask = causal[None, None] & (tgt_in != PAD)[:, None, None, :]
        y = self._emb(tgt_in)
        attn = None
        for layer in self.dec:
            y, attn = layer(y, mem, self_mask, mem_mask, return_attn)
        logits = self.dec_norm(y) @ self.embed.weight.t()
        return (logits, attn) if return_attn else logits

    def forward(self, src, tgt_in):
        mem, mask = self.encode(src)
        return self.decode(tgt_in, mem, mask)


@torch.no_grad()
def greedy_decode(model, src, max_len_a=1.5, max_len_b=10):
    model.eval()
    mem, mask = model.encode(src)
    B = src.size(0)
    max_len = int((src != PAD).sum(1).max().item() * max_len_a + max_len_b)
    ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    for _ in range(max_len):
        nxt = model.decode(ys, mem, mask)[:, -1].argmax(-1)
        nxt = nxt.masked_fill(done, PAD)
        ys = torch.cat([ys, nxt[:, None]], 1)
        done |= nxt == EOS
        if done.all():
            break
    return [_strip(row) for row in ys[:, 1:].tolist()]


@torch.no_grad()
def beam_decode(model, src, beam=5, max_len_a=1.5, max_len_b=10, len_penalty=1.0):
    """Batched beam search (decoder re-run on the full prefix each step; fine for short sentences)."""
    if beam == 1:
        return greedy_decode(model, src, max_len_a, max_len_b)
    model.eval()
    B = src.size(0)
    dev = src.device
    mem, mask = model.encode(src)
    mem, mask = mem.repeat_interleave(beam, 0), mask.repeat_interleave(beam, 0)
    max_len = int((src != PAD).sum(1).max().item() * max_len_a + max_len_b)
    seqs = torch.full((B * beam, 1), BOS, dtype=torch.long, device=dev)
    scores = torch.zeros(B, beam, device=dev)
    scores[:, 1:] = float("-inf")
    finished = [[] for _ in range(B)]
    done = [False] * B
    for t in range(max_len):
        logp = model.decode(seqs, mem, mask)[:, -1].float().log_softmax(-1)  # B*beam, V
        V = logp.size(-1)
        cand = (scores.view(-1, 1) + logp).view(B, beam * V)
        top_s, top_i = cand.topk(2 * beam, dim=1)
        top_s, top_i = top_s.tolist(), top_i.tolist()
        new_rows, new_scores = [], []
        for b in range(B):
            rows, sc = [], []
            if not done[b]:
                for s, i in zip(top_s[b], top_i[b]):
                    if s == float("-inf"):
                        break
                    bi, tok = divmod(i, V)
                    row = b * beam + bi
                    if tok == EOS:
                        if len(finished[b]) < beam:
                            finished[b].append((s / ((t + 1) ** len_penalty), seqs[row, 1:].tolist()))
                    else:
                        rows.append((row, tok))
                        sc.append(s)
                    if len(rows) == beam:
                        break
                if len(finished[b]) >= beam or not rows:
                    done[b] = True
            while len(rows) < beam:  # pad dead/finished beams
                rows.append((b * beam, EOS))
                sc.append(float("-inf"))
            new_rows += rows
            new_scores += sc
        if all(done):
            break
        idx = torch.tensor([r for r, _ in new_rows], device=dev)
        toks = torch.tensor([k for _, k in new_rows], device=dev)
        seqs = torch.cat([seqs[idx], toks[:, None]], 1)
        scores = torch.tensor(new_scores, device=dev).view(B, beam)
    out = []
    for b in range(B):
        if not finished[b]:  # nothing finished: take best alive hypothesis
            out.append(_strip(seqs[b * beam + int(scores[b].argmax()), 1:].tolist()))
        else:
            out.append(max(finished[b], key=lambda x: x[0])[1])
    return out


def _strip(ids):
    res = []
    for i in ids:
        if i in (EOS, PAD):
            break
        res.append(i)
    return res

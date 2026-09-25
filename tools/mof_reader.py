"""MoF micro-reader (layer-1): a small transformer over move-tokens.

Token per legal move: [mf(17) || witness(138)] with the position
features x(2746) projected as a shared context vector. 2-layer
transformer, ~3M params. Output: one logit per move; loss = approved-
set NLL (-log sum_{m in approved} p(m)) per the tolerance doctrine.

Train on all DEGM2 rows except a held-out 1,000; report held-out
top-1-in-approved and exact-best accuracy, plus the kNN-router
baseline on the same rows. Weights: singles/reader/reader.pt
"""
import sys
import os
import json
import random
import time

os.environ.setdefault("ANORM", "1")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import torch.nn as nn
import numpy as np
import flyfeat_cb
import fly_curriculum as fc

WIT = "/home/spec/chess-lab/singles/witness"
OUT = "/home/spec/chess-lab/singles/reader"
D_MODEL = 256
EPOCHS = 30
BS = 64
LR = 1e-3
HOLDOUT = 1000
NF = int(os.environ.get("MOF_N", "138"))      # witness flies used
os.makedirs(OUT, exist_ok=True)


class Reader(nn.Module):
    def __init__(self, n_fly, f_move=17, f_pos=2746, d=D_MODEL):
        super().__init__()
        self.tok = nn.Linear(f_move + n_fly, d)
        self.xproj = nn.Linear(f_pos, d)
        layer = nn.TransformerEncoderLayer(
            d, nhead=4, dim_feedforward=512, dropout=0.1,
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(d, 1)

    def forward(self, toks, x, mask):
        h = self.tok(toks) + self.xproj(x).unsqueeze(1)
        h = self.enc(h, src_key_padding_mask=~mask)
        z = self.head(h).squeeze(-1)              # [B, M]
        return z.masked_fill(~mask, -1e9)


def load_meta():
    meta = json.load(open(f"{WIT}/meta.json"))
    dims = json.load(open(f"{WIT}/dims.json"))
    W = np.load(f"{WIT}/witness.npy", mmap_mode="r")
    mf = np.load(f"{WIT}/tokens_mf.npy", mmap_mode="r")
    X = np.load(f"{WIT}/pos_x.npy", mmap_mode="r")
    ap = np.load(f"{WIT}/approvals.npy")
    return meta, dims, W, mf, X, ap


def best_indices(fens):
    """index of the row's best move within legal-move order."""
    out = []
    pool_cache = {}
    for fen in fens:
        b = chess.Board(fen)
        mus = [m.uci() for m in b.legal_moves]
        out.append(-1)                            # filled below
        out[-1] = (b, mus)
    return out


def main():
    flyfeat_cb.feat_vec(chess.Board())
    meta, dims, W, mf, X, ap = load_meta()
    N = len(meta["fens"])
    offs = meta["offs"]
    Ls = meta["L"]
    n_fly = min(NF, W.shape[0])
    print(json.dumps({"rows": N, "T": W.shape[1], "flies": n_fly}),
          flush=True)

    # best move index per row (from the pool rows' best uci)
    rows = fc.load_all() if hasattr(fc, "load_all") else None
    import glob as _g
    pool_rows = []
    pools = sorted(os.path.basename(f)[:-6] for f in _g.glob(
        "/home/spec/chess-lab/tbpools/DEGM2_Ch*.jsonl"))
    for pool in pools:
        pool_rows += fc.load_pools([pool])
    best_idx = np.full(N, -1, np.int64)
    for i, fen in enumerate(meta["fens"]):
        b = chess.Board(fen)
        mus = [m.uci() for m in b.legal_moves]
        bu = pool_rows[i].get("best")
        if bu in mus:
            best_idx[i] = mus.index(bu)

    rng = random.Random(7)
    hold = set(rng.sample(range(N), HOLDOUT))
    train = [i for i in range(N) if i not in hold]
    hold_l = sorted(hold)
    print(json.dumps({"train": len(train), "holdout": len(hold_l)}),
          flush=True)

    dev = torch.device(fc.DEV)
    model = Reader(n_fly).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    nparam = sum(p.numel() for p in model.parameters())
    print(json.dumps({"params": nparam}), flush=True)

    def batch_rows(idxs):
        B = len(idxs)
        M = max(Ls[i] for i in idxs)
        toks = np.zeros((B, M, 17 + n_fly), np.float32)
        mask = np.zeros((B, M), bool)
        appr = np.zeros((B, M), bool)
        xb = np.zeros((B, X.shape[1]), np.float32)
        for b, i in enumerate(idxs):
            o, L = int(offs[i]), int(Ls[i])
            toks[b, :L, :17] = mf[o:o + L]
            toks[b, :L, 17:] = W[:n_fly, o:o + L].T
            mask[b, :L] = True
            appr[b, :L] = ap[o:o + L] > 0.5
            xb[b] = X[i]
        return (torch.from_numpy(toks).to(dev),
                torch.from_numpy(xb).to(dev),
                torch.from_numpy(mask).to(dev),
                torch.from_numpy(appr).to(dev))

    def approved_nll(logits, appr):
        lse_all = torch.logsumexp(logits, dim=1)
        neg = logits.masked_fill(~appr, -1e9)
        lse_ap = torch.logsumexp(neg, dim=1)
        return (lse_all - lse_ap).mean()

    def eval_set(idxs):
        model.eval()
        hit_ap = hit_best = 0
        with torch.no_grad():
            for s in range(0, len(idxs), 256):
                chunk = idxs[s:s + 256]
                toks, xb, mask, appr = batch_rows(chunk)
                logits = model(toks, xb, mask)
                pick = logits.argmax(1).tolist()
                for b, i in enumerate(chunk):
                    o, L = int(offs[i]), int(Ls[i])
                    if pick[b] < L:
                        if appr[b, pick[b]]:
                            hit_ap += 1
                        if pick[b] == best_idx[i]:
                            hit_best += 1
        model.train()
        return round(hit_ap / len(idxs), 4), \
            round(hit_best / len(idxs), 4)

    t0 = time.time()
    rng.shuffle(train)
    for ep in range(EPOCHS):
        tot = 0.0
        model.train()
        for s in range(0, len(train), BS):
            idxs = train[s:s + BS]
            toks, xb, mask, appr = batch_rows(idxs)
            logits = model(toks, xb, mask)
            loss = approved_nll(logits, appr)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        ha, hb = eval_set(hold_l)
        print(json.dumps({"epoch": ep, "loss": round(tot, 3),
                          "hold_top1_approved": ha,
                          "hold_top1_exact": hb,
                          "elapsed_s": round(time.time() - t0)}),
              flush=True)
    torch.save(model.state_dict(), f"{OUT}/reader.pt")
    json.dump({"hold_top1_approved": ha, "hold_top1_exact": hb,
               "params": nparam, "n_fly": n_fly},
              open(f"{OUT}/final.json", "w"), indent=1)
    print("READER-TRAIN-COMPLETE", flush=True)


if __name__ == "__main__":
    main()

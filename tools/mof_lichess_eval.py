"""Off-book MoF evaluation: 1,000 ranked-endgame Lichess puzzles
(4 rating bands x 250). The reader and the kNN router have NEVER seen
these positions; the flies' member banks are corpus-only.

Per puzzle: build position inputs from scratch, score with every fly
(witness computed here), run the reader -> top move vs the puzzle
solution. Baselines: kNN router (route to nearest-corpus fly, take its
pick), base model, random legal.

Output: singles/reader/lichess_eval.json
"""
import sys
import os
import json
import time

os.environ.setdefault("ANORM", "1")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc
from mof_reader import Reader

WIT = "/home/spec/chess-lab/singles/witness"
OUTD = "/home/spec/chess-lab/singles/reader"
EVAL = "/home/spec/chess-lab/tbpools/lichess_endgame_eval.jsonl"
PC_IDX = {chess.KING: 0, chess.ROOK: 1, chess.BISHOP: 2,
          chess.KNIGHT: 3, chess.PAWN: 4, chess.QUEEN: 5}


def pack_board(b):
    """inputs for forward() from a bare board (pack_pre conventions)."""
    mvs = list(b.legal_moves)
    L = len(mvs)
    p2 = {(pm.from_square, pm.to_square)
          for pm in b.pseudo_legal_moves}
    slot = np.zeros((1, L), np.int64)
    pcrow = np.zeros((1, L), np.int64)
    mfb = np.zeros((1, L, 17), np.float32)
    mask = np.zeros((1, L), bool)
    psb = np.zeros((1, L), np.float32)
    thb = np.zeros((1, L), np.float32)
    ps2b = np.zeros((1, L), np.float32)
    x = flyfeat_cb.feat_vec(b)[0][None, :]
    for j, mv in enumerate(mvs):
        slot[0, j] = mv.from_square * 64 + mv.to_square
        pc = b.piece_at(mv.from_square)
        pcrow[0, j] = PC_IDX[pc.piece_type] if pc else 0
        mfb[0, j] = flyfeat_cb.move_feats(b, mv)
        mask[0, j] = True
        psb[0, j] = 1.0 if (b.attacks_mask(mv.from_square)
                            & chess.BB_SQUARES[mv.to_square]) else 0.0
        b.push(mv)
        thb[0, j] = min(bin(b.attacks_mask(mv.to_square)
                            & b.occupied_co[b.turn]).count("1"), 4) / 4.0
        b.pop()
        ps2b[0, j] = 1.0 if (mv.from_square, mv.to_square) in p2 else 0.0
    return (x, slot, pcrow, mfb, mask, psb, thb, ps2b), mvs


def main():
    flyfeat_cb.feat_vec(chess.Board())
    puzzles = [json.loads(l) for l in open(EVAL)]
    N = len(puzzles)
    boards = [chess.Board(p["fen"]) for p in puzzles]
    packs = [pack_board(b) for b in boards]
    print(json.dumps({"puzzles": N}), flush=True)

    # fly witnesses for all puzzles
    z = np.load("/home/spec/chess-lab/singles/closure/deltas_2.npz")
    state = json.load(open(
        "/home/spec/chess-lab/singles/closure/models_2.json"))
    base = tp.build_model(0)
    base_state = {k: p.detach().clone()
                  for k, p in base.named_parameters()}
    base_cat = torch.cat([base_state[k].flatten()
                          for k, _ in base.named_parameters()])
    meta = json.load(open(f"{WIT}/meta.json"))
    dims = json.load(open(f"{WIT}/dims.json"))
    n_fly = min(138, len(state))
    Wl = np.zeros((n_fly, N, 64), np.float16)      # pad to 64 moves
    Ls = [p[0].shape[1] for p, _ in packs]
    Xl = np.stack([p[0][0] for p, _ in packs])

    def apply_fly(f):
        with torch.no_grad():
            for k, p in base.named_parameters():
                p.copy_(base_state[k])
            flat = base_cat.clone()
            flat.index_add_(0, torch.from_numpy(f["idx"]).to(fc.DEV),
                            torch.from_numpy(f["val"]).to(fc.DEV))
            off = 0
            for _, p in base.named_parameters():
                n_ = p.numel()
                p.copy_(flat[off:off + n_].view_as(p))
                off += n_

    base_scores = [None] * N
    t0 = time.time()
    for fi in range(n_fly):
        f = {"idx": z[f"m{fi:05d}_idx"], "val": z[f"m{fi:05d}_val"]}
        apply_fly(f)
        for s in range(0, N, 512):
            e = min(s + 512, N)
            chunks = packs[s:e]
            B = len(chunks)
            M = max(c[0].shape[1] for c in chunks)
            slotb = np.zeros((B, M), np.int64)
            pcrowb = np.zeros((B, M), np.int64)
            mfb = np.zeros((B, M, 17), np.float32)
            maskb = np.zeros((B, M), bool)
            psb = np.zeros((B, M), np.float32)
            thb = np.zeros((B, M), np.float32)
            ps2b = np.zeros((B, M), np.float32)
            xs = np.zeros((B, packs[0][0].shape[1]), np.float32)
            for b_i, (arrays, _) in enumerate(chunks):
                x, slot, pcrow, mfm, mask, psb_, thb_, ps2b_ = arrays
                L = x.shape[0] * 0 + maskb.shape[1] * 0 + mask[0].sum()
                L = int(mask.sum())
                slotb[b_i, :L] = slot[0, :L]
                pcrowb[b_i, :L] = pcrow[0, :L]
                mfb[b_i, :L] = mfm[0, :L]
                maskb[b_i, :L] = True
                psb[b_i, :L] = psb_[0, :L]
                thb[b_i, :L] = thb_[0, :L]
                ps2b[b_i, :L] = ps2b_[0, :L]
                xs[b_i] = x[0]
            Tt, _, _, _ = fc.forward(base, xs, slotb, pcrowb, mfb,
                                     maskb, psb, thb, ps2b)
            Tt = Tt.detach().cpu().numpy().astype(np.float16)
            for b_i in range(B):
                L = Ls[s + b_i]
                Wl[fi, s + b_i, :L] = Tt[b_i, :L]
        if fi % 25 == 0:
            print(json.dumps({"fly": fi,
                              "elapsed_s": round(time.time() - t0)}),
                  flush=True)
    # base model scores (init) for the baseline
    for k, p in base.named_parameters():
        p.copy_(base_state[k])
    for s in range(0, N, 512):
        e = min(s + 512, N)
        chunks = packs[s:e]
        B = len(chunks)
        M = max(c[0].shape[1] for c in chunks)
        slotb = np.zeros((B, M), np.int64)
        pcrowb = np.zeros((B, M), np.int64)
        mfb = np.zeros((B, M, 17), np.float32)
        maskb = np.zeros((B, M), bool)
        psb = np.zeros((B, M), np.float32)
        thb = np.zeros((B, M), np.float32)
        ps2b = np.zeros((B, M), np.float32)
        xs = np.zeros((B, packs[0][0].shape[1]), np.float32)
        for b_i, (arrays, _) in enumerate(chunks):
            x, slot, pcrow, mfm, mask, psb_, thb_, ps2b_ = arrays
            L = int(mask.sum())
            slotb[b_i, :L] = slot[0, :L]
            pcrowb[b_i, :L] = pcrow[0, :L]
            mfb[b_i, :L] = mfm[0, :L]
            maskb[b_i, :L] = True
            psb[b_i, :L] = psb_[0, :L]
            thb[b_i, :L] = thb_[0, :L]
            ps2b[b_i, :L] = ps2b_[0, :L]
            xs[b_i] = x[0]
        Tt, _, _, _ = fc.forward(base, xs, slotb, pcrowb, mfb, maskb,
                                 psb, thb, ps2b)
        Tt = Tt.detach().cpu().numpy()
        for b_i in range(B):
            base_scores[s + b_i] = Tt[b_i, :Ls[s + b_i]]

    # reader
    model = Reader(n_fly).to(fc.DEV)
    model.load_state_dict(torch.load(f"{OUTD}/reader.pt",
                                     map_location=fc.DEV))
    model.eval()
    # knn router assets: corpus bank
    pos_x = np.load(f"{WIT}/pos_x.npy", mmap_mode="r")
    Xn = np.asarray(pos_x, dtype=np.float32)
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-9
    meta = json.load(open(f"{WIT}/meta.json"))
    fly_midx = [[] for _ in range(n_fly)]
    # home fly per corpus row: approximate via closure membership
    for fi, m in enumerate(state):
        pass
    from collections import defaultdict
    owner = {}
    for fi, m in enumerate(state):
        for q in m["qs"]:
            owner[tuple(q)] = fi
    fen_to_row = {f: i for i, f in enumerate(meta["fens"])}
    pool_of = meta["pool"]
    for i, (pool, fen) in enumerate(zip(pool_of, meta["fens"])):
        fi = owner.get((pool, fen))
        if fi is not None and fi < n_fly:
            fly_midx[fi].append(i)
    fly_midx = [np.array(m, dtype=np.int64) if m else np.zeros(0, int)
                for m in fly_midx]

    res = []
    for pi, p in enumerate(puzzles):
        L = Ls[pi]
        toks = np.zeros((1, L, 17 + n_fly), np.float32)
        toks[0, :L, :17] = np.asarray(
            packs[pi][0][2][0, :L], dtype=np.float32)
        toks[0, :L, 17:] = Wl[:n_fly, pi, :L].T
        xb = Xl[pi][None, :]
        mask = np.zeros((1, 64), bool)
        mask[0, :L] = True
        with torch.no_grad():
            logits = model(torch.from_numpy(toks).to(fc.DEV),
                           torch.from_numpy(xb).to(fc.DEV),
                           torch.from_numpy(mask).to(fc.DEV))
        logits = logits[0, :L].cpu().numpy()
        mus = [m.uci() for m in boards[pi].legal_moves]
        top5 = [mus[j] for j in np.argsort(-logits)[:5]]
        pick_reader = top5[0]
        # kNN router baseline
        q = Xl[pi] / (np.linalg.norm(Xl[pi]) + 1e-9)
        sims = Xn @ q
        best_fi, best_s = -1, -2.0
        for fi in range(n_fly):
            if len(fly_midx[fi]) == 0:
                continue
            s_ = float(sims[fly_midx[fi]].max())
            if s_ > best_s:
                best_s, best_fi = s_, fi
        pick_knn = None
        if best_fi >= 0:
            pick_knn = mus[int(np.argmax(
                Wl[best_fi, pi, :L].astype(np.float32)))]
        pick_base = mus[int(np.argmax(base_scores[pi]))]
        res.append({"id": p["id"], "band": p["band"],
                    "rating": p["rating"],
                    "reader": pick_reader,
                    "reader_top5": top5,
                    "knn_fly": best_fi, "knn_pick": pick_knn,
                    "base_pick": pick_base,
                    "solution": p["solution"]})
    json.dump(res, open(f"{OUTD}/lichess_eval.json", "w"))
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for r in res:
        a = agg[r["band"]]
        a[0] += int(r["reader"] == r["solution"])
        a[1] += int(r["solution"] in r["reader_top5"])
        a[2] += int(r["knn_pick"] == r["solution"])
        a[3] += int(r["base_pick"] == r["solution"])
    out = {band: {"reader_top1": round(a[0] / 250, 3),
                  "reader_top5": round(a[1] / 250, 3),
                  "knn_top1": round(a[2] / 250, 3),
                  "base_top1": round(a[3] / 250, 3)}
           for band, a in sorted(agg.items())}
    tot = [sum(v) for v in zip(*agg.values())]
    out["ALL"] = {"reader_top1": round(tot[0] / len(res), 3),
                  "reader_top5": round(tot[1] / len(res), 3),
                  "knn_top1": round(tot[2] / len(res), 3),
                  "base_top1": round(tot[3] / len(res), 3)}
    json.dump(out, open(f"{OUTD}/lichess_summary.json", "w"),
              indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print("LICHESS-EVAL-COMPLETE", flush=True)


if __name__ == "__main__":
    main()

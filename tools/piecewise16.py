"""PIECEWISE-16 (operator, totally new direction 2026-09-26):

16 flies — one per piece slot (8 pawns by file + QR,QN,QB,Q,K,KB,KN,
KR). Each fly scores ONLY its own piece's moves ("how strongly I want
to move and to where"). The ensemble's move score = the owning fly's
score; joint argmax = the ensemble move. Trained jointly with
approved-set CE on the initiative set (GAMBIT pool), starting with 10
games (train 8 / holdout 2).

Outputs: piecewise16/{records.json, signals_sample.json}
Per trained position, each piece's want-signal (top destination +
strength) is dumped — the operator's per-piece signal dashboard.
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
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

OUT = "/home/spec/chess-lab/piecewise16"
GAMES = int(os.environ.get("PW_GAMES", "10"))
CAP = int(os.environ.get("CAP", "600"))
LR = float(os.environ.get("LR", "3e-4"))
os.makedirs(OUT, exist_ok=True)


def slot_of(mv, board):
    """16 slots: pawns 0-7 by from-file; officers 8-15 by type+side."""
    pc = board.piece_at(mv.from_square)
    if pc is None:
        return None
    f = chess.square_file(mv.from_square)
    if pc.piece_type == chess.PAWN:
        return f
    side = 0 if f <= 3 else 1
    t = pc.piece_type
    if t == chess.QUEEN:
        return 11
    if t == chess.KING:
        return 12
    return {chess.ROOK: (8, 15), chess.KNIGHT: (9, 14),
            chess.BISHOP: (10, 13)}[t][side]


def main():
    flyfeat_cb.feat_vec(chess.Board())
    rows = fc.load_pools(["GAMBIT"])
    by_game = {}
    for r in rows:
        by_game.setdefault(r.get("gid"), []).append(r)
    gids = sorted(by_game, key=lambda g: -len(by_game[g]))[:GAMES]
    train_rows = []
    hold_rows = []
    for gi, g in enumerate(gids):
        rr = sorted(by_game[g], key=lambda r: r.get("ply", 0))
        if gi < GAMES - 2:
            train_rows += rr
        else:
            hold_rows += rr
    print(json.dumps({"games": len(gids), "train": len(train_rows),
                      "hold": len(hold_rows)}), flush=True)

    flies = [tp.build_model(0) for _ in range(16)]
    WT = flies[0].WT
    for f in flies:
        f.WT = WT
    params = [p for f in flies for p in f.parameters()]
    opt = torch.optim.Adam(params, lr=LR)
    rng = random.Random(1)

    def pack(rows_sl):
        B = len(rows_sl)
        Ls = [int(r["_pre"][0]["fen_off"][r["_pre"][1] + 1]
                   - r["_pre"][0]["fen_off"][r["_pre"][1]])
              for r in rows_sl]
        M = max(Ls)
        F = rows_sl[0]["_pre"][0]["mf"].shape[1]
        slotb = np.zeros((B, M), np.int64)
        pcrowb = np.zeros((B, M), np.int64)
        mfb = np.zeros((B, M, F), np.float32)
        maskb = np.zeros((B, M), bool)
        psb = np.zeros((B, M), np.float32)
        thb = np.zeros((B, M), np.float32)
        ps2b = np.zeros((B, M), np.float32)
        xs = np.stack([flyfeat_cb.feat_vec_by_fen(r["fen"])
                       for r in rows_sl])
        mus = []
        for b, r in enumerate(rows_sl):
            pres, ri = r["_pre"]
            s, e = int(pres["fen_off"][ri]), \
                int(pres["fen_off"][ri + 1])
            L = e - s
            slotb[b, :L] = pres["slot"][s:e]
            pcrowb[b, :L] = pres["pcrow"][s:e]
            mfb[b, :L] = pres["mf"][s:e]
            maskb[b, :L] = True
            psb[b, :L] = pres["pseudo"][s:e]
            thb[b, :L] = pres["threat"][s:e]
            ps2b[b, :L] = pres["pseudo2"][s:e]
            bd = chess.Board(r["fen"])
            mus.append([m.uci() for m in bd.legal_moves])
        return (xs, slotb, pcrowb, mfb, maskb, psb, thb, ps2b, Ls,
                mus)

    def build_slotmap(rows_sl, mus_sl):
        sm = []
        for r, mus_r in zip(rows_sl, mus_sl):
            bd = chess.Board(r["fen"])
            sm.append([slot_of(chess.Move.from_uci(u), bd)
                       for u in mus_r])
        return sm

    def run_batch(rows_sl):
        (xs, slotb, pcrowb, mfb, maskb, psb, thb, ps2b, Ls,
         mus) = pack(rows_sl)
        sm = build_slotmap(rows_sl, mus)
        B = len(rows_sl)
        M = slotb.shape[1]
        allT = []
        for f in flies:
            T, _, _, _ = fc.forward(f, xs, slotb, pcrowb, mfb,
                                    maskb, psb, thb, ps2b)
            allT.append(T)
        J = torch.zeros((B, M), device=fc.DEV)
        for b in range(B):
            for j in range(Ls[b]):
                sl = sm[b][j]
                if sl is not None:
                    J[b, j] = allT[sl][b, j]
                else:
                    J[b, j] = -1e9
        # approved sets
        app = torch.zeros((B, M), dtype=torch.bool, device=fc.DEV)
        for b, r in enumerate(rows_sl):
            ch = r.get("children", {})
            for u, c in ch.items():
                if u in mus[b]:
                    app[b, mus[b].index(u)] = \
                        torch.tensor(c.get("cat") != "loss")
        valid = torch.zeros((B, M), dtype=torch.bool,
                            device=fc.DEV)
        for b in range(B):
            valid[b, :Ls[b]] = True
        return J, app, valid, Ls, mus, allT, sm

    t0 = time.time()
    step = 0
    while step < CAP:
        batch = rng.sample(train_rows, min(8, len(train_rows)))
        J, app, valid, Ls, mus, allT, sm = run_batch(batch)
        Jm = J.masked_fill(~valid, -1e9)
        lse_all = torch.logsumexp(Jm, dim=1)
        Ja = Jm.masked_fill(~app, -1e9)
        lse_ap = torch.logsumexp(Ja, dim=1)
        # positions with empty approved set: skip
        has = app.any(1)
        joint = -((lse_ap - lse_all)[has]).mean()
        # AUX per-fly loss (dense credit): each fly ranks approved
        # moves among ITS OWN moves — full-strength signal per fly
        aux = []
        for b in range(J.shape[0]):
            if not has[b]:
                continue
            own = {}
            for j in range(Ls[b]):
                sl = sm[b][j] if b < len(sm) else None
                if sl is not None:
                    own.setdefault(sl, []).append(j)
            for sl, js in own.items():
                ap_js = [j for j in js if bool(app[b, j])]
                if not ap_js:
                    continue
                allv = Jm[b, js]
                apv = Jm[b, ap_js]
                aux.append(torch.logsumexp(allv, 0)
                           - torch.logsumexp(apv, 0))
        loss = joint + 0.5 * (torch.stack(aux).mean()
                              if aux else
                              torch.zeros((), device=fc.DEV))
        opt.zero_grad()
        loss.backward()
        for f in flies:      # per-fly clip: global clip starved each
            torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)  # fly of ~1/16 lr
        opt.step()
        step += 1
        if step % 50 == 0:
            tr_ok = tr_n = 0
            with torch.no_grad():
                Jv = Jm.argmax(1).tolist()
                for b in range(len(batch)):
                    ok = bool(app[b, Jv[b]])
                    tr_ok += ok
                    tr_n += 1
            print(json.dumps({"step": step,
                              "loss": round(float(loss), 3),
                              "batch_top1": round(tr_ok / tr_n, 3),
                              "elapsed_s": round(time.time() - t0)}),
                  flush=True)
        if step % 250 == 0:
            with torch.no_grad():
                ho_ok = ho_n = 0
                for s2 in range(0, len(hold_rows), 8):
                    bb = hold_rows[s2:s2 + 8]
                    J2, app2, valid2, Ls2, mus2, _, _ = \
                        run_batch(bb)
                    Jm2 = J2.masked_fill(~valid2, -1e9)
                    picks = Jm2.argmax(1).tolist()
                    for b in range(len(bb)):
                        ho_ok += bool(app2[b, picks[b]])
                        ho_n += 1
                print(json.dumps({"step": step,
                                  "hold_top1": round(
                                      ho_ok / max(ho_n, 1), 3),
                                  "hold_n": ho_n}), flush=True)

    # final: train + holdout accuracy, and signal dashboard sample
    def eval_rows(rows_sl):
        ok = n = 0
        sigs = []
        with torch.no_grad():
            for s in range(0, len(rows_sl), 8):
                bb = rows_sl[s:s + 8]
                J, app, valid, Ls, mus, allT, sm = run_batch(bb)
                Jm = J.masked_fill(~valid, -1e9)
                picks = Jm.argmax(1).tolist()
                for b in range(len(bb)):
                    ok += bool(app[b, picks[b]])
                    n += 1
        return round(ok / max(n, 1), 3), n

    tr_acc, tr_n = eval_rows(train_rows)
    ho_acc, ho_n = eval_rows(hold_rows)
    # signals: for one held-out position, every piece's want
    r0 = hold_rows[0]
    (xs, slotb, pcrowb, mfb, maskb, psb, thb, ps2b, Ls, mus) = \
        pack([r0])
    sm = build_slotmap([r0], mus)
    sigs = []
    with torch.no_grad():
        T16 = []
        for f in flies:
            T, _, _, _ = fc.forward(f, xs, slotb, pcrowb, mfb,
                                    maskb, psb, thb, ps2b)
            T16.append(T)
        bd = chess.Board(r0["fen"])
        for j, u in enumerate(mus[0]):
            sl = sm[0][j]
            if sl is not None:
                sigs.append({"move": u, "fly": sl,
                             "want": round(float(T16[sl][0, j]), 3)})
        sigs.sort(key=lambda s: -s["want"])
    out = {"games": len(gids), "train_acc": tr_acc,
           "train_n": tr_n, "hold_acc": ho_acc, "hold_n": ho_n,
           "sample_fen": r0["fen"], "sample_best": r0.get("best"),
           "top_wants": sigs[:10]}
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(f"{OUT}/records.json", "w"), indent=1)
    print("PIECEWISE16-COMPLETE", flush=True)


if __name__ == "__main__":
    main()

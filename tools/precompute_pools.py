"""Precompute every training-invariant per-row artifact for the pools.

Pure functions of (fen, children, cat): canon board features, legal-move
lists, move features, slot/piece indices, pseudo/pseudo2/threat scalars,
and the optimal-move mask (the gate's answer key). Emits ONE npz per pool
alongside the jsonl; the trainer/gate consume these and skip ALL python
board work at sweep/batch time. Only propagate + readout stay live
(retino_gain feeds the propagation input and it trains).

Schema (per pool, ragged rows concatenated):
  fen_off   [R+1]  row offsets into the packed arrays
  slot      [T]    from*64+to per move
  pcrow     [T]    piece-type index per move
  mf        [T, F] move features
  pseudo    [T]    attack-flag
  pseudo2   [T]    pseudo-legal flag
  threat    [T]    post-move threat level
  opt       [T]    1.0 if the move is in the optimal set (gate key)
  best_idx  [R]    index of the row's best move (for CE)
  fens      [R]    fen strings (object) for activation caching keys
"""
import json
import glob
import os
import sys
import numpy as np
import chess

sys.path.insert(0, "/home/spec/chess-lab")

POOL_DIR = "/home/spec/chess-lab/tbpools"
FLIP = {"win": "loss", "loss": "win", "draw": "draw",
        "cursed_win": "cursed_loss", "cursed_loss": "cursed_win"}
_PC_IDX = {chess.KING: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3,
           chess.PAWN: 4, chess.QUEEN: 5}

import flyfeat_cb
flyfeat_cb.feat_vec(chess.Board())      # initialize FEATURE_KEYS
import fly_curriculum as fc


def precompute(path):
    rows = [json.loads(l) for l in open(path)]
    R = len(rows)
    fen_off = [0]
    slot, pcrow, mfl, pseudo, pseudo2, threat, opt = [], [], [], [], [], [], []
    best_idx = np.zeros(R, dtype=np.int64)
    fens = []
    for r, e in enumerate(rows):
        fen = e["fen"]
        fens.append(fen)
        try:
            b = chess.Board(fen)
        except Exception:
            fen_off.append(fen_off[-1])
            best_idx[r] = -1
            continue
        mvs = list(b.legal_moves)
        p2 = {(pm.from_square, pm.to_square) for pm in b.pseudo_legal_moves}
        ch = e.get("children", {})
        # approved set (operator doctrine 2026-09-20): only a move that
        # actually LOSES is disapproved; keeping the game alive —
        # practical wins, cursed holds, draws — is approved. In
        # already-lost positions nothing is disapproved. Children cats
        # are parent-perspective.
        if e["cat"] == "loss":
            optset = set(ch)
        else:
            optset = {u for u, c in ch.items() if c.get("cat") != "loss"}
        bi = -1
        for j, mv in enumerate(mvs):
            u = mv.uci()
            slot.append(mv.from_square * 64 + mv.to_square)
            pc = b.piece_at(mv.from_square)
            pcrow.append(_PC_IDX[pc.piece_type] if pc else 0)
            mfl.append(flyfeat_cb.move_feats(b, mv))
            pseudo.append(1.0 if (b.attacks_mask(mv.from_square)
                                  & chess.BB_SQUARES[mv.to_square]) else 0.0)
            pseudo2.append(1.0 if (mv.from_square, mv.to_square) in p2
                           else 0.0)
            b.push(mv)
            threat.append(min(bin(b.attacks_mask(mv.to_square)
                                  & b.occupied_co[b.turn]).count("1"),
                              4) / 4.0)
            b.pop()
            opt.append(1.0 if u in optset else 0.0)
            if u == e.get("best"):
                bi = j
        best_idx[r] = bi if bi >= 0 else (0 if mvs else -1)
        fen_off.append(fen_off[-1] + len(mvs))
    T = fen_off[-1]
    F = flyfeat_cb.MOVE_DIMS
    out = dict(
        fen_off=np.array(fen_off, dtype=np.int64),
        slot=np.array(slot, dtype=np.int64),
        pcrow=np.array(pcrow, dtype=np.int64),
        mf=(np.array(mfl, dtype=np.float32).reshape(T, F)
            if T else np.zeros((0, F), np.float32)),
        pseudo=np.array(pseudo, dtype=np.float32),
        pseudo2=np.array(pseudo2, dtype=np.float32),
        threat=np.array(threat, dtype=np.float32),
        opt=np.array(opt, dtype=np.float32),
        best_idx=best_idx,
        fens=np.array(fens, dtype=object),
    )
    np.savez_compressed(path.replace(".jsonl", ".pre.npz"), **out)
    return R, T


def main():
    for path in sorted(glob.glob(f"{POOL_DIR}/*.jsonl")):
        R, T = precompute(path)
        print(f"PRE {os.path.basename(path)}: rows={R} moves={T}",
              flush=True)
    print("PRECOMPUTE-COMPLETE", flush=True)


if __name__ == "__main__":
    main()

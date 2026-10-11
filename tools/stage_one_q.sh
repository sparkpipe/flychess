#!/bin/bash
# Stage the ONE_Q pool + precompute (no kill patterns anywhere).
set -e
cd /home/spec/chess-lab
python3 - <<'PYEOF'
import json
e = json.loads(open("/tmp/one_question.jsonl").read())
with open("tbpools/ONE_Q.jsonl", "w") as f:
    f.write(json.dumps(dict(e, pool="ONE_Q")) + "\n")
print("pool written")
PYEOF
python3 - <<'PYEOF'
import json, numpy as np, chess, sys
sys.path.insert(0, "/home/spec/chess-lab")
import flyfeat_cb
import fly_curriculum as fc
flyfeat_cb.feat_vec(chess.Board())
_PC = fc._PC_IDX
FLIP = {"win": "loss", "loss": "win", "draw": "draw",
        "cursed_win": "cursed_loss", "cursed_loss": "cursed_win"}
rows = [json.loads(l) for l in open("tbpools/ONE_Q.jsonl")]
fen_off = [0]; slot = []; pcrow = []; mfl = []; ps = []; ps2 = []
thr = []; opt = []
best = np.zeros(len(rows), dtype=np.int64); fens = []
for r_, e in enumerate(rows):
    b = chess.Board(e["fen"]); mvs = list(b.legal_moves)
    p2 = {(pm.from_square, pm.to_square) for pm in b.pseudo_legal_moves}
    ch = e.get("children", {})
    optcat = {"win": "loss", "cursed_win": "loss", "draw": "draw",
              "cursed_loss": "win", "loss": "win"}[e["cat"]]
    optset = {u for u, c in ch.items() if FLIP.get(c.get("cat")) == optcat}
    bi = 0
    for j, mv in enumerate(mvs):
        u = mv.uci()
        slot.append(mv.from_square * 64 + mv.to_square)
        pc = b.piece_at(mv.from_square)
        pcrow.append(_PC[pc.piece_type] if pc else 0)
        mfl.append(flyfeat_cb.move_feats(b, mv))
        ps.append(1.0 if (b.attacks_mask(mv.from_square)
                          & chess.BB_SQUARES[mv.to_square]) else 0.0)
        ps2.append(1.0 if (mv.from_square, mv.to_square) in p2 else 0.0)
        b.push(mv)
        thr.append(min(bin(b.attacks_mask(mv.to_square)
                          & b.occupied_co[b.turn]).count("1"), 4) / 4.0)
        b.pop()
        opt.append(1.0 if u in optset else 0.0)
        if u == e.get("best"):
            bi = j
    best[r_] = bi; fens.append(e["fen"])
    fen_off.append(fen_off[-1] + len(mvs))
np.savez_compressed("tbpools/ONE_Q.pre.npz",
                    fen_off=np.array(fen_off, np.int64),
                    slot=np.array(slot, np.int64),
                    pcrow=np.array(pcrow, np.int64),
                    mf=np.array(mfl, np.float32),
                    pseudo=np.array(ps, np.float32),
                    pseudo2=np.array(ps2, np.float32),
                    threat=np.array(thr, np.float32),
                    opt=np.array(opt, np.float32),
                    best_idx=best,
                    fens=np.array(fens, dtype=object))
print("pre written; optset members:", int(np.array(opt).sum()))
PYEOF
echo STAGED

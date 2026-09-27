"""SEARCH PoC — the anti-cheating answer (operator: "training on the
new position is cheating"): solve the gap position with ZERO training.

Beam search: fly/base-scored move ordering (top-K), opponent replies
top-2, depth cap; leaves: syzygy probe when <=5 men (exact), else the
base rich scorer's band. No parameter is updated anywhere.

Position: the failpass gap (pos 742) where flyA/flyB/base/stretched
all scored 0.0. Success: search's root move lands in the approved set.
"""
import sys
import os
import json
import chess
import chess.syzygy

sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
os.environ.setdefault("ANORM", "1")
import numpy as np
import torch
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

SYZ = "/home/spec/syzygy"
FEN = json.load(open("/home/spec/chess-lab/singles/stretch/"
                     "failpass.json"))["fen"]


def board_scores(model, b, k=None):
    """rich-scorer move scores for a board, sorted desc."""
    mvs = list(b.legal_moves)
    if not mvs:
        return []
    from mof_lichess_eval import pack_board  # reuse
    (x, slot, pcrow, mfm, mask, psb, thb, ps2b), _ = pack_board(b)
    T, _, _, _ = fc.forward(model, x, slot, pcrow, mfm, mask, psb,
                            thb, ps2b)
    tv = T[0].detach().cpu().numpy()
    order = sorted(range(len(mvs)), key=lambda j: -tv[j])
    out = [(mvs[j], float(tv[j])) for j in order]
    return out[:k] if k else out


def main():
    flyfeat_cb.feat_vec(chess.Board())
    tb = chess.syzygy.open_tablebase(SYZ) if os.path.isdir(SYZ) \
        else None
    model = tp.build_model(0)          # base rich scorer as prior

    def probe(b):
        if tb is None or len(b.piece_map()) > 5:
            return None
        try:
            wdl = tb.probe_wdl(b)      # from side-to-move perspective
            dtz = tb.probe_dtz(b)
            return wdl, dtz
        except Exception:
            return None

    def search(b, depth, alpha=-2.0, beta=2.0):
        """policy-ordered negamax; value from side-to-move perspective."""
        if b.is_checkmate():
            return -1.0, None
        if b.is_stalemate() or b.is_insufficient_material() \
                or b.can_claim_draw():
            return 0.0, None
        pr = probe(b)
        if pr is not None:
            return float(pr[0]) / 2.0, None
        if depth == 0:
            sc = board_scores(model, b, k=1)
            return (max(-1.0, min(1.0, sc[0][1] / 6.0))
                    if sc else 0.0), None
        scored = board_scores(model, b, k=3)
        best_v, best_m = -3.0, None
        for mv, _s in scored:
            b.push(mv)
            v, _ = search(b, depth - 1, -beta, -alpha)
            b.pop()
            v = -v
            if v > best_v:
                best_v, best_m = v, mv
            if best_v > alpha:
                alpha = best_v
            if alpha >= beta:
                break
        return best_v, best_m

    b = chess.Board(FEN)
    DEPTH = int(os.environ.get("DEPTH", "8"))
    v, mv = search(b, DEPTH)
    pick = mv.uci() if mv else None
    # approved set from the corpus row
    approved = None
    for pl in ("DEGM2_Ch1", "DEGM2_Ch2", "DEGM2_Ch3", "DEGM2_Ch4",
               "DEGM2_Ch5", "DEGM2_Ch6", "DEGM2_Ch7", "DEGM2_Ch8",
               "DEGM2_Ch9", "DEGM2_Ch10", "DEGM2_Ch11", "DEGM2_Ch12",
               "DEGM2_Ch13", "DEGM2_Ch14", "DEGM2_Ch15"):
        for r in fc.load_pools([pl]):
            if r["fen"] == FEN:
                approved = {u for u, c in r["children"].items()
                            if c.get("cat") != "loss"}
                break
        if approved:
            break
    out = {"fen": FEN, "search_value": round(v, 3),
           "search_pick": pick, "approved": sorted(approved or []),
           "pass": pick in (approved or set())}
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open("/home/spec/chess-lab/singles/stretch/"
                        "search_poc.json", "w"), indent=1)
    print("SEARCH-POC-COMPLETE", flush=True)


if __name__ == "__main__":
    main()

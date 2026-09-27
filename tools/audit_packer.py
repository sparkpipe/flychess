"""AUDIT: sample random credited pieces; verify with INDEPENDENT methods.

For each sampled piece:
  1. Fresh SF depth-12 eval at EVERY ply (x86 build, not the stored ARM evals):
     verify the wp trajectory reproduces the credited category (walls 30/45/55/70),
     within a small cross-arch tolerance at band edges.
  2. Stored cp vs fresh cp (data-integrity of the join).
  3. Expert membership recomputed from the FEN with python-chess
     (independent config/residue/men/lock implementation).
  4. Bit-level round-trip: unpack(pack(fen)) reproduces the position.

Usage: audit_packer.py <segments.jsonl> [n_per_category]
"""
import sys
import os
import json
import math
import random
import struct
import subprocess
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~") + "/extnvme/phase-moe")
import chess
import chess.engine

from pack_expert_bins import (GROUPS, wp_group, CLIMB, assign, pack_sfen,
                              pack_move, HUFF)

SEG = sys.argv[1]
N_PER = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SF = os.path.expanduser("/home/spec/Stockfish/src/stockfish")

random.seed(20260927)

# ---------- bit-level unpacker (independent of BitWriter) ----------
class BitReader:
    def __init__(self, data):
        self.data = data
        self.bit = 0

    def one(self):
        b = (self.data[self.bit // 8] >> (self.bit & 7)) & 1
        self.bit += 1
        return b

    def n(self, n):
        v = 0
        for i in range(n):
            v |= self.one() << i
        return v


DEHUFF = {0b001: chess.PAWN, 0b011: chess.KNIGHT, 0b101: chess.BISHOP,
          0b111: chess.ROOK, 0b1001: chess.QUEEN}


def unpack_sfen(blob):
    r = BitReader(blob)
    turn = chess.BLACK if r.one() else chess.WHITE
    wk, bk = r.n(6), r.n(6)
    board = chess.Board(None)
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    for rank in range(7, -1, -1):
        for file in range(8):
            sq = rank * 8 + file
            if sq in (wk, bk):
                continue
            if not r.one():
                continue
            code = 1 | (r.one() << 1) | (r.one() << 2) | (r.one() << 3)
            color = chess.BLACK if r.one() else chess.WHITE
            board.set_piece_at(sq, chess.Piece(DEHUFF[code], color))
    cast = ""
    if r.one(): cast += "K"
    if r.one(): cast += "Q"
    if r.one(): cast += "k"
    if r.one(): cast += "q"
    cast = cast or "-"
    ep = r.n(6) if r.one() else None
    hm = r.n(6)
    fm = r.n(8)
    board.turn = turn
    board.set_castling_fen(cast)
    if ep is not None:
        board.ep_square = ep
    return board, hm, fm


# ---------- independent expert recomputation ----------
def expert_of_fen(fen):
    b = chess.Board(fen)
    men = len(b.piece_map())
    cw = {"Q": 0, "R": 0, "B": 0, "N": 0}
    cb = {"Q": 0, "R": 0, "B": 0, "N": 0}
    wcol = []
    bcol = []
    for s, p in b.piece_map().items():
        u = p.symbol().upper()
        if u == "K":
            continue
        if u == "P":
            continue
        (cw if p.color else cb)[u] += 1
        if u == "B":
            (wcol if p.color else bcol).append(
                (chess.square_file(s) + chess.square_rank(s)) % 2)
    res_w = "".join(sorted(sum(([k] * (cw[k] - min(cw[k], cb[k]))
                                 for k in "QRBN"), [])))
    res_b = "".join(sorted(sum(([k] * (cb[k] - min(cw[k], cb[k]))
                                 for k in "QRBN"), [])))
    kres = tuple(sorted((res_w, res_b)))
    sym = res_w == "" and res_b == ""
    RES = {("B", "N"): "nvb", ("N", "R"): "nvr", ("B", "R"): "bvr",
           ("NN", "R"): "rv2m", ("BN", "R"): "rv2m", ("BB", "R"): "rv2m",
           ("Q", "RR"): "qvmat", ("Q", "RN"): "qvmat", ("Q", "RB"): "qvmat"}
    if men <= 5:
        return "tb-region", men
    if not sym and kres in RES:
        return RES[kres], men
    if sym and wcol and bcol and (
            (sum(wcol) * 2 > len(wcol)) != (sum(bcol) * 2 > len(bcol))):
        return "oppb", men
    if men <= 10:
        return "dvoretsky", men
    # locked center files
    lock = 0
    for f in range(8):
        wr = [chess.square_rank(s) for s in chess.SquareSet(
            b.pieces(chess.PAWN, chess.WHITE)) if chess.square_file(s) == f]
        br = [chess.square_rank(s) for s in chess.SquareSet(
            b.pieces(chess.PAWN, chess.BLACK)) if chess.square_file(s) == f]
        if wr and br and min(br) - max(wr) == 1 and 2 <= f <= 5:
            lock += 1
    return "balanced_l%d" % (lock if lock < 3 else 3), men


def main():
    # phase 1: replicate the machine, reservoir-sample credited pieces
    by_cat = defaultdict(list)   # category -> [(positions...)]
    cur_seg = None
    piece_group = piece_expert = None
    piece_buf = []

    def keep(positions, cat):
        s = by_cat[cat]
        s.append(positions)
        if len(s) > 400:          # reservoir trim, keep it light
            s.pop(random.randrange(len(s)))

    for line in open(SEG):
        r = json.loads(line)
        seg_id = (r["gid"], r["traj"], r["seg_start_wp"], r["seg_end_wp"])
        g = wp_group(r["wp"])
        if seg_id != cur_seg or g != piece_group:
            if seg_id == cur_seg and CLIMB.get((piece_group, g)):
                keep(piece_buf, CLIMB[(piece_group, g)])
            elif seg_id != cur_seg and piece_buf and (
                    (piece_group == "convert" and piece_buf[-1]["wp"] > piece_buf[0]["wp"])
                    or (piece_group == "deep" and piece_buf[-1]["wp"] < piece_buf[0]["wp"])):
                keep(piece_buf, "70towin")
            cur_seg = seg_id
            piece_group = g
            piece_buf = []
        piece_buf.append(r)

    print("sampled pools:", {c: len(v) for c, v in by_cat.items()})

    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 256})
    os.environ["PHASE_MOE"] = "0"

    fails = defaultdict(int)
    checked = 0
    for cat, pool in sorted(by_cat.items()):
        for piece in random.sample(pool, min(N_PER, len(pool))):
            # 1. fresh evals every ply + stored-cp integrity
            fresh_wps = []
            for p in piece:
                b = chess.Board(p["fen"])
                info = eng.analyse(b, chess.engine.Limit(depth=12))
                cp = info["score"].pov(b.turn).score()
                if abs(cp - p["cp"]) > 30:
                    fails["cp_mismatch>30"] += 1
                    print("CP MISMATCH", p["fen"], "stored", p["cp"], "fresh", cp)
                w = 1.0 / (1.0 + math.exp(-cp / 361.0))
                fresh_wps.append(1 - w if p["winner"] == "b" else w)
            # category check: fresh trajectory crosses the claimed wall(s)
            g_first, g_last = wp_group(fresh_wps[0]), wp_group(fresh_wps[-1])
            ok = True
            if cat == "70towin":
                ok = (g_first == "convert" and fresh_wps[-1] > fresh_wps[0]) or \
                     (g_first == "deep" and fresh_wps[-1] < fresh_wps[0])
            else:
                rises = {"30to45": ("defend", "equalize"),
                         "45to60": ("equalize", "press"),
                         "55to70": ("press", "convert")}
                a, c = rises[cat]
                ok = (g_first == a and wp_group(fresh_wps[-1]) >= c) or \
                     (g_first == c and wp_group(fresh_wps[-1]) <= a)  # black mirror
            if not ok:
                fails["category_mismatch"] += 1
                print("CAT MISMATCH", cat, "fresh groups",
                      g_first, "->", g_last,
                      "wps", [round(w, 3) for w in fresh_wps[:6]])
            # 2. expert recomputation + 3. round-trip on the first position
            p0 = piece[0]
            ex, men = expert_of_fen(p0["fen"])
            packed = pack_sfen(chess.Board(p0["fen"]))
            ub, hm, fm = unpack_sfen(packed)
            orig = chess.Board(p0["fen"])
            same_pos = (ub.piece_map() == orig.piece_map()
                        and ub.turn == orig.turn
                        and ub.castling_rights == orig.castling_rights
                        and ub.ep_square == orig.ep_square)
            if not same_pos or hm != orig.halfmove_clock \
                    or fm != (orig.fullmove_number & 0xFF):
                fails["roundtrip_position"] += 1
                print("ROUNDTRIP FAIL", p0["fen"], "->", ub.fen(),
                      (hm, fm))
            expected0 = assign(p0)
            if ex != expected0:
                fails["expert_recompute_mismatch"] += 1
                print("EXPERT MISMATCH", p0["fen"], ex, expected0)
            checked += 1
    eng.quit()

    print("\nAUDIT: %d pieces checked" % checked)
    if fails:
        print("FAILURES:", dict(fails))
        print("AUDIT-FAIL")
        sys.exit(1)
    print("ALL CHECKS PASS: fresh-eval categories, stored cp, expert "
          "recomputation, bit-level round-trip")
    print("AUDIT-PASS")


if __name__ == "__main__":
    main()

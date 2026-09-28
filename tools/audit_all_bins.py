"""FULL AUDIT: 100% of the completed training bins.

Per bin (*.bin in expert_bins_both, including aug_* and tactics):
  - record count, side-to-move split (black vs white)
  - score distribution (bands, per-side winning counts, mean |cp|)
  - fullmove buckets, total-men buckets
  - move stats: captures, promotions by type, king two-file moves (castling-ish)
  - CLASS VERIFY: every position independently re-classified (python-chess);
    record's bin must equal its class (tactics exempt)
  - exact-duplicate records within bin
Output: audit_report.txt (human) + audit_report.json
"""
import sys
import os
import json
import struct
import glob
import hashlib
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
from audit_packer import unpack_sfen, DEHUFF

BINDIR = "/mnt/cold-raid6/chess-audit/expert_bins_both"
OUT = "/mnt/cold-raid6/chess-audit/audit_report"

CP_BANDS = [50, 150, 300, 600, 1500, 30000]
PLY_BANDS = [10, 20, 30, 50, 100, 1000]
MEN_BANDS = [5, 10, 20, 32]


def band(v, edges):
    for i, e in enumerate(edges):
        if v <= e:
            return i
    return len(edges)


def classify_fen(board):
    """Independent expert classification (audit_packer.expert_of_fen lineage)."""
    men = len(board.piece_map())
    if men <= 5:
        return "tb-region"
    cw = {"Q": 0, "R": 0, "B": 0, "N": 0}
    cb = {"Q": 0, "R": 0, "B": 0, "N": 0}
    wcol, bcol = [], []
    for s, p in board.piece_map().items():
        u = p.symbol().upper()
        if u == "K" or u == "P":
            continue
        (cw if p.color else cb)[u] += 1
        if u == "B":
            (wcol if p.color else bcol).append(
                (chess.square_file(s) + chess.square_rank(s)) % 2)
    resw = "".join(sorted(sum(([k] * max(0, cw[k] - cb[k]) for k in "QRBN"), [])))
    resb = "".join(sorted(sum(([k] * max(0, cb[k] - cw[k]) for k in "QRBN"), [])))
    kres = tuple(sorted((resw, resb)))
    sym = resw == "" and resb == ""
    RES = {("B", "N"): "nvb", ("N", "R"): "nvr", ("B", "R"): "bvr",
           ("NN", "R"): "rv2m", ("BN", "R"): "rv2m", ("BB", "R"): "rv2m",
           ("Q", "RR"): "qvmat", ("Q", "RN"): "qvmat", ("Q", "RB"): "qvmat"}
    if not sym and kres in RES:
        return RES[kres]
    if sym and wcol and bcol and \
            (sum(wcol) * 2 > len(wcol)) != (sum(bcol) * 2 > len(bcol)):
        return "oppb"
    if men <= 10:
        return "dvoretsky"
    lock = 0
    for f in range(8):
        wr = [chess.square_rank(s) for s in chess.SquareSet(
            board.pieces(chess.PAWN, chess.WHITE)) if chess.square_file(s) == f]
        br = [chess.square_rank(s) for s in chess.SquareSet(
            board.pieces(chess.PAWN, chess.BLACK)) if chess.square_file(s) == f]
        if wr and br and min(br) - max(wr) == 1 and 2 <= f <= 5:
            lock += 1
    return "balanced_l%d" % (lock if lock < 3 else 3)


def audit_file(path):
    name = os.path.basename(path)[:-4]
    st = {
        "count": 0, "stm_black": 0,
        "cp_band": defaultdict(int), "white_winning": 0, "black_winning": 0,
        "abs_cp_sum": 0,
        "ply_band": defaultdict(int), "men_band": defaultdict(int),
        "captures": 0, "promo": defaultdict(int), "king_2file": 0,
        "class_mismatch": defaultdict(int), "dups": 0,
        # piece-level chess effects (operator: "overall chess effects")
        "piece_moves": defaultdict(int), "piece_captures": defaultdict(int),
        "piece_captured": defaultdict(int), "ep": 0,
        "castle_OO": 0, "castle_OOO": 0, "promo_captures": 0,
        "checks": 0,
    }
    seen = set()
    with open(path, "rb") as f:
        while True:
            rec = f.read(40)
            if len(rec) < 40:
                break
            pos, cp, mvraw, ply, _r, _p = struct.unpack("<32shHHbB", rec)
            board, hm, fm = unpack_sfen(pos[:32])
            st["count"] += 1
            black = board.turn == chess.BLACK
            st["stm_black"] += black
            # cp is stm-perspective: white-perspective sign for reporting
            wcp = -cp if black else cp
            st["cp_band"][band(abs(cp), CP_BANDS)] += 1
            st["abs_cp_sum"] += abs(cp)
            if wcp >= 150:
                st["white_winning"] += 1
            elif wcp <= -150:
                st["black_winning"] += 1
            st["ply_band"][band(fm, PLY_BANDS)] += 1
            st["men_band"][band(len(board.piece_map()), MEN_BANDS)] += 1
            to = mvraw >> 4 & 63
            fr = mvraw >> 10
            promo = mvraw & 0xF
            pc = board.piece_at(fr)
            victim = board.piece_at(to)
            PN = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
                  chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
            if pc:
                st["piece_moves"][PN[pc.piece_type]] += 1
                if victim:
                    st["piece_captures"][PN[pc.piece_type]] += 1
                    st["piece_captured"][PN[victim.piece_type]] += 1
            if victim is not None:
                st["captures"] += 1
            if promo:
                st["promo"][" NBRQ"[promo] if promo < 5 else "?"] += 1
                if victim:
                    st["promo_captures"] += 1
            if pc and pc.piece_type == chess.PAWN and victim is None \
                    and to % 8 != fr % 8:
                st["ep"] += 1     # pawn diagonal to empty square = en passant
            if pc and pc.piece_type == chess.KING and abs(to % 8 - fr % 8) == 2:
                st["king_2file"] += 1
                if to % 8 == 6:
                    st["castle_OO"] += 1
                elif to % 8 == 2:
                    st["castle_OOO"] += 1
            try:
                mv = chess.Move(fr, to, promotion=promo or None)
                if board.gives_check(mv):
                    st["checks"] += 1
            except Exception:
                pass
            if name != "tactics":
                cls = classify_fen(board)
                want = name[4:] if name.startswith("aug_") else name
                if cls != want:
                    st["class_mismatch"]["%s!=%s" % (cls, want)] += 1
            h = hashlib.blake2b(rec, digest_size=12).digest()
            if h in seen:
                st["dups"] += 1
            else:
                seen.add(h)
    return name, st


def main():
    files = sorted(glob.glob(BINDIR + "/*.bin"))
    results = []
    if len(files) > 1:
        with Pool(12) as p:
            results = p.map(audit_file, files)
    else:
        results = [audit_file(files[0])]

    with open(OUT + ".txt", "w") as t, open(OUT + ".json", "w") as j:
        total = 0
        jdata = {}
        t.write("FULL BIN AUDIT — 100%% of records, %d bins\n\n" % len(files))
        t.write("%-16s %11s %6s %7s %7s %7s %7s %7s %6s %6s\n" % (
            "bin", "records", "%blk", "wWin", "bWin", "caps", "promo", "dup",
            "clsX", "men10"))
        for name, st in sorted(results, key=lambda kv: -kv[1]["count"]):
            total += st["count"]
            n = st["count"]
            t.write("%-16s %11s %5.1f%% %7s %7s %7s %7s %7s %6s %6s\n" % (
                name, format(n, ","), 100.0 * st["stm_black"] / max(n, 1),
                format(st["white_winning"], ","), format(st["black_winning"], ","),
                format(st["captures"], ","), sum(st["promo"].values()),
                st["dups"], sum(st["class_mismatch"].values()),
                st["men_band"][0] + st["men_band"][1]))
            jdata[name] = {k: (dict(v) if isinstance(v, defaultdict) else v)
                           for k, v in st.items()}
        t.write("\nTOTAL: %s records across %d bins\n" % (format(total, ","), len(files)))
        for name, st in sorted(results, key=lambda kv: -kv[1]["count"]):
            if st["count"]:
                t.write("\n%s: mean|cp|=%d stm_black=%.1f%% ply=%s men=%s promo=%s cls_mismatch=%s\n"
                        % (name, st["abs_cp_sum"] / st["count"],
                           100.0 * st["stm_black"] / st["count"],
                           dict(sorted(st["ply_band"].items())),
                           dict(sorted(st["men_band"].items())),
                           dict(st["promo"]), dict(st["class_mismatch"])))
                t.write("  moves: %s\n" % dict(sorted(st["piece_moves"].items())))
                t.write("  captures: %s  captured: %s\n"
                        % (dict(sorted(st["piece_captures"].items())),
                           dict(sorted(st["piece_captured"].items()))))
                t.write("  ep=%d O-O=%d O-O-O=%d promo=%s promo_caps=%d checks=%d\n"
                        % (st["ep"], st["castle_OO"], st["castle_OOO"],
                           dict(st["promo"]), st["promo_captures"], st["checks"]))
        j.dump(jdata, j, indent=1)
    print("AUDIT: %d bins, %d records -> %s.txt/.json" % (len(files), total, OUT))
    print("FULL-AUDIT-COMPLETE")


if __name__ == "__main__":
    main()

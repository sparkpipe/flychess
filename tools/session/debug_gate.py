#!/usr/bin/env python3
"""Pin the gate failure: pick fens straight from nongame, compare
(a) labels-table d12 value, (b) fresh SF17-d12, (c) the score stored in the
corrected bin for that fen (decoded). Any mismatch localizes the bug."""
import glob
import sqlite3
import struct
import subprocess
import sys

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import chess


class Rd:
    def __init__(s, d):
        s.d, s.i = d, 0

    def one(s):
        b = (s.d[s.i // 8] >> (s.i & 7)) & 1
        s.i += 1
        return b

    def n(s, k):
        v = 0
        for j in range(k):
            v |= s.one() << j
        return v


SYM = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
       chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
_sq = lambda sq: chr(ord("a") + (sq % 8)) + str(sq // 8 + 1)


def unpack_sfen(b):
    r = Rd(b)
    stm = "b" if r.one() else "w"
    wk, bk = r.n(6), r.n(6)
    board = [""] * 64
    board[wk] = "K"
    board[bk] = "k"
    for rank in range(7, -1, -1):
        for file in range(8):
            sq = rank * 8 + file
            if sq in (wk, bk):
                continue
            if not r.one():
                continue
            b1, b2, b3 = r.one(), r.one(), r.one()
            val = 1 | (b1 << 1) | (b2 << 2) | (b3 << 3)
            pt = {1: chess.PAWN, 3: chess.KNIGHT, 5: chess.BISHOP,
                  7: chess.ROOK, 9: chess.QUEEN}.get(val)
            board[sq] = SYM[pt].lower() if r.one() else SYM[pt]
    cast = "".join(c for c, f in zip("KQkq", (r.one(), r.one(), r.one(), r.one())) if f) or "-"
    ep = "-"
    if r.one():
        ep = _sq(r.n(6))
    hm = r.n(6)
    fm = r.n(8)
    rows = []
    for rank in range(7, -1, -1):
        e = 0
        row = ""
        for file in range(8):
            pc = board[rank * 8 + file]
            if pc:
                if e:
                    row += str(e)
                    e = 0
                row += pc
            else:
                e += 1
        if e:
            row += str(e)
        rows.append(row)
    return " ".join(["/".join(rows), stm, cast, ep, str(hm), str(fm)])

DB = "/srv/workspace/chess-active/store/games.db"
SF = "/srv/workspace/flychess/src/sf17/src/stockfish"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rows = con.execute(
    "select n.fen, l.cp from nongame n join labels l on l.fen=n.fen"
    " and l.engine='sf17' and l.depth=12 order by random() limit 5").fetchall()

p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\nsetoption name Threads value 1\nisready\n")
p.stdin.flush()
while "readyok" not in p.stdout.readline():
    pass


def d12(fen):
    p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo depth 12\n")
    p.stdin.flush()
    cp = None
    while True:
        ln = p.stdout.readline()
        if not ln:
            return None
        if ln.startswith("info") and not ln.startswith("info string") \
                and " score " in ln:
            t = ln.split()
            for i, x in enumerate(t):
                if i and t[i-1] == "score" and x in ("cp", "mate"):
                    try:
                        v = int(t[i+1])
                    except (ValueError, IndexError):
                        break
                    raw = ((10000 + min(abs(v), 900)) * (1 if v > 0 else -1)
                           if x == "mate" else v)
                    cp = raw * (1 if fen.split()[1] == "w" else -1)
                    break
        elif ln.startswith("bestmove"):
            return cp


# bin scores: map decoded fen-prefix -> score for piece_down
binmap = {}
for tb in glob.glob("/srv/workspace/chess-active/bins-correct16/piece_down.train.bin"):
    raw = open(tb, "rb").read()
    for i in range(0, len(raw) // 40, 400):     # sparse sample, keep it quick
        sfen, score, mv, ply, b, u = struct.unpack_from("<32shHHbB", raw, i * 40)
        fen = unpack_sfen(sfen)
        if fen:
            binmap[tuple(fen.split()[:3])] = (score, tb.split("/")[-1])

for fen, lab in rows:
    fresh = d12(fen)
    bin_entry = binmap.get(tuple(fen.split()[:3]))
    print(f"labels={lab:>6}  fresh={str(fresh):>6}  bin={bin_entry}"
          f"  {fen[:40]}")

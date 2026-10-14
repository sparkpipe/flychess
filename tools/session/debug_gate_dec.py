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


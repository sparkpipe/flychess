import sqlite3, re, sys
from collections import Counter

DB = "file:/mnt/cold-raid6/chess-audit/wp_fit/store/games.db?mode=ro"
SEGIDX = "/mnt/cold-raid6/chess-audit/wp_fit/segments/seg_index.tsv"

db = sqlite3.connect(DB, uri=True)
c = db.cursor()

def segidx_row(sid):
    with open(SEGIDX) as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] == sid:
                side, band, sig, n, lo, hi = p[-6:]
                return dict(seg_id=sid, gid=p[1], side=side, band=band, sig=sig,
                            n=int(n), lo=int(lo), hi=int(hi),
                            meta=p[2:9])
    return None

VALS = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}

def board_of(fen):
    return fen.split(" ")[0]

def matvec(fen):
    b = board_of(fen)
    cnt = Counter(ch.lower() for ch in b if ch.isalpha())
    return (cnt["q"], cnt["r"], cnt["b"], cnt["n"])

def men(fen):
    return sum(1 for ch in board_of(fen) if ch.isalpha())

def show(sid, want_band=None):
    row = segidx_row(sid)
    print("==== seg", sid, row)
    gid = int(row["gid"])
    g = c.execute("SELECT result,welo,belo FROM games WHERE gid=?", (gid,)).fetchone()
    plies = c.execute("SELECT ply,fen,stm,src,cp FROM plies WHERE gid=? ORDER BY ply", (gid,)).fetchall()
    side = row["side"]
    win = "w" if g[0] == "1-0" else "b"
    traj = [(p, f, cp) for (p, f, stm, src, cp) in plies if stm == side and src in ("f", "x")]
    lo, hi = row["lo"], row["hi"]
    win_entries = [(p, f, cp) for (p, f, cp) in traj if lo <= p <= hi]
    print("  game result/elos:", g, "side", side, "winner", win)
    prev_mv = None
    for (p, f, cp) in traj:
        mark = "IN " if lo <= p <= hi else "   "
        mv = matvec(f)
        chg = "" if prev_mv is None or mv == prev_mv else "  <== CHANGE %s->%s" % (prev_mv, mv)
        if lo - 4 <= p <= hi + 2:
            print("  %s ply%3d mat=%s qrbn=%s cp=%5d src=%s%s" % (
                mark, p, sum(1 for ch2 in board_of(f) if ch2.isalpha()), mv, cp,
                dict((pp, s) for (pp, ff, stm, s, cc) in plies if pp == p).get(p), chg))
        prev_mv = mv
    # all trainset lines for this seg
    import glob
    for fp in sorted(glob.glob("/mnt/cold-raid6/chess-audit/wp_fit/trainsets/*/*.final.tsv")):
        out = []
        with open(fp) as fh:
            for line in fh:
                if "|%s|" % sid in line:
                    p = line.rstrip("\n").split("|")
                    out.append((int(p[6]), p[-2], p[-1]))
        if out:
            print("  FILE", fp.split("/")[-2] + "/" + fp.split("/")[-1], sorted(out))

for sid in sys.argv[1:]:
    show(sid)

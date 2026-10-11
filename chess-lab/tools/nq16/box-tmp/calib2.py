import sqlite3, sys, glob
from collections import Counter
db = sqlite3.connect("file:/mnt/cold-raid6/chess-audit/wp_fit/store/games.db?mode=ro", uri=True)
c = db.cursor()
def segidx_row(sid):
    with open("/mnt/cold-raid6/chess-audit/wp_fit/segments/seg_index.tsv") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] == sid:
                side, band, sig, n, lo, hi = p[-6:]
                return dict(seg_id=sid, gid=int(p[1]), side=side, band=band, sig=sig, n=int(n), lo=int(lo), hi=int(hi))
def matvec(fen):
    b = fen.split(" ")[0]
    cnt = Counter(ch.lower() for ch in b if ch.isalpha())
    return (cnt["q"], cnt["r"], cnt["b"], cnt["n"])
def show(sid):
    row = segidx_row(sid); gid=row["gid"]
    g = c.execute("SELECT result FROM games WHERE gid=?", (gid,)).fetchone()
    plies = c.execute("SELECT ply,fen,stm,src,cp FROM plies WHERE gid=? ORDER BY ply", (gid,)).fetchall()
    side = row["side"]; win = "w" if g[0]=="1-0" else "b"
    traj = [(p,f,cp) for (p,f,stm,src,cp) in plies if stm==side and src in ("f","x")]
    lo,hi = row["lo"],row["hi"]
    print("==== seg",sid,row,"winner",win,"side",side)
    prev=None
    for (p,f,cp) in traj:
        mv=matvec(f); chg="" if prev is None or mv==prev else " <==CHG %s->%s"%(prev,mv)
        if lo-6<=p<=hi+2: print("  %s p%3d qrbn=%s cp=%5d %s%s"%("IN" if lo<=p<=hi else "  ",p,mv,cp,src_of(plies,p),chg))
        prev=mv
    for fp in sorted(glob.glob("/mnt/cold-raid6/chess-audit/wp_fit/trainsets/*/*.final.tsv")):
        out=[]
        with open(fp) as fh:
            for line in fh:
                p=line.rstrip("\n").split("|")
                if p[5]==sid: out.append((int(p[6]),p[-2],p[-1]))
        if out: print("  ", "/".join(fp.split("/")[-2:]), sorted(out))
def src_of(plies,p):
    for (pp,f,stm,src,cp) in plies:
        if pp==p: return src
for sid in sys.argv[1:]: show(sid)

"""PROVENANCE — every candidate position's disposition, into the DB.

Re-derives assemble16's decisions deterministically (pure routing + hash
splits + same skip conditions) and writes, for every position from every
source: which training set(s) took it (train16/anti16), which expert, which
split — or why it was not taken (no_move, illegal, no_label, no_segment).

Output: TSV chunks -> loaded into store DB table train_provenance, plus a
positions table for non-game sources so every position exists in the DB.
"""
import sys, os, csv, io, glob, struct, json
import chess
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
sys.path.insert(0, "/tmp")
from routerB import routeB

SEG = "/extnvme/segments"
GAMES = "/mnt/cold-raid6/chess-audit/wp_fit/games_all.tsv"
PGN = "/mnt/cold-raid6/rtx5090-archive/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn"
SHARDS = os.path.expanduser("~/backlog_shards/puzzle_eval")
CSV = "/srv/workspace/flychess/src/chess-lab/puzzles/lichess_db_puzzle.csv"
DEGM = "/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/degm.tsv"
TBTSV = "/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/tb.tsv"
MINI = "/extnvme/active/miniature_rows.tsv"
MINILBL = "/extnvme/active/miniature_labels.tsv"
OUT = "/extnvme/active/provenance.tsv"

# columns: fen, source, src_key, engine_set (train16|anti16|NONE), expert, split, reason
out = open(OUT, "w")
n = 0

def row(fen, source, key, engine, expert, split, reason=""):
    global n
    out.write(f"{fen}\t{source}\t{key}\t{engine}\t{expert or '-'}\t{split or '-'}\t{reason}\n")
    n += 1
    if n % 1000000 == 0:
        print(f"{n:,} rows", flush=True)

# ---- puzzles ----
sol = {}
with open(CSV, encoding="utf-8", errors="replace") as f:
    for r in csv.reader(f):
        if len(r) >= 3:
            sol[r[0]] = r[2].split(" ")
for shard in sorted(glob.glob(f"{SHARDS}/w*.tsv")):
    for line in open(shard, encoding="utf-8", errors="replace"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 5:
            continue
        fen, pid, depth = p[0], p[1], int(p[2])
        moves = sol.get(pid)
        if not moves or depth >= len(moves):
            row(fen, "puzzle", f"{pid}:{depth}", "NONE", None, None, "no_solution_move")
            continue
        try:
            b = chess.Board(fen)
            mv = chess.Move.from_uci(moves[depth])
            ok = mv in b.legal_moves
        except Exception:
            ok = False
        if not ok:
            row(fen, "puzzle", f"{pid}:{depth}", "NONE", None, None, "illegal")
            continue
        e = routeB(fen)
        tag = "val" if (hash(pid) % 20 == 0) else "train"
        row(fen, "puzzle", f"{pid}:{depth}", "train16", e, tag)

# ---- degm + syzygy ----
i = 0
for line in open(DEGM, encoding="utf-8", errors="replace"):
    p = line.rstrip("\n").split("|")
    i += 1
    if len(p) < 4:
        continue
    try:
        b = chess.Board(p[0]); next(iter(b.legal_moves)); ok = True
    except Exception:
        ok = False
    if not ok:
        row(p[0], "degm", str(i), "NONE", None, None, "illegal")
        continue
    e = routeB(p[0])
    tag = "val" if (hash(("degm", i)) % 20 == 0) else "train"
    row(p[0], "degm", str(i), "train16", e, tag)
i = 0
for line in open(TBTSV):
    p = line.rstrip("\n").split("|")
    i += 1
    try:
        b = chess.Board(p[0]); next(iter(b.legal_moves)); ok = True
    except Exception:
        ok = False
    if not ok:
        row(p[0], "syzygy", str(i), "NONE", None, None, "illegal")
        continue
    e = routeB(p[0])
    tag = "val" if (hash(("tb", i)) % 20 == 0) else "train"
    row(p[0], "syzygy", str(i), "train16", e, tag)

# ---- game segments: both sig classes, skip-aware ----
rows = []
for fn in sorted(os.listdir(SEG)):
    if not (fn.startswith("pos_") and (fn.endswith(".clean.tsv") or fn.endswith(".draws.tsv"))):
        continue
    for line in open(f"{SEG}/{fn}", errors="ignore"):
        p = line.rstrip("\n").split("|")
        if len(p) < 8 or p[3] not in ("pos", "anti"):
            continue
        rows.append((int(p[7]), int(p[6]), p[0], p[5], p[3]))
print(f"[prov] segment rows: {len(rows):,}", flush=True)

# played-move replay (same as assemble16)
games = {}
for line in open(GAMES):
    f_ = line.rstrip("\n").split("\t")
    if f_[0] == "game_id":
        continue
    gid = int(f_[0])
    if gid in {r[0] for r in rows}:
        games[gid] = (int(f_[1]), int(f_[2]), f_[5])
pgn_f = open(PGN, "rb")
moves = {}
for gid, (off, ln, res) in games.items():
    pgn_f.seek(off)
    blob = pgn_f.read(ln if ln > 0 else 65536)
    movetext = b""
    for lne in blob.split(b"\n"):
        if not lne.startswith(b"[") and lne.strip():
            movetext += lne + b" "
    if not movetext.strip():
        continue
    try:
        g = chess.pgn.read_game(io.StringIO(
            '[Result "%s"]\n\n' % res + movetext.decode("utf-8", "replace")))
    except Exception:
        continue
    if g is None:
        continue
    b = g.board()
    for ply, mv in enumerate(g.mainline_moves(), start=1):
        moves[(gid, ply)] = mv.uci()

for gid, ply, fen, seg_id, sig in rows:
    mv = moves.get((gid, ply + 1))
    engine = "train16" if sig == "pos" else "anti16"
    if mv is None:
        row(fen, "segment", f"{gid}:{ply}", "NONE", None, None, "no_played_move")
        continue
    try:
        chess.Move.from_uci(mv)
    except Exception:
        row(fen, "segment", f"{gid}:{ply}", "NONE", None, None, "illegal")
        continue
    e = routeB(fen)
    tag = "val" if (hash(seg_id) % 20 == 0) else "train"
    row(fen, "segment", f"{gid}:{ply}", engine, e, tag)

# ---- miniatures (if labeled) ----
if os.path.exists(MINILBL):
    labels = {}
    for line in open(MINILBL):
        h, cp = line.rstrip("\n").split("\t")
        labels[h] = cp
    for line in open(MINI):
        fen, uci, gid, ply = line.rstrip("\n").split("\t")
        if fen not in labels:
            row(fen, "miniature", f"{gid}:{ply}", "NONE", None, None, "no_label")
            continue
        e = routeB(fen)
        tag = "val" if (hash(("mini", gid)) % 20 == 0) else "train"
        row(fen, "miniature", f"{gid}:{ply}", "train16", e, tag)
else:
    print("[prov] miniatures not labeled yet; rerun for those", flush=True)

out.close()
print(f"DONE {n:,} provenance rows -> {OUT}")
